import numpy as np
np.bool = bool
import torch
from transformers import BertModel, BertTokenizer
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
import os
from kobert.pytorch_kobert import get_pytorch_kobert_model
import torch.nn as nn
import whisper
from pydub import AudioSegment
from fastapi.middleware.cors import CORSMiddleware
import logging
from typing import List
from functools import lru_cache
from collections import deque
import asyncio
from datetime import datetime
from azure.storage.blob import BlobServiceClient

app = FastAPI()

# ✅ Azure 환경 변수 설정
AZURE_ACCOUNT_NAME = os.getenv("AZURE_ACCOUNT_NAME")
AZURE_ACCOUNT_KEY = os.getenv("AZURE_ACCOUNT_KEY")
AZURE_CONTAINER = os.getenv("AZURE_CONTAINER", "user_file")
AZURE_CONNECTION_STRING = f"DefaultEndpointsProtocol=https;AccountName={AZURE_ACCOUNT_NAME};AccountKey={AZURE_ACCOUNT_KEY};EndpointSuffix=core.windows.net"

# ✅ Azure Blob Storage 클라이언트 설정
blob_service_client = BlobServiceClient.from_connection_string(AZURE_CONNECTION_STRING)
container_client = blob_service_client.get_container_client(AZURE_CONTAINER)



# ✅ CORS 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3001", "http://localhost:3000", "https://vscamsniffer.work.gd", "https://4.230.156.117:443"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ✅ 배치 처리 설정
BATCH_SIZE = 32
BATCH_TIMEOUT = 0.3
MAX_QUEUE_SIZE = 100

# ✅ 배치 큐 및 이벤트
class BatchProcessor:
    def __init__(self):
        self.queue = deque()
        self.event = asyncio.Event()
        self.processing = False
        self.last_process_time = datetime.now()

    async def add_to_queue(self, item):
        if len(self.queue) >= MAX_QUEUE_SIZE:
            raise HTTPException(status_code=503, detail="서버가 너무 많은 요청을 처리 중입니다.")
        
        future = asyncio.Future()
        self.queue.append((item, future))
        
        if not self.processing:
            asyncio.create_task(self.process_batch())
        
        return await future

    async def process_batch(self):
        self.processing = True
        while self.queue:
            current_time = datetime.now()
            if (len(self.queue) < BATCH_SIZE and 
                (current_time - self.last_process_time).total_seconds() < BATCH_TIMEOUT):
                await asyncio.sleep(0.01)
                continue
                
            batch = []
            futures = []
            batch_size = min(BATCH_SIZE, len(self.queue))
            
            for _ in range(batch_size):
                item, future = self.queue.popleft()
                batch.append(item)
                futures.append(future)
            
            try:
                results = await process_text_batch(batch)
                for future, result in zip(futures, results):
                    future.set_result(result)
                    
            except Exception as e:
                for future in futures:
                    future.set_exception(e)
            
            self.last_process_time = datetime.now()
        
        self.processing = False

# ✅ BERT 모델 클래스 정의
class BERTClassifier(nn.Module):
    def __init__(self, hidden_size=768, num_classes=2):
        super(BERTClassifier, self).__init__()
        self.bert, _ = get_pytorch_kobert_model()
        self.classifier = torch.nn.Linear(768, 1)

    def forward(self, token_ids, valid_length, segment_ids):
        _, pooled_output = self.bert(input_ids=token_ids, return_dict=False)
        return self.classifier(pooled_output)

# ✅ 모델 초기화
MODEL_PATH = os.path.join(os.path.dirname(__file__), "kobert_state_dict4.pth")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

TEMP_DIR = "/app/temp_files"
os.makedirs(TEMP_DIR, exist_ok=True)

@lru_cache(maxsize=1)
def load_models():
    model = BERTClassifier()
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.to(device)
    model.eval()
    
    tokenizer = BertTokenizer(vocab_file="tokenizer_vocab.txt", do_lower_case=False)
    whisper_model = whisper.load_model("base", device=device)
    
    return model, tokenizer, whisper_model

model, tokenizer, whisper_model = load_models()
batch_processor = BatchProcessor()

# ✅ 배치 처리 함수
async def process_text_batch(texts: List[str]) -> List[float]:
    try:
        inputs = tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt"
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}
        valid_lengths = torch.tensor([len(ids) for ids in inputs["input_ids"]]).to(device)
        segment_ids = torch.zeros_like(inputs["input_ids"]).to(device)

        with torch.no_grad():
            outputs = model(inputs["input_ids"], valid_lengths, segment_ids)
            probabilities = torch.sigmoid(outputs.squeeze(1)).cpu().numpy().tolist()

        return probabilities
    except Exception as e:
        logger.error(f"🚨 [ERROR] 배치 처리 중 오류 발생: {str(e)}")
        return [0.0] * len(texts)

@app.post("/upload-audio/")
async def upload_audio_file(file: UploadFile = File(...)):
    """음성 파일을 업로드하고 변환 후 분석"""
    logger.info("[요청] 파일 업로드 및 분석 요청")

    ext = os.path.splitext(file.filename)[1].lower()
    allowed_extensions = [".mp3", ".wav", ".ogg", ".m4a"]

    if ext not in allowed_extensions:
        raise HTTPException(status_code=400, detail="지원되지 않는 파일 형식입니다.")

    try:
        unique_filename = f"{datetime.now().strftime('%Y%m%d%H%M%S')}-{file.filename}"
        temp_audio_path = os.path.join(TEMP_DIR, unique_filename)
        
        # ✅ FastAPI 컨테이너 내부에 저장
        with open(temp_audio_path, "wb") as f:
            f.write(await file.read())

        # ✅ Azure Storage에 업로드
        blob_client = container_client.get_blob_client(f"user_file/{unique_filename}")
        with open(temp_audio_path, "rb") as data:
            blob_client.upload_blob(data, overwrite=True)

        # ✅ WAV 변환 (초기화 포함)
        wav_file_path = temp_audio_path  # 기본적으로 원본 파일을 사용
        if ext != ".wav":
            audio = AudioSegment.from_file(temp_audio_path, format=ext[1:])
            wav_file_path = os.path.join(TEMP_DIR, f"{os.path.splitext(unique_filename)[0]}.wav")
            audio.export(wav_file_path, format="wav", parameters=["-ac", "1", "-ar", "16000"])

        # ✅ STT 수행
        text = await audio_to_text(wav_file_path)
        logger.info(f"[분석할 텍스트]: {text}")

        # ✅ 배치 처리 큐에 추가
        probability = await batch_processor.add_to_queue(text)

        # ✅ 임시 파일 삭제
        try:
            os.remove(temp_audio_path)
            if ext != ".wav" and os.path.exists(wav_file_path):
                os.remove(wav_file_path)
        except Exception as e:
            logger.warning(f"임시 파일 삭제 실패: {e}")

        return {"probability": probability * 100, "text": text}

    except Exception as e:
        logger.error(f"🚨 [ERROR] 업로드 중 오류 발생!: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


async def audio_to_text(wav_file_path: str) -> str:
    if not os.path.exists(wav_file_path):
        raise FileNotFoundError(f"File not found: {wav_file_path}")

    try:
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            app.state.thread_pool, 
            lambda: whisper_model.transcribe(wav_file_path, fp16=False, language='ko')
        )
        return result["text"]
    except Exception as e:
        logger.error(f"🚨 [오류] Whisper 변환 실패: {str(e)}")
        return f"Whisper 변환 실패: {str(e)}"
