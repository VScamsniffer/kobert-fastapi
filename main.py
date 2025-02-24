import numpy as np
np.bool = bool
import torch
from transformers import BertModel, BertTokenizer
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
import os
from kobert_tokenizer import KoBERTTokenizer
from kobert.pytorch_kobert import get_pytorch_kobert_model
import torch.nn as nn
import whisper
from pydub import AudioSegment
from pydub.utils import which
import logging
from typing import Optional, List
from functools import lru_cache
from collections import deque
import asyncio
from datetime import datetime, timedelta
import asyncio

app = FastAPI()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 배치 처리를 위한 설정
BATCH_SIZE = 32
BATCH_TIMEOUT = 0.3  # 초
MAX_QUEUE_SIZE = 100

# 배치 처리를 위한 큐와 이벤트
class BatchProcessor:
    def __init__(self):
        self.queue = deque()
        self.event = asyncio.Event()
        self.processing = False
        self.last_process_time = datetime.now()

    async def add_to_queue(self, item):
        if len(self.queue) >= MAX_QUEUE_SIZE:
            raise HTTPException(status_code=503, detail="서버가 너무 많은 요청을 처리중입니다")
        
        future = asyncio.Future()
        self.queue.append((item, future))
        
        if not self.processing:
            asyncio.create_task(self.process_batch())
        
        return await future

    async def process_batch(self):
        self.processing = True
        
        while self.queue:
            current_time = datetime.now()
            
            # 배치 크기나 시간 초과에 도달할 때까지 대기
            if (len(self.queue) < BATCH_SIZE and 
                (current_time - self.last_process_time).total_seconds() < BATCH_TIMEOUT):
                await asyncio.sleep(0.01)
                continue
                
            # 현재 배치 처리
            batch = []
            futures = []
            batch_size = min(BATCH_SIZE, len(self.queue))
            
            for _ in range(batch_size):
                item, future = self.queue.popleft()
                batch.append(item)
                futures.append(future)
            
            try:
                # 배치 처리 실행
                results = await process_text_batch(batch)
                
                # 결과 반환
                for future, result in zip(futures, results):
                    future.set_result(result)
                    
            except Exception as e:
                # 에러 처리
                for future in futures:
                    future.set_exception(e)
            
            self.last_process_time = datetime.now()
        
        self.processing = False

# 기존 BERTClassifier 클래스는 동일하게 유지
class BERTClassifier(nn.Module):
    def __init__(self, hidden_size=768, num_classes=2):
        super(BERTClassifier, self).__init__()
        self.bert, _ = get_pytorch_kobert_model()
        self.classifier = torch.nn.Linear(768, 1)

    def forward(self, token_ids, valid_length, segment_ids):
        _, pooled_output = self.bert(input_ids=token_ids, return_dict=False)
        return self.classifier(pooled_output)

# 모델 초기화
MODEL_PATH = os.path.join(os.path.dirname(__file__), "kobert_state_dict2.pth")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

TEMP_DIR = os.path.join(os.getcwd(), "temp_files")
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

# 배치 처리를 위한 함수
async def process_text_batch(texts: List[str]) -> List[float]:
    """배치로 텍스트 분석 처리"""
    try:
        # 토큰화
        inputs = tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt"
        )
        
        # 디바이스로 이동
        inputs = {k: v.to(device) for k, v in inputs.items()}
        valid_lengths = torch.tensor([len(ids) for ids in inputs["input_ids"]]).to(device)
        segment_ids = torch.zeros_like(inputs["input_ids"]).to(device)

        # 배치 추론
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
        # 파일 처리
        temp_filename = f"{os.urandom(8).hex()}{ext}"
        temp_audio_path = os.path.join(TEMP_DIR, temp_filename)
        
        contents = await file.read()
        with open(temp_audio_path, "wb") as f:
            f.write(contents)

        # WAV 변환
        wav_file_path = os.path.join(TEMP_DIR, f"{os.path.splitext(temp_filename)[0]}.wav")
        if ext != '.wav':
            logger.info(f"[🎙️ 변환] {ext} → WAV 변환 중...")
            audio = AudioSegment.from_file(temp_audio_path, format=ext[1:])
            audio.export(wav_file_path, format="wav", parameters=["-ac", "1", "-ar", "16000"])
        else:
            wav_file_path = temp_audio_path

        # STT 수행
        text = await audio_to_text(wav_file_path)
        if isinstance(text, str) and text.startswith("Whisper 변환 실패"):
            raise HTTPException(status_code=500, detail=text)

        logger.info(f"[분석할 텍스트]: {text}")

        # 배치 처리 큐에 추가
        probability = await batch_processor.add_to_queue(text)

        # 임시 파일 정리
        try:
            os.remove(temp_audio_path)
            if ext != '.wav':
                os.remove(wav_file_path)
        except Exception as e:
            logger.warning(f"임시 파일 삭제 실패: {e}")

        return {"probability": probability * 100, "text": text}

    except Exception as e:
        logger.error(f"🚨 [ERROR] 업로드 중 오류 발생: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

async def audio_to_text(wav_file_path: str) -> str:
    """오디오 분석"""
    if not os.path.exists(wav_file_path):
        raise FileNotFoundError(f"File not found: {wav_file_path}")

    try:
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            app.state.thread_pool, 
            lambda: whisper_model.transcribe(
                wav_file_path,
                fp16=False,
                language='ko'
            )
        )
        return result["text"]
    except Exception as e:
        logger.error(f"🚨 [오류] Whisper 변환 실패: {str(e)}")
        return f"Whisper 변환 실패: {str(e)}"

@app.on_event("startup")
async def startup_event():
    """사이트 시작시 모델링 대기시키기"""
    import concurrent.futures
    app.state.thread_pool = concurrent.futures.ThreadPoolExecutor(max_workers=4)
    
    # 더미데이터로 모델링 대기시키기
    dummy_text = "안녕하세요"
    await batch_processor.add_to_queue(dummy_text)
    logger.info("Models warmed up successfully")

@app.on_event("shutdown")
async def shutdown_event():
    """스레드 정리"""
    app.state.thread_pool.shutdown()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
