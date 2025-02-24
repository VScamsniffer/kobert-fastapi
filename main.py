
import numpy as np
np.bool = bool
import torch
from transformers import BertModel
from transformers import BertTokenizer
from fastapi import FastAPI,UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
import os

from kobert_tokenizer import KoBERTTokenizer
from kobert.pytorch_kobert import get_pytorch_kobert_model
import torch.nn as nn
import whisper  # Whisper 추가
from pydub import AudioSegment
from pydub.utils import which
import logging


app = FastAPI()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# 아래코드를 통해 먼저 모델을 로드하여 캐쉬를 받아온 후, 코드 진행해야함
# https://github.com/SKTBrain/KoBERT/tree/master/kobert_hf
# tokenizer = KoBERTTokenizer.from_pretrained('skt/kobert-base-v1')
# tokenizer.encode("한국어 모델을 공유합니다.")


# model = BertModel.from_pretrained('skt/kobert-base-v1')
# text = "한국어 모델을 공유합니다."
# inputs = tokenizer.batch_encode_plus([text])
# out = model(input_ids = torch.tensor(inputs['input_ids']),
#               attention_mask = torch.tensor(inputs['attention_mask']))
# out.pooler_output.shape
# torch.Size([1, 768])

# model, vocab = get_pytorch_kobert_model()
# print("모델 로드 완료:", model is not None)


# model, vocab = get_pytorch_kobert_model()
# torch.save(model, "kobert.pkl")
# print("모델 로드 완료:",model is not None)

TEMP_DIR = os.path.join(os.getcwd(), "temp_files")
os.makedirs(TEMP_DIR, exist_ok=True)  # 폴더 없으면 생성

# BERTClassifier 모델 정의
class BERTClassifier(nn.Module):
    def __init__(self, hidden_size=768, num_classes=2):
        super(BERTClassifier, self).__init__()
        self.bert, _ = get_pytorch_kobert_model()
        self.classifier = torch.nn.Linear(768, 1)

    def forward(self, token_ids, valid_length, segment_ids):
        _, pooled_output = self.bert(input_ids=token_ids, return_dict=False)
        return self.classifier(pooled_output)
    

MODEL_PATH = os.path.join(os.path.dirname(__file__), "kobert_state_dict21.pth")
model = BERTClassifier()
model.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device("cpu")))
model.eval()
tokenizer = BertTokenizer(vocab_file="tokenizer_vocab.txt", do_lower_case=False)
print(f"모델 로드 완료:{MODEL_PATH}",model is not None)


# 윈도우의 경우 경로 설정후 직접 불러오기
# FFMPEG_PATH = r"User\parksojin\Download\ffmpeg-master-latest-win64-gpl-shared\ffmpeg.exe"
# AudioSegment.converter = FFMPEG_PATH


#mac의 경우 ffmpeg 로컬설치 후 which 로 가져옴
AudioSegment.converter = which("ffmpeg")
print(f"🔧 FFmpeg 설정 완료")

whisper_model = whisper.load_model("base")


@app.post("/upload-audio/")
async def upload_audio_file(file: UploadFile = File(...)):
    """음성 파일을 업로드하고 변환 후 분석"""
    logger.info("[요청] 파일 업로드 및 분석 요청")

    # 파일 확장자 확인
    ext = os.path.splitext(file.filename)[1].lower()
    allowed_extensions = [".mp3", ".wav", ".ogg", ".m4a"]

    if ext not in allowed_extensions:
        raise HTTPException(status_code=400, detail="지원되지 않는 파일 형식입니다.")

    try:
        # 🔹 파일을 TEMP_DIR에 저장
        temp_audio_path = os.path.join(TEMP_DIR, file.filename)
        with open(temp_audio_path, "wb") as f:
            contents = await file.read()
            f.write(contents)

        # 🔹 변환: mp3, ogg → wav
        wav_file_path = os.path.join(TEMP_DIR, os.path.splitext(file.filename)[0] + ".wav")
        logger.info(f"[🎙️ 변환] {ext} → WAV 변환 중...")

        try:
            audio = AudioSegment.from_file(temp_audio_path, format=ext[1:])
            audio.export(wav_file_path, format="wav")
        except Exception as e:
            logger.error(f"🚨 [오류] FFmpeg 변환 실패: {str(e)}")
            raise HTTPException(status_code=500, detail="FFmpeg 변환 실패")

        # 🔹 변환 후 파일 확인
        if not os.path.exists(wav_file_path):
            logger.error(f"🚨 [오류] WAV 파일이 존재하지 않음: {wav_file_path}")
            raise HTTPException(status_code=500, detail="WAV 변환 후 파일이 존재하지 않습니다.")

        logger.info(f"[STT 시작]: {wav_file_path}")

        # 🔹 STT 실행
        text = audio_to_text(wav_file_path)

        if text.startswith("Whisper 변환 실패"):
            raise HTTPException(status_code=500, detail=text)

        logger.info(f"[분석할 텍스트]: {text}")

        probability = analyze_text(text) * 100

        # 🔹 임시 파일 삭제 (필요 시 주석 해제)
        # os.remove(temp_audio_path)
        # os.remove(wav_file_path)

        return {"probability": probability, "text": text}

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"🚨 [ERROR] 업로드 중 오류 발생: {str(e)}")
        raise HTTPException(status_code=500, detail="파일 업로드 중 오류가 발생했습니다.")

def audio_to_text(wav_file_path):
    """음성 파일을 텍스트로 변환하는 함수"""
    if not os.path.exists(wav_file_path):
        logger.error(f"🚨 [오류] 파일이 존재하지 않음: {wav_file_path}")
        return "Whisper 변환 실패: 파일 없음"

    logger.info(f"[🎙️ STT 시작]: {wav_file_path}")
    try:
        result = whisper_model.transcribe(wav_file_path)
        logger.info(result)
        return result["text"]
    except Exception as e:
        logger.error(f"🚨 [오류] Whisper 변환 실패: {str(e)}")
        return f"Whisper 변환 실패: {str(e)}"

def analyze_text(text):
    """KoBERT 모델을 사용해 텍스트 분석"""
    try:
        inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
        valid_length = torch.tensor([len(inputs["input_ids"][0])])  # 문장 길이
        segment_ids = torch.zeros_like(inputs["input_ids"])  # 모든 단어의 segment ID를 0으로 설정

        with torch.no_grad():
            output = model(inputs["input_ids"], valid_length, segment_ids)
        
        val = output.squeeze(1)
        chk = torch.sigmoid(val)
        chk = chk.item()

        return chk
    
    except Exception as e:
        logger.error(f"🚨 [ERROR] 분석 중 오류 발생: {str(e)}")
        return 0.0  # 오류 시 기본값 반환

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)