from celery import Celery
import numpy as np
np.bool = bool
import logging
from celery_setting import celery_app
import torch
from transformers import BertTokenizer
from kobert.pytorch_kobert import get_pytorch_kobert_model
import torch.nn as nn
import whisper
import torch
import os


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Celery 설정
REDIS_URL = "redis://localhost:6379/0"
app = Celery("tasks", broker=REDIS_URL, backend=REDIS_URL)

# 모델 로드
MODEL_PATH = os.path.join(os.path.dirname(__file__), "kobert_state_dict2.pth")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
whisper_model = whisper.load_model("base", device=device)


class BERTClassifier(nn.Module):
    def __init__(self, hidden_size=768, num_classes=2):
        super(BERTClassifier, self).__init__()
        self.bert, _ = get_pytorch_kobert_model()
        self.classifier = torch.nn.Linear(768, 1)

    def forward(self, token_ids, valid_length, segment_ids):
        _, pooled_output = self.bert(input_ids=token_ids, return_dict=False)
        return self.classifier(pooled_output)

# 모델 및 토크나이저 초기화
model = BERTClassifier()
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.to(device)
model.eval()

tokenizer = BertTokenizer(vocab_file="tokenizer_vocab.txt", do_lower_case=False)

@app.task
def predict_text(text):
    """텍스트를 받아서 확률 예측"""
    if not isinstance(text, str):  # text가 str인지 체크
        raise ValueError("text input must be of type `str`")
    
    inputs = tokenizer(
        text,
        padding=True,
        truncation=True,
        max_length=512,
        return_tensors="pt"
    )
    logger.info(f"text type: {type(text)}")
    logger.info(f"text value: {text}")


    # 디바이스로 이동
    inputs = {k: v.to(device) for k, v in inputs.items()}
    valid_lengths = torch.tensor([len(ids) for ids in inputs["input_ids"]]).to(device)
    segment_ids = torch.zeros_like(inputs["input_ids"]).to(device)

    # 추론
    with torch.no_grad():
        outputs = model(inputs["input_ids"], valid_lengths, segment_ids)
        probabilities = torch.sigmoid(outputs.squeeze(1)).cpu().numpy().tolist()
    print(outputs)  # 출력값 확인
    return probabilities

# 배치 사이즈 설정
BATCH_SIZE = 32

@app.task
def batch_predict_text(texts):
    """배치 텍스트를 받아서 확률 예측"""
    batch_results = []

    # Celery 작업을 비동기적으로 실행
    from celery_setting import group

    # predict_text 작업을 비동기적으로 호출
    result = group(predict_text.s(text) for text in texts)  # predict_text를 비동기적으로 호출
    results = result.apply_async()  # 실제 작업 실행
    
    # 결과 반환
    batch_results = results.get()  # 결과를 기다림

    return batch_results  # 배치 처리 후 결과 반환


@app.task
def transcribe_audio(wav_file_path: str) -> str:
    """Whisper STT 변환"""
    try:
        result = whisper_model.transcribe(
            wav_file_path,
            fp16=False,
            language='ko'
        )
        return result["text"]
    except Exception as e:
        return f"Whisper 변환 실패: {str(e)}"
