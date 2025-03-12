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
from tasks import predict_text, transcribe_audio
from celery.result import AsyncResult
from fastapi import Path
from celery_setting import celery_app

app = FastAPI()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# mac의 경우 ffmpeg 로컬설치 후 which로 가져옴
AudioSegment.converter = which("ffmpeg")
print(f"🔧 FFmpeg 설정 완료")

TEMP_DIR = os.path.join(os.getcwd(), "temp_files")
os.makedirs(TEMP_DIR, exist_ok=True)

@app.get("/task-result/{task_id}")
def get_result(task_id: str):
    """Celery 작업 결과 조회"""
    result = AsyncResult(task_id, app=celery_app)  # Celery 앱 사용
    if result.ready():
        return {"status": "Completed", "result": result.result}
    return {"status": "Pending"}

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
            logger.info(f"WAV file path: {wav_file_path}")
            logger.info(f"WAV file exists: {os.path.exists(wav_file_path)}")
            logger.info(f"WAV file size: {os.path.getsize(wav_file_path)}")
        else:
            wav_file_path = temp_audio_path

        # STT 수행
        text = await audio_to_text(wav_file_path)
        if isinstance(text, str):
            logger.info(f"[분석할 텍스트 첫번째]: {text}")
        else:
            logger.warning(f"[경고] 예상치 못한 값 반환: {text},타입: {type(text)}")

        if isinstance(text, str) and text.startswith("Whisper 변환 실패"):
            raise HTTPException(status_code=500, detail=text)
        
        logger.info(f"[분석할 텍스트 두번째]: {text}")

        task = predict_text.delay(text)  # Celery Task 실행
        
        return {"task_id": task.id, "text": text}  # 비동기 작업 ID 반환

    except Exception as e:
        #여기서도 에러발생 
        logger.error(f"🚨 [ERROR] 업로드 중 오류 발생: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

async def audio_to_text(wav_file_path: str) -> str:
    """오디오 분석"""
    if not os.path.exists(wav_file_path):
        raise FileNotFoundError(f"File not found: {wav_file_path}")

    try:
        task = transcribe_audio.delay(wav_file_path)  # Celery 작업 요청
        result = task.get(timeout=300)  # Celery 작업 결과를 기다림 (timeout을 설정)
        if isinstance(result, str):
            return result  # 텍스트가 str이라면 반환
        else:
            raise ValueError(f"Expected string result but got {type(result)}")
    except Exception as e:
        #여기서 한번더 오류가 걸림
        logger.error(f"🚨 [오류] Whisper 변환 실패: {str(e)}")
        return f"Whisper 변환 실패: {str(e)}"


@app.on_event("startup")
async def startup_event():
    """사이트 시작시 모델링 대기시키기"""
    import concurrent.futures
    app.state.thread_pool = concurrent.futures.ThreadPoolExecutor(max_workers=4)
    
    # 더미데이터로 모델링 대기시키기
    # transcribe_audio.delay("dummy.wav")
    predict_text.delay("안녕하세요")
    logger.info("Celery models warmed up successfully")

@app.on_event("shutdown")
async def shutdown_event():
    """스레드 정리"""
    app.state.thread_pool.shutdown()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)


# import numpy as np
# np.bool = bool
# import torch
# from transformers import BertModel, BertTokenizer
# from fastapi import FastAPI, UploadFile, File, HTTPException
# from fastapi.responses import JSONResponse
# import os
# from kobert_tokenizer import KoBERTTokenizer
# from kobert.pytorch_kobert import get_pytorch_kobert_model
# import torch.nn as nn
# import whisper
# from pydub import AudioSegment
# from pydub.utils import which
# import logging
# from typing import Optional, List
# from functools import lru_cache
# from collections import deque
# import asyncio
# from datetime import datetime, timedelta
# import asyncio
# from tasks import predict_text, transcribe_audio
# from celery.result import AsyncResult
# from fastapi import Path
# from celery_setting import celery_app



# app = FastAPI()
# logging.basicConfig(level=logging.INFO)
# logger = logging.getLogger(__name__)

# #mac의 경우 ffmpeg 로컬설치 후 which 로 가져옴
# AudioSegment.converter = which("ffmpeg")
# print(f"🔧 FFmpeg 설정 완료")

# TEMP_DIR = os.path.join(os.getcwd(), "temp_files")
# os.makedirs(TEMP_DIR, exist_ok=True)


# @app.get("/task-result/{task_id}")
# def get_result(task_id: str):
#     """Celery 작업 결과 조회"""
#     result = AsyncResult(task_id, app=celery_app)  # Celery 앱 사용
#     if result.ready():
#         return {"status": "Completed", "result": result.result}
#     return {"status": "Pending"}

# @app.post("/upload-audio/")
# async def upload_audio_file(file: UploadFile = File(...)):
#     """음성 파일을 업로드하고 변환 후 분석"""
#     logger.info("[요청] 파일 업로드 및 분석 요청")

#     ext = os.path.splitext(file.filename)[1].lower()
#     allowed_extensions = [".mp3", ".wav", ".ogg", ".m4a"]

#     if ext not in allowed_extensions:
#         raise HTTPException(status_code=400, detail="지원되지 않는 파일 형식입니다.")

#     try:
#         # 파일 처리
#         temp_filename = f"{os.urandom(8).hex()}{ext}"
#         temp_audio_path = os.path.join(TEMP_DIR, temp_filename)
        
#         contents = await file.read()
#         with open(temp_audio_path, "wb") as f:
#             f.write(contents)

#         # WAV 변환
#         wav_file_path = os.path.join(TEMP_DIR, f"{os.path.splitext(temp_filename)[0]}.wav")
#         if ext != '.wav':
#             logger.info(f"[🎙️ 변환] {ext} → WAV 변환 중...")
#             audio = AudioSegment.from_file(temp_audio_path, format=ext[1:])
#             audio.export(wav_file_path, format="wav", parameters=["-ac", "1", "-ar", "16000"])
#         else:
#             wav_file_path = temp_audio_path

#         # STT 수행
#         text = await audio_to_text(wav_file_path)
#         if isinstance(text, str):
#             logger.info(f"[분석할 텍스트]: {text}")
#         else:
#             logger.warning(f"[경고] 예상치 못한 값 반환: {text}")

#         if isinstance(text, str) and text.startswith("Whisper 변환 실패"):
#             raise HTTPException(status_code=500, detail=text)
        
#         logger.info(f"[분석할 텍스트]: {text}")

#         task = predict_text.delay(text)  # Celery Task 실행
        
#         return {"task_id": task.id, "text": text}  # 비동기 작업 ID 반환

#         # 임시 파일 정리
#         # try:
#         #     os.remove(temp_audio_path)
#         #     if ext != '.wav':
#         #         os.remove(wav_file_path)
#         # except Exception as e:
#         #     logger.warning(f"임시 파일 삭제 실패: {e}")

#         # return {"probability": probability * 100, "text": text}

#     except Exception as e:
#         logger.error(f"🚨 [ERROR] 업로드 중 오류 발생: {str(e)}")
#         raise HTTPException(status_code=500, detail=str(e))

# async def audio_to_text(wav_file_path: str) -> str:
#     """오디오 분석"""
#     if not os.path.exists(wav_file_path):
#         raise FileNotFoundError(f"File not found: {wav_file_path}")

#     try:
#         task = transcribe_audio.delay(wav_file_path)  # Celery 작업 요청
#         return {"task_id": task.id}
#     except Exception as e:
#         logger.error(f"🚨 [오류] Whisper 변환 실패: {str(e)}")
#         return f"Whisper 변환 실패: {str(e)}"

# @app.on_event("startup")
# async def startup_event():
#     """사이트 시작시 모델링 대기시키기"""
#     import concurrent.futures
#     app.state.thread_pool = concurrent.futures.ThreadPoolExecutor(max_workers=4)
    
#     # 더미데이터로 모델링 대기시키기
#     transcribe_audio.delay("dummy.wav")
#     predict_text.delay("안녕하세요")
#     logger.info("Celery models warmed up successfully")

# @app.on_event("shutdown")
# async def shutdown_event():
#     """스레드 정리"""
#     app.state.thread_pool.shutdown()

# if __name__ == "__main__":
#     import uvicorn
#     uvicorn.run(app, host="127.0.0.1", port=8000)

# @app.get("/task-result/{task_id}")
# def get_result(task_id: str = Path(...)):
#     """Celery 작업 결과 조회"""
#     result = AsyncResult(task_id)
#     if result.ready():
#         return {"status": "Completed", "result": result.result}
#     return {"status": "Pending"}
