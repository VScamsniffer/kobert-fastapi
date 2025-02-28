# Base image
FROM python:3.8-slim

# 필요한 시스템 패키지 설치
RUN apt-get update && apt-get install -y \
    build-essential \
    ffmpeg \
    git \
    curl \
    libssl-dev \
    pkg-config \
    && rm -rf /var/lib/apt/lists/*

# Rust 설치 (tokenizers 패키지 문제 해결)
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
ENV PATH="/root/.cargo/bin:${PATH}"

# 작업 디렉토리 설정
WORKDIR /app

# requirements.txt 파일 복사
COPY requirements.txt /app/

# pip 최신 버전 업데이트
RUN pip install --upgrade pip

# 의존성 설치
RUN pip install -r requirements.txt

# ✅ Azure Storage SDK 및 Whisper 설치 추가
RUN pip install azure-storage-blob openai-whisper


# 애플리케이션 코드 복사
COPY . /app

# 임시 파일 저장 폴더 생성
RUN mkdir -p /app/temp_files

# 포트 설정
EXPOSE 8090

# 실행 명령어
CMD ["sh", "-c", "python -m uvicorn main:app --host 0.0.0.0 --port 8090"]




# FROM python:3.8-slim
# WORKDIR /app
# RUN apt-get update && apt-get install -y \
#     build-essential \
#     ffmpeg \
#     git \
#     curl \
#     && curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y \
#     && . $HOME/.cargo/env \
#     && rm -rf /var/lib/apt/lists/*
# COPY requirements.txt /app/
# ENV PATH="/root/.cargo/bin:${PATH}"
# RUN pip install --no-cache-dir --upgrade pip && \
#     pip install --no-cache-dir -r requirements.txt && \
#     pip install git+https://github.com/SKTBrain/KoBERT.git@master

# COPY kobert_state_dict2.pth /app/
# COPY tokenizer_vocab.txt /app/
# COPY . /app/
# CMD ["python", "-m", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8090"]


# FROM python:3.8-slim
# WORKDIR /app
# RUN apt-get update && apt-get install -y \
#     build-essential \
#     ffmpeg \
#     git \
#     curl \
#     && curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y \
#     && . $HOME/.cargo/env \
#     && rm -rf /var/lib/apt/lists/*
# COPY requirements.txt /app/
# ENV PATH="/root/.cargo/bin:${PATH}"
# RUN pip install --no-cache-dir --upgrade pip && \
#     pip install --no-cache-dir -r requirements.txt && \
#     pip install git+https://github.com/SKTBrain/KoBERT.git@master

# COPY kobert_state_dict2.pth /app/
# COPY tokenizer_vocab.txt /app/
# COPY . /app/
# CMD ["python", "-m", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8090"]





# ✅ Git 설치 후 KoBERT 설치
# RUN pip install git+https://github.com/SKTBrain/KoBERT.git@master

# FROM python:3.8-slim-buster

# WORKDIR /app

# RUN apt-get update && apt-get install -y build-essential ffmpeg libsndfile1

# COPY fastapi/requirements.txt .
# RUN pip install --no-cache-dir -r requirements.txt

# COPY fastapi/ .

# ENV MODEL_PATH /app/kobert_state_dict2.pth

# EXPOSE 8090

# CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8090"]
