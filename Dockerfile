# 1. Python 3.10 기반 이미지 사용
FROM python:3.10-slim

# 2. 시스템 패키지 설치 (OpenCV, MediaPipe용 필수)
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# 3. 작업 디렉토리 설정
WORKDIR /app

# 4. requirements.txt 먼저 복사 후 설치
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 5. 전체 코드 복사
COPY . .

# 6. Render가 제공하는 PORT 환경변수 사용하여 uvicorn 실행
CMD ["uvicorn", "predict.main:app", "—host", "0.0.0.0", "—port", "10000"]