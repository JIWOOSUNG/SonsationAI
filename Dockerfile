FROM python:3.10-slim

WORKDIR /app

COPY . .

# PYTHONPATH를 루트(/app)로 설정 → predict/main.py에서 상위폴더 import 가능
ENV PYTHONPATH=/app

RUN pip install --upgrade pip
RUN pip install -r requirements.txt

CMD ["uvicorn", "predict.main:app", "--host", "0.0.0.0", "--port", "8000"]
