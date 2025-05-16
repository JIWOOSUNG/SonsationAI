import sys
import os
import json
import torch
import numpy as np
from fastapi import FastAPI, Query
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware

# 실행 명령어 - uvicorn main:app —reload

# 🔧 상위 경로에 있는 모델 정의 파일 접근
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from signbert_model import SignBERT  # 정확한 구조 사용

# ✅ 경로 설정
MODEL_PATH = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "signbert_model.pth")
)
LABEL_PATH = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "label.json")
)
INPUT_DIM = 225
TARGET_FRAMES = 100

# ✅ FastAPI 앱 생성
app = FastAPI()

# ✅ CORS 설정 (클라이언트 요청 허용)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 개발 중이므로 모두 허용 (배포 시에는 제한 권장)
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ✅ label.json 불러오기
with open(LABEL_PATH, "r", encoding="utf-8") as f:
    sign_dict = json.load(f)
reverse_sign_dict = {v: k for k, v in sign_dict.items()}


# ✅ 요청 바디 정의
class RequestBody(BaseModel):
    sequence: list  # 100x225 landmark 시퀀스


# ✅ 모델 로드
if not os.path.exists(MODEL_PATH):
    raise ValueError(f"❌ 모델 파일이 존재하지 않습니다: {MODEL_PATH}")

# 체크포인트 불러오기
checkpoint = torch.load(MODEL_PATH, map_location="cpu")

# 🔁 label_map이 {label: index}일 경우 → {index: label}로 뒤집기
raw_map = checkpoint["label_map"]
label_map = {v: k for k, v in raw_map.items()}

# 모델 생성 및 가중치 로딩
model = SignBERT(
    input_dim=INPUT_DIM, num_classes=len(label_map), max_seq_len=TARGET_FRAMES
)
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()


# ✅ 예측 API 엔드포인트
@app.post("/predict/quiz")
def predict(req: RequestBody, sign: str = Query(...)):
    # 수어 단어 한글 -> 영문 변환
    expected_label = sign_dict.get(sign)
    if expected_label is None:
        return {
            "match": False,
            "message": f"❌ '{sign}'에 해당하는 영문 라벨이 없습니다.",
            "confidence": 0.0,
        }

    x = torch.tensor([req.sequence], dtype=torch.float32)
    print("sign: " + sign)

    with torch.no_grad():
        output = model(x)
        probs = torch.softmax(output, dim=1)
        confidence, pred = torch.max(probs, dim=1)

    label_idx = pred.item()
    predicted_label = label_map.get(label_idx, "unknown")
    confidence_val = float(confidence.item())

    # 매치 판단 기준 설정
    threshold = 0.97
    is_match = predicted_label == expected_label and confidence_val >= threshold

    return {
        "match": is_match,
        "message": (
            "🥳 예상한 단어와 일치합니다!"
            if is_match
            else "🤔 예상한 단어와 일치하지 않습니다\n다시 시도해보세요!"
        ),
        "confidence": round(confidence_val, 4),
    }


@app.post("/predict/learn")
def predict(req: RequestBody, sign: str = Query(...)):
    x = torch.tensor([req.sequence], dtype=torch.float32)

    # 수어 단어 한글 -> 영문 변환
    expected_label = sign_dict.get(sign)
    if expected_label is None:
        return {
            "match": False,
            "message": f"❌ '{sign}'에 해당하는 영문 라벨이 없습니다.",
            "confidence": 0.0,
        }

    with torch.no_grad():
        output = model(x)
        probs = torch.softmax(output, dim=1)
        confidence, pred = torch.max(probs, dim=1)

    # 매치 판단 기준 설정
    threshold = 0.97

    label_idx = pred.item()
    predicted_label = label_map.get(label_idx, "unknown")
    confidence_val = float(confidence.item())

    result = {
        "match": True,
        "message": (
            f"👍 현재 수어는 학습중인 '{sign}' 입니다"
            if predicted_label == expected_label and confidence_val >= threshold
            else (
                f"💡 현재 수어는 ''{reverse_sign_dict.get(predicted_label, predicted_label)}'' 입니다"
                if predicted_label != expected_label and confidence_val >= threshold
                else "❗️ 올바르지 않는 동작입니다! 다시 시도해주세요!"
            )
        ),
        "confidence": round(confidence_val, 4),
    }

    return result
