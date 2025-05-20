import sys
import os
import torch
import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware

# 🔧 상위 경로에 있는 모델 정의 파일 접근
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from signbert_model import SignBERT  # 정확한 구조 사용

# ✅ 경로 설정
MODEL_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "signbert_model.pth"))
INPUT_DIM = 225
TARGET_FRAMES = 100

# ✅ FastAPI 앱 생성
app = FastAPI()

# ✅ CORS 설정 (클라이언트 요청 허용)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://sonsation.netlify.app"],  # 개발 중이므로 모두 허용 (배포 시에는 제한 권장)
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ✅ 요청 바디 정의
class SequenceInput(BaseModel):
    sequence: list  # 100x225 landmark 시퀀스

# ✅ 모델 로드
if not os.path.exists(MODEL_PATH):
    raise ValueError(f"❌ 모델 파일이 존재하지 않습니다: {MODEL_PATH}")

checkpoint = torch.load(MODEL_PATH, map_location='cpu')

# 🔁 label_map 뒤집기 (index → label)
raw_map = checkpoint['label_map']
label_map = {v: k for k, v in raw_map.items()}

# ✅ 모델 정의 및 로드
model = SignBERT(input_dim=INPUT_DIM, num_classes=len(label_map), max_seq_len=TARGET_FRAMES)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# ✅ 예측 API
@app.post("/predict")
def predict(req: SequenceInput):
    try:
        sequence = np.array(req.sequence).astype(np.float32)

        if sequence.shape != (TARGET_FRAMES, INPUT_DIM):
            return {"error": f"입력 시퀀스 shape 오류: {sequence.shape}, 기대값: ({TARGET_FRAMES}, {INPUT_DIM})"}

        x = torch.tensor([sequence], dtype=torch.float32)  # (1, 100, 225)

        with torch.no_grad():
            output = model(x)
            probs = torch.softmax(output, dim=1)
            confidence, pred = torch.max(probs, dim=1)

        label_idx = pred.item()
        label = label_map.get(label_idx, "unknown")

        return {
            "label": label,
            "confidence": float(confidence.item())
        }

    except Exception as e:
        return {"error": f"예측 실패: {str(e)}"}
