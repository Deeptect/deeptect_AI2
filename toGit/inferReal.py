import argparse
import numpy as np
import torch
import torch.nn as nn
import os
from feature_extractor import extract_feature_sequence_from_video  # 👈 추론용 feature 추출 함수 사용

# ============================
# 🧠 모델 정의
# ============================
class LSTMAttentionClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim=128, num_layers=2, num_classes=2):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.attn = nn.Linear(hidden_dim, 1)
        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        h_seq, _ = self.lstm(x)
        attn_weights = torch.softmax(self.attn(h_seq).squeeze(-1), dim=1)
        weighted_sum = torch.sum(h_seq * attn_weights.unsqueeze(-1), dim=1)
        out = self.fc(weighted_sum)
        return out

# ============================
# 🚀 단일 영상 추론 실행 (정답률 극대화 전략)
# ============================
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--video_path", type=str, required=True, help="추론할 동영상 파일 경로")
    parser.add_argument("--model_path", type=str, default="./best_model.pth", help="학습된 모델 경로")
    parser.add_argument("--threshold", type=float, default=0.9, help="Original로 간주할 최소 확률 (default: 0.9)")
    args = parser.parse_args()

    # 👁️ Feature 추출
    print(f"📹 영상 처리 중: {args.video_path}")
    seq = extract_feature_sequence_from_video(args.video_path)

    if seq is None:
        print("❌ 얼굴 인식 실패 또는 유효 프레임 없음 - 추론 중단")
        exit(1)

    # 🔢 입력 차원 및 모델 불러오기
    input_dim = seq.shape[1]
    model = LSTMAttentionClassifier(input_dim=input_dim)
    model.load_state_dict(torch.load(args.model_path, map_location="cpu"))

    # 🧠 추론
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    x = torch.tensor(seq, dtype=torch.float32).unsqueeze(0).to(device)  # (1, 256, D)

    with torch.no_grad():
        output = model(x)
        probs = torch.softmax(output, dim=1)
        original_prob = probs[0, 0].item()
        deepfake_prob = probs[0, 1].item()

        # ✅ 정답률 극대화: Original 확률이 높을 때만 Original로 판단
        if original_prob >= args.threshold:
            pred = 0  # Original
        else:
            pred = 1  # Deepfake

    print("\n✅ 추론 결과:")
    print(f"- Original 확률: {original_prob:.4f}")
    print(f"- Deepfake 확률: {deepfake_prob:.4f}")
    print(f"- 예측 결과: {'Original' if pred == 0 else 'Deepfake'} (Original 기준 임계값: {args.threshold})")
