# deeptect_AI2

# 🧠 Deeptect_AI - Deepfake Detection Inference

본 프로젝트는 학습된 Deepfake 탐지 모델을 기반으로 추론을 실행하는 코드와 구조를 포함하고 있으며, 학습 데이터로는 FaceForensics++(FF++) 데이터셋을 사용했습니다.

## 📁 프로젝트 구성

├── best_model.pth # 학습된 LSTM Attention 모델  


├── feature_extractor.py # 영상에서 프레임별 얼굴 특징 벡터 추출  


├── inferReal.py # 단일 영상 추론 스크립트 (정답률 기준 예측)  


---

## 📦 1. Python 환경 및 의존성 설치

```bash

# 가상환경 생성
python -m venv deepfake-lstm

#가상환경 활성화
source deepfake-lstm/bin/activate      # Linux / macOS (bash/zsh)
deepfake-lstm\Scripts\activate.bat    # Windows CMD
deepfake-lstm\Scripts\Activate.ps1    # Windows PowerShell

#의존성 설치
pip install torch==2.7.0+cu118 torchvision==0.22.0+cu118 torchaudio==2.7.0+cu118 \
            opencv-python==4.11.0.86 mediapipe==0.10.21 numpy==1.26.4 tqdm

```



---

## ⬇️ 2. 기타  다운로드

모델 가중치 다운로드 생략 (폴더에 포함됨)
학습 코드, 학습 로그, 학습 데이터셋 등은 생략하고 업로드  

추론용 데이터셋은 따로 구조를 지킬 필요 없으며 추론 명령어 옵션을 참고하면 됨 

---

## ▶️ 3. 추론 코드 실행 예시

```bash
python inferReal.py \
  --video_path /path/to/video.mp4 \ 
  --model_path ./best_model.pth \
  --threshold 0.9
```
video_path: 비디오 경로, 설정 필수 
model_path: 설정 필수 아님 (디폴트값: ./best_model.pth)
threshold: 추론 결과 출력 임계값, 설정 필수 아님 (디폴트값: 0.9) 

예시 출력:
```
📹 영상 처리 중: /home/elicer/kgr/000_003.mp4
INFO: Created TensorFlow Lite XNNPACK delegate for CPU.
WARNING: All log messages before absl::InitializeLog() is called are written to STDERR
W0000 00:00:1748232068.177609   28629 inference_feedback_manager.cc:114] Feedback manager requires a model with a single signature inference. Disabling support for feedback tensors.
W0000 00:00:1748232068.195034   28629 inference_feedback_manager.cc:114] Feedback manager requires a model with a single signature inference. Disabling support for feedback tensors.
W0000 00:00:1748232068.198297   28628 landmark_projection_calculator.cc:186] Using NORM_RECT without IMAGE_DIMENSIONS is only supported for the square ROI. Provide IMAGE_DIMENSIONS or use PROJECTION_MATRIX.

✅ 추론 결과:
- Original 확률: 0.7639
- Deepfake 확률: 0.2361
- 예측 결과: Deepfake (Original 기준 임계값: 0.9)
```
경고 무시해도 됨 (추론에 영향 없음)

---
