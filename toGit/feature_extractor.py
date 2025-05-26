import cv2
import mediapipe as mp
import numpy as np
import math

# === Landmark 인덱스 정의 (학습용과 동일하게 유지) ===
left_eye_indices = [33, 160, 158, 133, 153, 144]
right_eye_indices = [263, 387, 385, 362, 373, 380]
left_iris_ids = [474, 475, 476, 477]
right_iris_ids = [469, 470, 471, 472]
eye_landmark_ids = [
    33, 160, 158, 133, 153, 144, 163, 7,
    263, 387, 385, 362, 373, 380, 390, 249,
    474, 475, 476, 477,
    469, 470, 471, 472
]

# === 특징 계산 함수들 (학습용과 동일하게 복사) ===
def calculate_ear(landmarks, indices):
    A = math.dist([landmarks[indices[1]].x, landmarks[indices[1]].y],
                  [landmarks[indices[5]].x, landmarks[indices[5]].y])
    B = math.dist([landmarks[indices[2]].x, landmarks[indices[2]].y],
                  [landmarks[indices[4]].x, landmarks[indices[4]].y])
    C = math.dist([landmarks[indices[0]].x, landmarks[indices[0]].y],
                  [landmarks[indices[3]].x, landmarks[indices[3]].y])
    return (A + B) / (2.0 * C)

def calculate_center(landmarks, indices):
    x = sum([landmarks[i].x for i in indices]) / len(indices)
    y = sum([landmarks[i].y for i in indices]) / len(indices)
    return (x, y)

def classify_gaze(x_diff, y_diff, threshold=0.01):
    direction = ""
    if abs(x_diff) < threshold and abs(y_diff) < threshold:
        direction = "center"
    else:
        if y_diff < -threshold:
            direction += "up"
        elif y_diff > threshold:
            direction += "down"
        if x_diff < -threshold:
            direction += "_left" if direction else "left"
        elif x_diff > threshold:
            direction += "_right" if direction else "right"
    return direction

def extract_features(lm, face_id):
    features = [face_id] 
    left_ear = calculate_ear(lm, left_eye_indices)
    right_ear = calculate_ear(lm, right_eye_indices)
    features.append(left_ear)
    features.append(right_ear)

    lec_x, lec_y = calculate_center(lm, left_eye_indices)
    rec_x, rec_y = calculate_center(lm, right_eye_indices)
    lic_x, lic_y = calculate_center(lm, left_iris_ids)
    ric_x, ric_y = calculate_center(lm, right_iris_ids)

    dx = lic_x - lec_x
    dy = lic_y - lec_y

    features += [lec_x, lec_y, rec_x, rec_y, lic_x, lic_y, ric_x, ric_y]
    features += [dx, dy]

    for i in eye_landmark_ids:
        features += [lm[i].x, lm[i].y, lm[i].z]
    for i in range(478):
        features += [lm[i].x, lm[i].y, lm[i].z]

    return features  # gaze_label 제외 (학습용 .npy 기준)

def pad_or_crop_sequence(seq, target_len):
    T, D = seq.shape
    if T >= target_len:
        return seq[:target_len]
    else:
        pad = np.zeros((target_len - T, D), dtype=np.float32)
        return np.vstack([seq, pad])

# === 핵심 함수 ===
def extract_feature_sequence_from_video(video_path, target_seq_len=256):
    mp_face_mesh = mp.solutions.face_mesh
    feature_vectors = []

    with mp_face_mesh.FaceMesh(
        static_image_mode=False,
        max_num_faces=2,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    ) as face_mesh:
        cap = cv2.VideoCapture(video_path)
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = face_mesh.process(rgb)

            if results.multi_face_landmarks:
                for face_id, face_landmarks in enumerate(results.multi_face_landmarks):
                    lm = face_landmarks.landmark
                    features = extract_features(lm, face_id)
                    feature_vectors.append(features[:-1])

        cap.release()

    if feature_vectors:
        raw_seq = np.array(feature_vectors, dtype=np.float32)
        fixed_seq = pad_or_crop_sequence(raw_seq, target_seq_len)
        return fixed_seq
    else:
        return None
