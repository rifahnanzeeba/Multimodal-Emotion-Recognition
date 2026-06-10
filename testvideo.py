# ============================================================
# Test Saved Video-only Model on Unseen Subjects
# Subjects: 11 to 42
# Randomly select 1 speaking video from each subject
# Model: 2D CNN + BiLSTM + Temporal Attention
# No retraining, no model modification
# ============================================================

import os
import re
import random
import warnings
import numpy as np
import cv2

import torch
import torch.nn as nn

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix
)

warnings.filterwarnings("ignore")


# ============================================================
# CONFIG
# ============================================================

ROOT_DIR = r"F:\emotion recognition\EAV\EAV"   # change if needed

MODEL_PATH = "best_eav_video_2dcnn_bilstm_attention.pth"

TEST_SUBJECT_IDS = list(range(11, 43))   # subjects 11 to 42

NUM_FRAMES = 8
FRAME_SIZE = 64
SEGMENT_SECONDS = 5
SEGMENTS_PER_VIDEO = 4

NUM_CLASSES = 5
RANDOM_SEED = 42

ID_TO_EMOTION = {
    0: "Neutral",
    1: "Sadness",
    2: "Anger",
    3: "Happiness",
    4: "Calmness",
}

EMOTION_TO_ID = {
    "neutral": 0,
    "sadness": 1,
    "sad": 1,
    "anger": 2,
    "angry": 2,
    "happiness": 3,
    "happy": 3,
    "calmness": 4,
    "calm": 4,
}

random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)


# ============================================================
# MODEL CLASSES
# Must match the training model exactly
# ============================================================

class FrameCNNEncoder(nn.Module):
    def __init__(self, feature_dim=128):
        super().__init__()

        self.cnn = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1))
        )

        self.fc = nn.Sequential(
            nn.Linear(128, feature_dim),
            nn.ReLU(),
            nn.Dropout(0.3)
        )

    def forward(self, x):
        x = self.cnn(x)
        x = x.flatten(1)
        x = self.fc(x)
        return x


class TemporalAttention(nn.Module):
    def __init__(self, feature_dim):
        super().__init__()

        self.att = nn.Sequential(
            nn.Linear(feature_dim, feature_dim),
            nn.Tanh(),
            nn.Linear(feature_dim, 1)
        )

    def forward(self, x):
        scores = self.att(x)
        weights = torch.softmax(scores, dim=1)
        context = torch.sum(weights * x, dim=1)

        return context, weights


class Video2DCNNBiLSTMAttention(nn.Module):
    def __init__(self, num_classes=5, frame_feature_dim=128, hidden_dim=128):
        super().__init__()

        self.frame_encoder = FrameCNNEncoder(feature_dim=frame_feature_dim)

        self.lstm = nn.LSTM(
            input_size=frame_feature_dim,
            hidden_size=hidden_dim,
            num_layers=1,
            batch_first=True,
            bidirectional=True
        )

        self.attention = TemporalAttention(feature_dim=hidden_dim * 2)

        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        """
        x shape:
            (batch, 3, T, H, W)
        """

        batch, channels, time, height, width = x.shape

        x = x.permute(0, 2, 1, 3, 4)
        x = x.reshape(batch * time, channels, height, width)

        frame_features = self.frame_encoder(x)
        frame_features = frame_features.reshape(batch, time, -1)

        lstm_out, _ = self.lstm(frame_features)

        context, att_weights = self.attention(lstm_out)

        logits = self.classifier(context)

        return logits, att_weights


# ============================================================
# LOAD MODEL SAFELY
# ============================================================

model = Video2DCNNBiLSTMAttention(
    num_classes=NUM_CLASSES,
    frame_feature_dim=128,
    hidden_dim=128
).to(device)

model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval()

print("Loaded saved model:", MODEL_PATH)


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def extract_label_from_filename(fname):
    fname_lower = fname.lower()

    emotion_aliases_ordered = [
        ("calmness", 4),
        ("happiness", 3),
        ("sadness", 1),
        ("neutral", 0),
        ("anger", 2),
        ("happy", 3),
        ("sad", 1),
        ("calm", 4),
        ("angry", 2),
    ]

    for emotion_word, label_id in emotion_aliases_ordered:
        if emotion_word in fname_lower:
            return label_id

    return None


def get_speaking_videos_for_subject(subject_id):
    video_dir = os.path.join(ROOT_DIR, f"subject{subject_id}", "Video")

    if not os.path.exists(video_dir):
        print(f"Subject {subject_id}: Video folder missing")
        return []

    video_files = [
        f for f in os.listdir(video_dir)
        if f.lower().endswith((".mp4", ".avi", ".mov", ".mkv"))
    ]

    samples = []

    for fname in video_files:
        fname_lower = fname.lower()

        if "speaking" not in fname_lower:
            continue

        label = extract_label_from_filename(fname)

        if label is None:
            print("Could not find emotion in filename:", fname)
            continue

        video_path = os.path.join(video_dir, fname)

        samples.append({
            "subject_id": subject_id,
            "video_path": video_path,
            "filename": fname,
            "label": label
        })

    return samples


def load_video_segment_frames(
    video_path,
    start_sec,
    end_sec,
    num_frames=8,
    frame_size=64
):
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    frame_times = np.linspace(start_sec, end_sec, num_frames, endpoint=False)

    frames = []
    last_valid_frame = None

    for t in frame_times:
        cap.set(cv2.CAP_PROP_POS_MSEC, float(t * 1000.0))
        ret, frame = cap.read()

        if not ret or frame is None:
            if last_valid_frame is not None:
                frame = last_valid_frame.copy()
            else:
                frame = np.zeros((frame_size, frame_size, 3), dtype=np.uint8)
        else:
            last_valid_frame = frame.copy()

        frame = cv2.resize(frame, (frame_size, frame_size))
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        frame = frame.astype(np.float32) / 255.0
        frame = (frame - 0.5) / 0.5

        frames.append(frame)

    cap.release()

    frames = np.stack(frames, axis=0)            # (T, H, W, C)
    frames = np.transpose(frames, (3, 0, 1, 2))  # (C, T, H, W)

    return frames.astype(np.float32)


# ============================================================
# PREDICTION FUNCTIONS
# ============================================================

def predict_segment(video_path, start_sec, end_sec):
    frames = load_video_segment_frames(
        video_path=video_path,
        start_sec=start_sec,
        end_sec=end_sec,
        num_frames=NUM_FRAMES,
        frame_size=FRAME_SIZE
    )

    x = torch.tensor(frames, dtype=torch.float32).unsqueeze(0).to(device)

    with torch.no_grad():
        logits, attention = model(x)
        probs = torch.softmax(logits, dim=1)
        pred_id = torch.argmax(probs, dim=1).item()

    return pred_id, probs[0].detach().cpu().numpy(), attention.detach().cpu().numpy()


def predict_full_video(video_path):
    """
    Predicts full 20-sec speaking video by averaging probabilities
    across four 5-sec segments.
    """

    all_probs = []
    segment_preds = []

    for seg_id in range(SEGMENTS_PER_VIDEO):
        start_sec = seg_id * SEGMENT_SECONDS
        end_sec = (seg_id + 1) * SEGMENT_SECONDS

        pred_id, probs, attention = predict_segment(
            video_path=video_path,
            start_sec=start_sec,
            end_sec=end_sec
        )

        all_probs.append(probs)
        segment_preds.append(pred_id)

    avg_probs = np.mean(np.stack(all_probs, axis=0), axis=0)

    final_pred_id = int(np.argmax(avg_probs))
    final_confidence = float(avg_probs[final_pred_id])

    return final_pred_id, final_confidence, avg_probs, segment_preds


# ============================================================
# TEST ONE RANDOM SPEAKING VIDEO PER UNSEEN SUBJECT
# ============================================================

def test_random_one_video_per_subject():
    y_true = []
    y_pred = []

    selected_results = []

    for subject_id in TEST_SUBJECT_IDS:
        speaking_videos = get_speaking_videos_for_subject(subject_id)

        if len(speaking_videos) == 0:
            print(f"Subject {subject_id}: No speaking videos found")
            continue

        selected = random.choice(speaking_videos)

        true_label = selected["label"]
        video_path = selected["video_path"]
        fname = selected["filename"]

        try:
            pred_label, confidence, avg_probs, segment_preds = predict_full_video(video_path)
        except Exception as e:
            print(f"Subject {subject_id}: prediction failed for {fname}")
            print("Error:", e)
            continue

        y_true.append(true_label)
        y_pred.append(pred_label)

        correct = true_label == pred_label

        selected_results.append({
            "subject_id": subject_id,
            "filename": fname,
            "true_label": true_label,
            "pred_label": pred_label,
            "confidence": confidence,
            "correct": correct,
            "segment_preds": segment_preds,
            "avg_probs": avg_probs
        })

        print("\n" + "-" * 70)
        print(f"Subject {subject_id}")
        print("Video:", fname)
        print("True emotion:", ID_TO_EMOTION[true_label])
        print("Predicted emotion:", ID_TO_EMOTION[pred_label])
        print("Confidence:", round(confidence, 4))
        print("Correct:", correct)

        print("Segment predictions:", [ID_TO_EMOTION[p] for p in segment_preds])

        print("Average probabilities:")
        for i in range(NUM_CLASSES):
            print(f"  {ID_TO_EMOTION[i]}: {avg_probs[i]:.4f}")

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    print("\n" + "=" * 70)
    print("FINAL UNSEEN SUBJECT TEST SUMMARY")
    print("=" * 70)
    print("Total tested videos:", len(y_true))

    if len(y_true) == 0:
        print("No videos were tested.")
        return

    acc = accuracy_score(y_true, y_pred) * 100
    f1_weighted = f1_score(y_true, y_pred, average="weighted", zero_division=0) * 100
    f1_macro = f1_score(y_true, y_pred, average="macro", zero_division=0) * 100

    print("Accuracy:", round(acc, 2))
    print("Weighted F1:", round(f1_weighted, 2))
    print("Macro F1:", round(f1_macro, 2))

    print("\nClassification Report:")
    print(
        classification_report(
            y_true,
            y_pred,
            labels=list(range(NUM_CLASSES)),
            target_names=[ID_TO_EMOTION[i] for i in range(NUM_CLASSES)],
            digits=4,
            zero_division=0
        )
    )

    print("\nConfusion Matrix:")
    print(
        confusion_matrix(
            y_true,
            y_pred,
            labels=list(range(NUM_CLASSES))
        )
    )

    return selected_results, y_true, y_pred


# ============================================================
# RUN
# ============================================================

results, y_true, y_pred = test_random_one_video_per_subject()
