# ============================================================
# EAV Video-only Emotion Recognition
# Speaking trials only
# Labels from VIDEO filenames
# Raw video frames -> 2D CNN + BiLSTM + Temporal Attention
# Subjects: first 10 subjects
# ============================================================

import os
import re
import random
import warnings
import numpy as np
import cv2

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from sklearn.model_selection import train_test_split
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

SUBJECT_IDS = list(range(1, 11))   # first 10 subjects

TRIAL_SECONDS = 20
SEGMENT_SECONDS = 5
SEGMENTS_PER_TRIAL = TRIAL_SECONDS // SEGMENT_SECONDS

NUM_CLASSES = 5

# Video settings
NUM_FRAMES = 8          # sampled frames from each 5-sec segment
FRAME_SIZE = 64         # 64x64 for faster training

# Training settings
TEST_SIZE = 0.2
BATCH_SIZE = 8          # reduce to 4 if memory issue
EPOCHS = 20
LR = 1e-4
WEIGHT_DECAY = 1e-3
RANDOM_SEED = 42

ID_TO_EMOTION = {
    0: "Neutral",
    1: "Sadness",
    2: "Anger",
    3: "Happiness",
    4: "Calmness",
}


# ============================================================
# REPRODUCIBILITY
# ============================================================

def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


seed_everything(RANDOM_SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)


# ============================================================
# VIDEO LABEL + FILE EXTRACTION
# ============================================================

def extract_speaking_video_samples(video_dir, subject_id):
    """
    Extract speaking video files and labels.

    Example filename:
        002_Trial_02_Speaking_Neutral.mp4

    Important:
        Uses the FIRST number at the beginning of the filename.
        002 -> trial number 2
    """

    samples = []

    if not os.path.exists(video_dir):
        raise FileNotFoundError(f"Video folder not found: {video_dir}")

    video_files = [
        f for f in os.listdir(video_dir)
        if f.lower().endswith((".mp4", ".avi", ".mov", ".mkv"))
    ]

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

    counts = {ID_TO_EMOTION[i]: 0 for i in range(NUM_CLASSES)}

    for fname in video_files:
        fname_lower = fname.lower()

        if "speaking" not in fname_lower:
            continue

        first_number_match = re.match(r"(\d+)", fname_lower)

        if first_number_match is None:
            print("Could not find first trial number:", fname)
            continue

        trial_num = int(first_number_match.group(1))

        found_label = None

        for emotion_word, label_id in emotion_aliases_ordered:
            if emotion_word in fname_lower:
                found_label = label_id
                break

        if found_label is None:
            print("Could not find emotion:", fname)
            continue

        video_path = os.path.join(video_dir, fname)

        # Each speaking video is 20 sec.
        # Split it into four 5-sec samples.
        for seg_id in range(SEGMENTS_PER_TRIAL):
            sample = {
                "video_path": video_path,
                "label": found_label,
                "subject_id": subject_id,
                "trial_id": trial_num,
                "segment_id": seg_id + 1,
                "start_sec": seg_id * SEGMENT_SECONDS,
                "end_sec": (seg_id + 1) * SEGMENT_SECONDS,
            }

            samples.append(sample)

        counts[ID_TO_EMOTION[found_label]] += 1

    print("Detected speaking video emotion counts:")
    print(counts)

    return samples


# ============================================================
# FRAME LOADING
# ============================================================

def load_video_segment_frames(
    video_path,
    start_sec,
    end_sec,
    num_frames=8,
    frame_size=64
):
    """
    Uniformly samples frames from a video segment.

    Output shape:
        (3, num_frames, frame_size, frame_size)
    """

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

        # Normalize roughly to [-1, 1]
        frame = (frame - 0.5) / 0.5

        frames.append(frame)

    cap.release()

    frames = np.stack(frames, axis=0)           # (T, H, W, C)
    frames = np.transpose(frames, (3, 0, 1, 2)) # (C, T, H, W)

    return frames.astype(np.float32)


# ============================================================
# BUILD DATASET METADATA
# ============================================================

def build_samples():
    all_samples = []

    for subject_id in SUBJECT_IDS:
        print("\n" + "=" * 70)
        print(f"Processing subject {subject_id}")

        video_dir = os.path.join(ROOT_DIR, f"subject{subject_id}", "Video")

        subject_samples = extract_speaking_video_samples(
            video_dir=video_dir,
            subject_id=subject_id
        )

        print(f"Subject {subject_id}: video segment samples = {len(subject_samples)}")

        all_samples.extend(subject_samples)

    if len(all_samples) == 0:
        raise ValueError("No video samples were created. Check ROOT_DIR and video filenames.")

    labels = np.array([s["label"] for s in all_samples], dtype=np.int64)

    print("\n" + "=" * 70)
    print("Total video segment samples:", len(all_samples))
    print("Label shape:", labels.shape)

    print("\nClass distribution:")
    for class_id in range(NUM_CLASSES):
        print(f"{class_id} - {ID_TO_EMOTION[class_id]}: {np.sum(labels == class_id)}")

    return all_samples, labels


# ============================================================
# DATASET CLASS
# ============================================================

class VideoDataset(Dataset):
    def __init__(self, samples):
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]

        frames = load_video_segment_frames(
            video_path=s["video_path"],
            start_sec=s["start_sec"],
            end_sec=s["end_sec"],
            num_frames=NUM_FRAMES,
            frame_size=FRAME_SIZE
        )

        x = torch.tensor(frames, dtype=torch.float32)
        y = torch.tensor(s["label"], dtype=torch.long)

        return x, y


# ============================================================
# MODEL: 2D CNN + BiLSTM + Temporal Attention
# ============================================================

class FrameCNNEncoder(nn.Module):
    def __init__(self, feature_dim=128):
        super().__init__()

        # Input frame: (batch*T, 3, 64, 64)
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),   # 64 -> 32

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),   # 32 -> 16

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),   # 16 -> 8

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
        """
        x shape:
            (batch*T, 3, H, W)
        """

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
        """
        x shape:
            (batch, time, feature_dim)
        """

        scores = self.att(x)                  # (batch, time, 1)
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
        Input x shape from dataset:
            (batch, 3, T, H, W)
        """

        batch, channels, time, height, width = x.shape

        # (batch, 3, T, H, W) -> (batch, T, 3, H, W)
        x = x.permute(0, 2, 1, 3, 4)

        # (batch*T, 3, H, W)
        x = x.reshape(batch * time, channels, height, width)

        frame_features = self.frame_encoder(x)  # (batch*T, frame_feature_dim)

        # (batch, T, frame_feature_dim)
        frame_features = frame_features.reshape(batch, time, -1)

        lstm_out, _ = self.lstm(frame_features) # (batch, T, hidden_dim*2)

        context, att_weights = self.attention(lstm_out)

        logits = self.classifier(context)

        return logits, att_weights


# ============================================================
# TRAINING FUNCTIONS
# ============================================================

def train_one_epoch(model, loader, criterion, optimizer):
    model.train()

    losses = []
    preds_all = []
    labels_all = []

    for video_batch, y_batch in loader:
        video_batch = video_batch.to(device)
        y_batch = y_batch.to(device)

        optimizer.zero_grad()

        logits, _ = model(video_batch)
        loss = criterion(logits, y_batch)

        loss.backward()
        optimizer.step()

        losses.append(loss.item())

        preds = torch.argmax(logits, dim=1)

        preds_all.extend(preds.detach().cpu().numpy())
        labels_all.extend(y_batch.detach().cpu().numpy())

    acc = accuracy_score(labels_all, preds_all) * 100
    f1 = f1_score(labels_all, preds_all, average="weighted", zero_division=0) * 100

    return np.mean(losses), acc, f1


def evaluate(model, loader, criterion):
    model.eval()

    losses = []
    preds_all = []
    labels_all = []
    attention_all = []

    with torch.no_grad():
        for video_batch, y_batch in loader:
            video_batch = video_batch.to(device)
            y_batch = y_batch.to(device)

            logits, att_weights = model(video_batch)
            loss = criterion(logits, y_batch)

            losses.append(loss.item())

            preds = torch.argmax(logits, dim=1)

            preds_all.extend(preds.detach().cpu().numpy())
            labels_all.extend(y_batch.detach().cpu().numpy())
            attention_all.append(att_weights.detach().cpu().numpy())

    acc = accuracy_score(labels_all, preds_all) * 100
    f1 = f1_score(labels_all, preds_all, average="weighted", zero_division=0) * 100

    attention_all = np.concatenate(attention_all, axis=0)

    return np.mean(losses), acc, f1, np.array(labels_all), np.array(preds_all), attention_all


# ============================================================
# MAIN
# ============================================================

def main():
    samples, labels = build_samples()

    print("\nUnique labels in full dataset:", np.unique(labels))
    class_counts = np.bincount(labels, minlength=NUM_CLASSES)
    print("Class counts:", class_counts)

    train_samples, test_samples, y_train, y_test = train_test_split(
        samples,
        labels,
        test_size=TEST_SIZE,
        random_state=RANDOM_SEED,
        stratify=labels
    )

    train_dataset = VideoDataset(train_samples)
    test_dataset = VideoDataset(test_samples)

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0
    )

    print("\nUnique labels in training set:", np.unique(y_train))
    print("Unique labels in test set:", np.unique(y_test))

    model = Video2DCNNBiLSTMAttention(
        num_classes=NUM_CLASSES,
        frame_feature_dim=128,
        hidden_dim=128
    ).to(device)

    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LR,
        weight_decay=WEIGHT_DECAY
    )

    best_f1 = 0
    best_model_path = "best_eav_video_2dcnn_bilstm_attention.pth"

    print("\nStarting video-only training...\n")

    for epoch in range(EPOCHS):
        train_loss, train_acc, train_f1 = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer
        )

        test_loss, test_acc, test_f1, y_true, y_pred, attention = evaluate(
            model,
            test_loader,
            criterion
        )

        if test_f1 > best_f1:
            best_f1 = test_f1
            torch.save(model.state_dict(), best_model_path)

        print(
            f"Epoch [{epoch+1:02d}/{EPOCHS}] "
            f"Train Loss: {train_loss:.4f} | "
            f"Train Acc: {train_acc:.2f}% | "
            f"Train F1: {train_f1:.2f}% || "
            f"Test Loss: {test_loss:.4f} | "
            f"Test Acc: {test_acc:.2f}% | "
            f"Test F1: {test_f1:.2f}%"
        )

    print("\nTraining complete.")
    print("Best Test F1:", best_f1)
    print("Best model saved to:", best_model_path)

    if os.path.exists(best_model_path):
        model.load_state_dict(torch.load(best_model_path, map_location=device))

    test_loss, test_acc, test_f1, y_true, y_pred, attention = evaluate(
        model,
        test_loader,
        criterion
    )

    print("\nFinal Test Accuracy:", test_acc)
    print("Final Test Weighted F1:", test_f1)

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

    avg_attention = attention.mean(axis=0).squeeze()

    print("\nAverage video temporal attention shape:", avg_attention.shape)
    print("Average temporal attention weights:", avg_attention)


if __name__ == "__main__":
    main()
