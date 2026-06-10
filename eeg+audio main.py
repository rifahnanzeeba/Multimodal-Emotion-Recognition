# ============================================================
# EAV EEG + Audio Fusion Emotion Recognition
# EEG: Raw EEG Conv1D + BiLSTM
# Audio: Log-Mel Spectrogram + CNN + Attention
# Fusion options:
#   1. concat
#   2. gated
#   3. attention
# Subjects: first 10 subjects
# ============================================================

import os
import re
import random
import warnings
import numpy as np

from scipy.io import loadmat
from scipy.signal import butter, filtfilt

import librosa

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

ROOT_DIR = r"F:\emotion recognition\EAV\EAV"

SUBJECT_IDS = list(range(1, 11))

# Choose one: "concat", "gated", "attention"
FUSION_TYPE = "gated"

ORIGINAL_FS = 500
DOWNSAMPLE_FACTOR = 2
FS = ORIGINAL_FS // DOWNSAMPLE_FACTOR

TRIAL_SECONDS = 20
SEGMENT_SECONDS = 5
SEGMENTS_PER_TRIAL = TRIAL_SECONDS // SEGMENT_SECONDS

NUM_CHANNELS = 30
NUM_CLASSES = 5

# Audio settings
AUDIO_SR = 16000
N_MELS = 64
N_FFT = 1024
HOP_LENGTH = 512
MAX_AUDIO_FRAMES = 160

# Training settings
TEST_SIZE = 0.2
BATCH_SIZE = 16
EPOCHS = 25
LR = 1e-4
WEIGHT_DECAY = 1e-3
RANDOM_SEED = 42

EMBED_DIM = 128

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
print("Fusion type:", FUSION_TYPE)


# ============================================================
# EEG PROCESSING
# ============================================================

def butter_bandpass_filter(data, lowcut, highcut, fs, order=4):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq

    b, a = butter(order, [low, high], btype="band")
    filtered = filtfilt(b, a, data, axis=-1)

    return filtered


def normalize_eeg(trial):
    mean = trial.mean(axis=1, keepdims=True)
    std = trial.std(axis=1, keepdims=True) + 1e-6
    return (trial - mean) / std


# ============================================================
# AUDIO PROCESSING: LOG-MEL
# ============================================================

def load_audio_file(audio_path):
    wav, sr = librosa.load(audio_path, sr=AUDIO_SR, mono=True)

    expected_len = AUDIO_SR * TRIAL_SECONDS

    if len(wav) < expected_len:
        wav = np.pad(wav, (0, expected_len - len(wav)), mode="constant")
    else:
        wav = wav[:expected_len]

    return wav.astype(np.float32)


def normalize_logmel(logmel):
    mean = logmel.mean()
    std = logmel.std() + 1e-6
    return (logmel - mean) / std


def pad_or_truncate_logmel(logmel, max_frames=160):
    n_mels, frames = logmel.shape

    if frames < max_frames:
        logmel = np.pad(logmel, ((0, 0), (0, max_frames - frames)), mode="constant")
    else:
        logmel = logmel[:, :max_frames]

    return logmel.astype(np.float32)


def extract_logmel_segment(audio_segment):
    mel = librosa.feature.melspectrogram(
        y=audio_segment,
        sr=AUDIO_SR,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        n_mels=N_MELS,
        power=2.0
    )

    logmel = librosa.power_to_db(mel, ref=np.max)
    logmel = normalize_logmel(logmel)
    logmel = pad_or_truncate_logmel(logmel, MAX_AUDIO_FRAMES)

    return logmel


# ============================================================
# LABEL EXTRACTION FROM VIDEO FILENAMES
# ============================================================

def extract_speaking_labels_from_video(video_dir):
    trial_to_label = {}

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

    for fname in video_files:
        fname_lower = fname.lower()

        if "speaking" not in fname_lower:
            continue

        first_number_match = re.match(r"(\d+)", fname_lower)

        if first_number_match is None:
            print("Could not find first trial number:", fname)
            continue

        trial_num = int(first_number_match.group(1))
        eeg_index = trial_num - 1

        found_label = None

        for emotion_word, label_id in emotion_aliases_ordered:
            if emotion_word in fname_lower:
                found_label = label_id
                break

        if found_label is None:
            print("Could not find emotion:", fname)
            continue

        trial_to_label[eeg_index] = found_label

    print("Detected emotion counts from VIDEO filenames:")
    counts = {ID_TO_EMOTION[i]: 0 for i in range(NUM_CLASSES)}

    for label in trial_to_label.values():
        counts[ID_TO_EMOTION[label]] += 1

    print(counts)

    return trial_to_label


# ============================================================
# AUDIO FILE MATCHING
# ============================================================

def build_audio_file_map(audio_dir):
    audio_map = {}

    if not os.path.exists(audio_dir):
        raise FileNotFoundError(f"Audio folder not found: {audio_dir}")

    audio_files = [
        f for f in os.listdir(audio_dir)
        if f.lower().endswith((".wav", ".mp3", ".flac", ".m4a"))
    ]

    for fname in audio_files:
        fname_lower = fname.lower()

        if "speaking" not in fname_lower:
            continue

        first_number_match = re.match(r"(\d+)", fname_lower)

        if first_number_match is None:
            print("Could not find first trial number in audio:", fname)
            continue

        trial_num = int(first_number_match.group(1))
        eeg_index = trial_num - 1

        audio_map[eeg_index] = os.path.join(audio_dir, fname)

    return audio_map


# ============================================================
# LOAD EEG
# ============================================================

def load_subject_eeg(subject_id):
    eeg_path = os.path.join(
        ROOT_DIR,
        f"subject{subject_id}",
        "EEG",
        f"subject{subject_id}_eeg.mat"
    )

    if not os.path.exists(eeg_path):
        raise FileNotFoundError(f"EEG file not found: {eeg_path}")

    mat = loadmat(eeg_path)

    available_keys = [k for k in mat.keys() if not k.startswith("__")]
    print(f"Subject {subject_id} EEG keys:", available_keys)

    if "seg" in mat:
        seg = mat["seg"]
        selected_key = "seg"
    else:
        seg_keys = [k for k in available_keys if k.lower().startswith("seg")]

        if len(seg_keys) == 0:
            raise KeyError(
                f"No EEG key like 'seg' or 'seg1' found in {eeg_path}. "
                f"Available keys: {available_keys}"
            )

        selected_key = seg_keys[0]
        seg = mat[selected_key]

    print(f"Using EEG key: {selected_key}")
    print(f"Subject {subject_id} raw EEG shape:", seg.shape)

    if seg.ndim != 3:
        raise ValueError(f"Expected 3D EEG array, got shape {seg.shape}")

    if seg.shape[0] == 10000 and seg.shape[1] == 30:
        eeg = np.transpose(seg, (2, 1, 0)).astype(np.float32)

    elif seg.shape[1] == 30 and seg.shape[2] == 10000:
        eeg = seg.astype(np.float32)

    elif seg.shape[0] == 30 and seg.shape[1] == 10000:
        eeg = np.transpose(seg, (2, 0, 1)).astype(np.float32)

    else:
        raise ValueError(
            f"Unexpected EEG shape for subject {subject_id}: {seg.shape}. "
            "Expected something like (10000, 30, 200)."
        )

    print(f"Subject {subject_id} converted EEG shape:", eeg.shape)

    return eeg


# ============================================================
# BUILD EEG + AUDIO DATASET
# ============================================================

def build_dataset():
    X_eeg = []
    X_audio = []
    y = []

    subject_ids = []
    trial_ids = []
    segment_ids = []

    downsampled_segment_len = FS * SEGMENT_SECONDS
    audio_segment_len = AUDIO_SR * SEGMENT_SECONDS

    missing_audio_count = 0

    for subject_id in SUBJECT_IDS:
        print("\n" + "=" * 70)
        print(f"Processing subject {subject_id}")

        eeg = load_subject_eeg(subject_id)

        video_dir = os.path.join(ROOT_DIR, f"subject{subject_id}", "Video")
        audio_dir = os.path.join(ROOT_DIR, f"subject{subject_id}", "Audio")

        trial_to_label = extract_speaking_labels_from_video(video_dir)
        audio_map = build_audio_file_map(audio_dir)

        print(f"Subject {subject_id}: speaking video labels found = {len(trial_to_label)}")
        print(f"Subject {subject_id}: speaking audio files found = {len(audio_map)}")

        for eeg_index, label in sorted(trial_to_label.items()):
            if eeg_index < 0 or eeg_index >= eeg.shape[0]:
                print(f"Skipping invalid trial index {eeg_index} for subject {subject_id}")
                continue

            if eeg_index not in audio_map:
                missing_audio_count += 1
                continue

            # -------------------------
            # EEG processing
            # -------------------------
            trial = eeg[eeg_index]

            trial = butter_bandpass_filter(trial, 0.5, 50, ORIGINAL_FS)
            trial = trial[:, ::DOWNSAMPLE_FACTOR]
            trial = normalize_eeg(trial)

            # -------------------------
            # Audio processing
            # -------------------------
            audio_path = audio_map[eeg_index]

            try:
                wav = load_audio_file(audio_path)
            except Exception as e:
                print("Audio load failed:", audio_path, e)
                missing_audio_count += 1
                continue

            for seg_id in range(SEGMENTS_PER_TRIAL):
                # EEG segment
                eeg_start = seg_id * downsampled_segment_len
                eeg_end = eeg_start + downsampled_segment_len

                eeg_segment = trial[:, eeg_start:eeg_end]

                if eeg_segment.shape[1] != downsampled_segment_len:
                    continue

                # Audio segment
                aud_start = seg_id * audio_segment_len
                aud_end = aud_start + audio_segment_len

                audio_segment = wav[aud_start:aud_end]

                if len(audio_segment) < audio_segment_len:
                    audio_segment = np.pad(
                        audio_segment,
                        (0, audio_segment_len - len(audio_segment)),
                        mode="constant"
                    )

                logmel = extract_logmel_segment(audio_segment)

                X_eeg.append(eeg_segment.astype(np.float32))
                X_audio.append(logmel.astype(np.float32))
                y.append(label)

                subject_ids.append(subject_id)
                trial_ids.append(eeg_index + 1)
                segment_ids.append(seg_id + 1)

    if len(X_eeg) == 0:
        raise ValueError("No samples were created. Check ROOT_DIR and Audio/Video filenames.")

    X_eeg = np.stack(X_eeg, axis=0)
    X_audio = np.stack(X_audio, axis=0)
    y = np.array(y, dtype=np.int64)

    print("\n" + "=" * 70)
    print("Final EEG dataset shape:", X_eeg.shape)
    print("Final Audio Log-Mel dataset shape:", X_audio.shape)
    print("Final label shape:", y.shape)
    print("Missing audio trials:", missing_audio_count)

    print("\nClass distribution:")
    for class_id in range(NUM_CLASSES):
        print(f"{class_id} - {ID_TO_EMOTION[class_id]}: {np.sum(y == class_id)}")

    return X_eeg, X_audio, y, subject_ids, trial_ids, segment_ids


# ============================================================
# DATASET CLASS
# ============================================================

class EEGAudioDataset(Dataset):
    def __init__(self, X_eeg, X_audio, y):
        self.X_eeg = torch.tensor(X_eeg, dtype=torch.float32)
        self.X_audio = torch.tensor(X_audio, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X_eeg[idx], self.X_audio[idx], self.y[idx]


# ============================================================
# ENCODERS
# ============================================================

class EEGEncoder(nn.Module):
    def __init__(self, num_channels=30, embedding_dim=128):
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv1d(num_channels, 64, kernel_size=7, padding=3),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(4),

            nn.Conv1d(64, 128, kernel_size=5, padding=2),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(4),

            nn.Conv1d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(2),
        )

        self.lstm = nn.LSTM(
            input_size=128,
            hidden_size=128,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=0.3
        )

        self.proj = nn.Sequential(
            nn.Linear(256, embedding_dim),
            nn.ReLU(),
            nn.Dropout(0.4)
        )

    def forward(self, x):
        # x: (batch, 30, 1250)
        x = self.conv(x)
        x = x.permute(0, 2, 1)

        lstm_out, _ = self.lstm(x)

        x = lstm_out.mean(dim=1)
        emb = self.proj(x)

        return emb


class TemporalAttention(nn.Module):
    def __init__(self, feature_dim):
        super().__init__()

        self.att = nn.Sequential(
            nn.Linear(feature_dim, feature_dim),
            nn.Tanh(),
            nn.Linear(feature_dim, 1)
        )

    def forward(self, x):
        # x: (batch, time, feature_dim)
        scores = self.att(x)
        weights = torch.softmax(scores, dim=1)
        context = torch.sum(weights * x, dim=1)

        return context, weights


class AudioLogMelEncoder(nn.Module):
    def __init__(self, embedding_dim=128):
        super().__init__()

        # Input: (batch, 1, 64, 160)
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2)),    # (32, 32, 80)

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2)),    # (64, 16, 40)

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 1)),    # (128, 8, 40)
        )

        self.temporal_proj = nn.Sequential(
            nn.Linear(128 * 8, 256),
            nn.ReLU(),
            nn.Dropout(0.3)
        )

        self.attention = TemporalAttention(feature_dim=256)

        self.proj = nn.Sequential(
            nn.Linear(256, embedding_dim),
            nn.ReLU(),
            nn.Dropout(0.4)
        )

    def forward(self, x):
        # x: (batch, 64, 160)
        x = x.unsqueeze(1)              # (batch, 1, 64, 160)

        x = self.cnn(x)                 # (batch, 128, 8, 40)

        x = x.permute(0, 3, 1, 2)       # (batch, 40, 128, 8)

        batch, time, channels, freq = x.shape
        x = x.reshape(batch, time, channels * freq)  # (batch, 40, 1024)

        x = self.temporal_proj(x)       # (batch, 40, 256)

        context, att_weights = self.attention(x)     # (batch, 256)

        emb = self.proj(context)        # (batch, embedding_dim)

        return emb, att_weights


# ============================================================
# FUSION MODEL
# ============================================================

class EEGAudioFusionModel(nn.Module):
    def __init__(self, num_classes=5, embedding_dim=128, fusion_type="concat"):
        super().__init__()

        self.fusion_type = fusion_type

        self.eeg_encoder = EEGEncoder(
            num_channels=NUM_CHANNELS,
            embedding_dim=embedding_dim
        )

        self.audio_encoder = AudioLogMelEncoder(
            embedding_dim=embedding_dim
        )

        if fusion_type == "concat":
            self.classifier = nn.Sequential(
                nn.Linear(embedding_dim * 2, 256),
                nn.ReLU(),
                nn.Dropout(0.5),
                nn.Linear(256, 128),
                nn.ReLU(),
                nn.Dropout(0.4),
                nn.Linear(128, num_classes)
            )

        elif fusion_type == "gated":
            self.gate = nn.Sequential(
                nn.Linear(embedding_dim * 2, embedding_dim),
                nn.Sigmoid()
            )

            self.classifier = nn.Sequential(
                nn.Linear(embedding_dim, 128),
                nn.ReLU(),
                nn.Dropout(0.5),
                nn.Linear(128, num_classes)
            )

        elif fusion_type == "attention":
            self.modality_attention = nn.Sequential(
                nn.Linear(embedding_dim, embedding_dim),
                nn.Tanh(),
                nn.Linear(embedding_dim, 1)
            )

            self.classifier = nn.Sequential(
                nn.Linear(embedding_dim, 128),
                nn.ReLU(),
                nn.Dropout(0.5),
                nn.Linear(128, num_classes)
            )

        else:
            raise ValueError("fusion_type must be 'concat', 'gated', or 'attention'")

    def forward(self, eeg, audio):
        eeg_emb = self.eeg_encoder(eeg)
        audio_emb, audio_att = self.audio_encoder(audio)

        if self.fusion_type == "concat":
            fused = torch.cat([eeg_emb, audio_emb], dim=1)

        elif self.fusion_type == "gated":
            combined = torch.cat([eeg_emb, audio_emb], dim=1)
            gate = self.gate(combined)

            # gate close to 1 means rely more on EEG
            # gate close to 0 means rely more on audio
            fused = gate * eeg_emb + (1 - gate) * audio_emb

        elif self.fusion_type == "attention":
            modality_stack = torch.stack([eeg_emb, audio_emb], dim=1)  # (batch, 2, emb)

            scores = self.modality_attention(modality_stack)           # (batch, 2, 1)
            weights = torch.softmax(scores, dim=1)

            fused = torch.sum(weights * modality_stack, dim=1)         # (batch, emb)

        logits = self.classifier(fused)

        return logits, audio_att


# ============================================================
# TRAINING FUNCTIONS
# ============================================================

def train_one_epoch(model, loader, criterion, optimizer):
    model.train()

    losses = []
    preds_all = []
    labels_all = []

    for eeg_batch, audio_batch, y_batch in loader:
        eeg_batch = eeg_batch.to(device)
        audio_batch = audio_batch.to(device)
        y_batch = y_batch.to(device)

        optimizer.zero_grad()

        logits, _ = model(eeg_batch, audio_batch)
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
        for eeg_batch, audio_batch, y_batch in loader:
            eeg_batch = eeg_batch.to(device)
            audio_batch = audio_batch.to(device)
            y_batch = y_batch.to(device)

            logits, audio_att = model(eeg_batch, audio_batch)
            loss = criterion(logits, y_batch)

            losses.append(loss.item())

            preds = torch.argmax(logits, dim=1)

            preds_all.extend(preds.detach().cpu().numpy())
            labels_all.extend(y_batch.detach().cpu().numpy())
            attention_all.append(audio_att.detach().cpu().numpy())

    acc = accuracy_score(labels_all, preds_all) * 100
    f1 = f1_score(labels_all, preds_all, average="weighted", zero_division=0) * 100

    attention_all = np.concatenate(attention_all, axis=0)

    return np.mean(losses), acc, f1, np.array(labels_all), np.array(preds_all), attention_all


# ============================================================
# MAIN
# ============================================================

def main():
    X_eeg, X_audio, y, subject_ids, trial_ids, segment_ids = build_dataset()

    print("\nUnique labels in full dataset:", np.unique(y))
    class_counts = np.bincount(y, minlength=NUM_CLASSES)
    print("Class counts:", class_counts)

    can_stratify = np.all(class_counts[class_counts > 0] >= 2)

    X_eeg_train, X_eeg_test, X_audio_train, X_audio_test, y_train, y_test = train_test_split(
        X_eeg,
        X_audio,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_SEED,
        stratify=y if can_stratify else None
    )

    train_dataset = EEGAudioDataset(X_eeg_train, X_audio_train, y_train)
    test_dataset = EEGAudioDataset(X_eeg_test, X_audio_test, y_test)

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

    model = EEGAudioFusionModel(
        num_classes=NUM_CLASSES,
        embedding_dim=EMBED_DIM,
        fusion_type=FUSION_TYPE
    ).to(device)

    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LR,
        weight_decay=WEIGHT_DECAY
    )

    best_f1 = 0
    best_model_path = f"best_eav_eeg_audio_logmel_{FUSION_TYPE}_fusion.pth"

    print("\nStarting training...\n")

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
    print("\nAverage audio temporal attention shape:", avg_attention.shape)
    print("First 10 average attention weights:", avg_attention[:10])


if __name__ == "__main__":
    main()





