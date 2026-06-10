# ============================================================
# ABEMA-inspired EEG-only Emotion Recognition on Raw EAV
# Speaking trials only
# Features: DE + PSD from Delta/Theta/Alpha/Beta/Gamma bands
# Model: BiGRU + Band Attention
# Subjects: first 10 subjects
# ============================================================

import os
import re
import random
import warnings
import numpy as np

from scipy.io import loadmat
from scipy.signal import butter, filtfilt, welch

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
from sklearn.utils.class_weight import compute_class_weight

warnings.filterwarnings("ignore")


# ============================================================
# CONFIG
# ============================================================

ROOT_DIR = r"F:\emotion recognition\EAV\EAV"   # change this if needed

SUBJECT_IDS = list(range(1, 11))   # first 10 subjects

ORIGINAL_FS = 500                  # 10000 samples / 20 sec = 500 Hz
DOWNSAMPLE_FACTOR = 2              # 500 Hz -> 250 Hz
FS = ORIGINAL_FS // DOWNSAMPLE_FACTOR

TRIAL_SECONDS = 20
SEGMENT_SECONDS = 5
SEGMENTS_PER_TRIAL = TRIAL_SECONDS // SEGMENT_SECONDS

NUM_CHANNELS = 30
NUM_CLASSES = 5

TEST_SIZE = 0.2
BATCH_SIZE = 32
EPOCHS = 50
LR = 1e-4
WEIGHT_DECAY = 1e-4
RANDOM_SEED = 42

BANDS = {
    "Delta": (0.5, 4),
    "Theta": (4, 8),
    "Alpha": (8, 13),
    "Beta": (13, 30),
    "Gamma": (30, 50),
}

ID_TO_EMOTION = {
    0: "Neutral",
    1: "Sadness",
    2: "Anger",
    3: "Happiness",
    4: "Calmness",
}

# Includes possible filename variations
EMOTION_ALIASES = {
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
# SIGNAL PROCESSING
# ============================================================

def butter_bandpass_filter(data, lowcut, highcut, fs, order=4):
    """
    data shape: (channels, time)
    """
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq

    b, a = butter(order, [low, high], btype="band")
    filtered = filtfilt(b, a, data, axis=-1)

    return filtered


def normalize_eeg(segment):
    """
    Channel-wise z-score normalization.
    segment shape: (channels, time)
    """
    mean = segment.mean(axis=1, keepdims=True)
    std = segment.std(axis=1, keepdims=True) + 1e-6
    return (segment - mean) / std


def compute_de(signal):
    """
    Differential Entropy:
    DE = 0.5 * log(2*pi*e*variance)

    signal shape: (channels, time)
    output shape: (channels,)
    """
    var = np.var(signal, axis=1) + 1e-6
    de = 0.5 * np.log(2 * np.pi * np.e * var)
    return de


def compute_psd(signal, fs, band):
    """
    Mean PSD power in a frequency band.

    signal shape: (channels, time)
    output shape: (channels,)
    """
    freqs, psd = welch(
        signal,
        fs=fs,
        nperseg=min(256, signal.shape[1]),
        axis=1
    )

    low, high = band
    idx = np.logical_and(freqs >= low, freqs <= high)

    if idx.sum() == 0:
        return np.zeros(signal.shape[0], dtype=np.float32)

    band_power = psd[:, idx].mean(axis=1)
    return band_power


def extract_de_psd_features(segment, fs):
    """
    Extract DE + PSD for each EEG band.

    Input:
        segment shape: (30 channels, time)

    Output:
        features shape: (5 bands, 60 features)
        60 = 30 DE + 30 PSD
    """

    band_features = []

    for band_name, band_range in BANDS.items():
        low, high = band_range

        band_signal = butter_bandpass_filter(segment, low, high, fs)

        de = compute_de(band_signal)
        psd = compute_psd(band_signal, fs, band_range)

        psd = np.log(psd + 1e-6)

        feat = np.concatenate([de, psd], axis=0)  # (60,)
        band_features.append(feat)

    band_features = np.stack(band_features, axis=0)  # (5, 60)

    return band_features.astype(np.float32)


# ============================================================
# LABEL EXTRACTION FROM AUDIO FILENAMES
# ============================================================

def extract_speaking_labels_from_video(video_dir):
    """
    Extract labels from video filenames like:
    002_Trial_01_Speaking_Neutral.mp4
    004_Trial_02_Speaking_Anger.mp4

    Important:
    We use the FIRST number, e.g. 004, as the EEG trial index.
    Do NOT use Trial_02, because that may refer to conversation pair number.
    """

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

        # Use the first number at the beginning of the filename
        # Example: 004_Trial_02_Speaking_Anger.mp4 -> trial_num = 4
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
# LOAD RAW EEG
# ============================================================

def load_subject_eeg(subject_id):
    """
    Loads subject EEG file.

    Handles keys like:
    seg
    seg1
    seg2
    etc.

    Expected raw shape:
    (10000, 30, 200)

    Returns:
    eeg shape:
    (200, 30, 10000)
    """

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

    # Case 1: (time, channels, trials)
    if seg.shape[0] == 10000 and seg.shape[1] == 30:
        eeg = np.transpose(seg, (2, 1, 0)).astype(np.float32)

    # Case 2: already (trials, channels, time)
    elif seg.shape[1] == 30 and seg.shape[2] == 10000:
        eeg = seg.astype(np.float32)

    # Case 3: (channels, time, trials)
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
# BUILD DATASET
# ============================================================

def build_dataset():
    X = []
    y = []

    subject_ids = []
    trial_ids = []
    segment_ids = []

    raw_segment_len = ORIGINAL_FS * SEGMENT_SECONDS
    downsampled_segment_len = FS * SEGMENT_SECONDS

    for subject_id in SUBJECT_IDS:
        print("\n" + "=" * 70)
        print(f"Processing subject {subject_id}")

        eeg = load_subject_eeg(subject_id)

        video_dir = os.path.join(ROOT_DIR, f"subject{subject_id}", "Video")
        trial_to_label = extract_speaking_labels_from_video(video_dir)

      
        print(f"Subject {subject_id}: speaking trials found = {len(trial_to_label)}")

        for eeg_index, label in sorted(trial_to_label.items()):
            if eeg_index < 0 or eeg_index >= eeg.shape[0]:
                print(f"Skipping invalid trial index {eeg_index} for subject {subject_id}")
                continue

            trial = eeg[eeg_index]  # (30, 10000)

            # Broad band-pass filtering: 0.5–50 Hz
            trial = butter_bandpass_filter(trial, 0.5, 50, ORIGINAL_FS)

            # Downsample
            trial = trial[:, ::DOWNSAMPLE_FACTOR]  # (30, 5000)

            # Normalize whole trial
            trial = normalize_eeg(trial)

            # Split 20 seconds into four 5-second segments
            for seg_id in range(SEGMENTS_PER_TRIAL):
                start = seg_id * downsampled_segment_len
                end = start + downsampled_segment_len

                segment = trial[:, start:end]

                if segment.shape[1] != downsampled_segment_len:
                    continue

                features = extract_de_psd_features(segment, FS)

                X.append(features)
                y.append(label)

                subject_ids.append(subject_id)
                trial_ids.append(eeg_index + 1)
                segment_ids.append(seg_id + 1)

    if len(X) == 0:
        raise ValueError("No samples were created. Check ROOT_DIR and Audio filenames.")

    X = np.stack(X, axis=0)
    y = np.array(y, dtype=np.int64)

    print("\n" + "=" * 70)
    print("Final feature shape:", X.shape)
    print("Final label shape:", y.shape)

    print("\nClass distribution:")
    for class_id in range(NUM_CLASSES):
        print(f"{class_id} - {ID_TO_EMOTION[class_id]}: {np.sum(y == class_id)}")

    return X, y, subject_ids, trial_ids, segment_ids


# ============================================================
# DATASET CLASS
# ============================================================

class EEGFeatureDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


# ============================================================
# MODEL
# ============================================================

class BandAttention(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()

        self.att = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x):
        """
        x shape: (batch, bands, hidden_dim)
        """
        scores = self.att(x)
        weights = torch.softmax(scores, dim=1)
        context = torch.sum(weights * x, dim=1)

        return context, weights


class EEGFrequencyBiGRU(nn.Module):
    def __init__(self, input_dim=60, hidden_dim=128, num_classes=5):
        super().__init__()

        self.bigru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=0.3
        )

        self.band_attention = BandAttention(hidden_dim * 2)

        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, 128),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        """
        x shape: (batch, 5 bands, 60 features)
        """
        gru_out, _ = self.bigru(x)
        context, weights = self.band_attention(gru_out)
        logits = self.classifier(context)

        return logits, weights


# ============================================================
# TRAINING FUNCTIONS
# ============================================================

def train_one_epoch(model, loader, criterion, optimizer):
    model.train()

    losses = []
    preds_all = []
    labels_all = []

    for X_batch, y_batch in loader:
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)

        optimizer.zero_grad()

        logits, _ = model(X_batch)
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
        for X_batch, y_batch in loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)

            logits, weights = model(X_batch)
            loss = criterion(logits, y_batch)

            losses.append(loss.item())

            preds = torch.argmax(logits, dim=1)

            preds_all.extend(preds.detach().cpu().numpy())
            labels_all.extend(y_batch.detach().cpu().numpy())
            attention_all.append(weights.detach().cpu().numpy())

    acc = accuracy_score(labels_all, preds_all) * 100
    f1 = f1_score(labels_all, preds_all, average="weighted", zero_division=0) * 100

    attention_all = np.concatenate(attention_all, axis=0)

    return np.mean(losses), acc, f1, np.array(labels_all), np.array(preds_all), attention_all


# ============================================================
# MAIN
# ============================================================

def main():
    X, y, subject_ids, trial_ids, segment_ids = build_dataset()

    unique_labels = np.unique(y)
    print("\nUnique labels in full dataset:", unique_labels)

    # If all classes are not present, stratify can fail.
    # So we use stratify only when every present class has at least 2 samples.
    class_counts = np.bincount(y, minlength=NUM_CLASSES)
    print("Class counts:", class_counts)

    can_stratify = np.all(class_counts[class_counts > 0] >= 2)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_SEED,
        stratify=y if can_stratify else None
    )

    train_dataset = EEGFeatureDataset(X_train, y_train)
    test_dataset = EEGFeatureDataset(X_test, y_test)

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

    # ============================================================
    # Safe class weights
    # ============================================================

    print("\nUnique labels in training set:", np.unique(y_train))
    print("Unique labels in test set:", np.unique(y_test))

    class_weights_np = np.ones(NUM_CLASSES, dtype=np.float32)

    present_classes = np.unique(y_train)

    computed_weights = compute_class_weight(
        class_weight="balanced",
        classes=present_classes,
        y=y_train
    )

    for cls, weight in zip(present_classes, computed_weights):
        class_weights_np[int(cls)] = weight

    class_weights = torch.tensor(class_weights_np, dtype=torch.float32).to(device)

    print("\nClass weights:", class_weights.detach().cpu().numpy())

    model = EEGFrequencyBiGRU(
        input_dim=NUM_CHANNELS * 2,
        hidden_dim=128,
        num_classes=NUM_CLASSES
    ).to(device)

    criterion = nn.CrossEntropyLoss(weight=class_weights)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LR,
        weight_decay=WEIGHT_DECAY
    )

    best_f1 = 0
    best_model_path = "best_eav_eeg_de_psd_bigru_attention.pth"

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

    print("\nAverage learned band attention:")
    for i, band_name in enumerate(BANDS.keys()):
        print(f"{band_name}: {avg_attention[i]:.4f}")


if __name__ == "__main__":
    main()
