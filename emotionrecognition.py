import os
import re
import glob
import cv2
import pickle
import random
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# =========================================================
# CONFIG
# =========================================================
BASE_DATA_DIR = r"F:\emotion recognition\EAV\EAV"
VISION_PKL_DIR = r"F:\emotion recognition\Input_images\Input_images\Vision"
EEG_PKL_DIR = r"F:\emotion recognition\Input_images\Input_images\EEG"

ALL_SUBJECT_IDS = list(range(1, 21))

# split options:
# "subject_independent" -> train on TRAIN_SUBJECT_IDS, test on TEST_SUBJECT_IDS
# "random_mixed"        -> random sample split across ALL_SUBJECT_IDS
SPLIT_MODE = "subject_independent"
TRAIN_SUBJECT_IDS = list(range(1, 17))
TEST_SUBJECT_IDS = list(range(17, 21))

BATCH_SIZE = 4
EPOCHS = 25
LR = 1e-4
TEST_SIZE = 0.2          # only used if SPLIT_MODE == "random_mixed"
RANDOM_STATE = 42

NUM_VIDEO_FRAMES = 16
VIDEO_SIZE = 112

MODEL_SAVE_PATH = "eeg_video_multimodal_subject_pkl_5emotion_model.pth"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

EMOTION_TO_INDEX = {
    "Neutral": 0,
    "Sadness": 1,
    "Anger": 2,
    "Happiness": 3,
    "Calmness": 4,
}
INDEX_TO_EMOTION = {v: k for k, v in EMOTION_TO_INDEX.items()}


# =========================================================
# FILE PARSING
# =========================================================
def get_sorted_video_label_files(video_dir):
    """Use mp4 filenames as the label source."""
    video_files = glob.glob(os.path.join(video_dir, "*.mp4"))

    def extract_id(path):
        name = os.path.basename(path)
        m = re.match(r"(\d+)_Trial_(\d+)_(Listening|Speaking)_(\w+)\.mp4", name)
        if not m:
            raise ValueError(f"Unexpected video filename format: {name}")
        return int(m.group(1))

    return sorted(video_files, key=extract_id)


def parse_video_filename(path):
    name = os.path.basename(path)
    m = re.match(r"(\d+)_Trial_(\d+)_(Listening|Speaking)_(\w+)\.mp4", name)
    if not m:
        raise ValueError(f"Unexpected video filename format: {name}")

    global_idx = int(m.group(1))
    trial_no = int(m.group(2))
    mode = m.group(3)
    emotion = m.group(4)

    return global_idx, trial_no, mode, emotion


def get_subject_vision_pkl_path(subject_id):
    return os.path.join(VISION_PKL_DIR, f"subject_{subject_id:02d}_vis.pkl")


def get_subject_eeg_pkl_path(subject_id):
    return os.path.join(EEG_PKL_DIR, f"subject_{subject_id:02d}_eeg.pkl")


# =========================================================
# BUILD DATASET METADATA
# =========================================================
def build_subject_metadata(subject_id, video_dir):
    video_files = get_sorted_video_label_files(video_dir)

    if len(video_files) != 200:
        raise ValueError(f"Expected 200 videos, found {len(video_files)} in {video_dir}")

    subject_vis_pkl = get_subject_vision_pkl_path(subject_id)
    if not os.path.exists(subject_vis_pkl):
        raise FileNotFoundError(f"Vision pkl not found for subject {subject_id}: {subject_vis_pkl}")

    subject_eeg_pkl = get_subject_eeg_pkl_path(subject_id)
    if not os.path.exists(subject_eeg_pkl):
        raise FileNotFoundError(f"EEG pkl not found for subject {subject_id}: {subject_eeg_pkl}")

    rows = []
    for trial_idx, mp4_path in enumerate(video_files):
        global_idx, trial_no, mode, emotion = parse_video_filename(mp4_path)

        rows.append({
            "subject_id": subject_id,
            "trial_index": trial_idx,         # 0..199
            "eeg_index": trial_idx,           # aligned with EEG trial index
            "global_idx": global_idx,
            "trial_no": trial_no,
            "mode": mode,
            "emotion_name": emotion,
            "label": EMOTION_TO_INDEX[emotion],
            "label_source_mp4": mp4_path,
            "vision_pkl_path": subject_vis_pkl,
            "eeg_pkl_path": subject_eeg_pkl,
        })

    return rows


# =========================================================
# GENERIC PKL HELPERS
# =========================================================
def _extract_array_from_pkl_obj(obj):
    if isinstance(obj, np.ndarray):
        return obj

    if isinstance(obj, (list, tuple)):
        # prefer ndarray items
        for item in obj:
            if isinstance(item, np.ndarray):
                return item
        # search nested
        for item in obj:
            if isinstance(item, (list, tuple, dict)):
                arr = _extract_array_from_pkl_obj(item)
                if arr is not None:
                    return arr

    if isinstance(obj, dict):
        preferred_keys = ["data", "eeg", "vision", "video", "frames", "x", "seg"]
        for key in preferred_keys:
            if key in obj and isinstance(obj[key], np.ndarray):
                return obj[key]
        for _, value in obj.items():
            if isinstance(value, np.ndarray):
                return value

    return None


def load_pkl_array(pkl_path, kind="data"):
    with open(pkl_path, "rb") as f:
        try:
            obj = pickle.load(f)
        except Exception:
            f.seek(0)
            obj = pickle.load(f, encoding="latin1")

    arr = _extract_array_from_pkl_obj(obj)
    if arr is None:
        raise ValueError(f"Could not find usable ndarray in {kind} pkl: {pkl_path}")

    print(f"\nLoaded {kind} pkl: {pkl_path}")
    print(f"{kind} array shape:", arr.shape)
    print(f"{kind} array dtype:", arr.dtype)

    return arr


# =========================================================
# EEG PKL LOADING
# =========================================================
def extract_eeg_trials_from_subject_array(subject_eeg_array):
    """
    Convert subject EEG array to shape (N, C, T).

    Supported common layouts:
    - (N, C, T)
    - (N, T, C)
    - (T, C, N)
    - (C, T, N)

    Returns:
        X_eeg: (N, C, T)
    """
    arr = subject_eeg_array

    if arr.ndim != 3:
        raise ValueError(f"Unsupported EEG array ndim: {arr.ndim}, shape: {arr.shape}")

    # (N, C, T)
    if arr.shape[0] >= 200 and arr.shape[1] <= 64 and arr.shape[2] > arr.shape[1]:
        X = arr.astype(np.float32)

    # (N, T, C)
    elif arr.shape[0] >= 200 and arr.shape[2] <= 64:
        X = np.transpose(arr, (0, 2, 1)).astype(np.float32)

    # (T, C, N)
    elif arr.shape[0] > arr.shape[1] and arr.shape[2] >= 200 and arr.shape[1] <= 64:
        X = np.transpose(arr, (2, 1, 0)).astype(np.float32)

    # (C, T, N)
    elif arr.shape[0] <= 64 and arr.shape[2] >= 200:
        X = np.transpose(arr, (2, 0, 1)).astype(np.float32)

    else:
        raise ValueError(f"Could not infer EEG layout from shape {arr.shape}")

    return X


def load_subject_eeg_pkl_trials(eeg_pkl_path):
    arr = load_pkl_array(eeg_pkl_path, kind="EEG")
    X = extract_eeg_trials_from_subject_array(arr)
    print("Converted EEG shape (N, C, T):", X.shape)
    return X


# =========================================================
# VIDEO PKL LOADING
# =========================================================
def extract_video_trial_from_subject_array(subject_video_array, trial_index, num_frames=16, size=112):
    """
    Supported layouts:
    1) (N, T, H, W, C)   e.g. (280, 25, 56, 56, 3)
    2) (N, T, C, H, W)
    3) (T, H, W, C)      one trial only
    4) (T, C, H, W)      one trial only

    Returns:
        video tensor of shape (T, C, H, W)
    """
    arr = subject_video_array

    # -----------------------------------------------------
    # Case 1: subject-level storage with trial axis first
    # Example: (280, 25, 56, 56, 3)
    # -----------------------------------------------------
    if arr.ndim == 5:
        # (N, T, H, W, C)
        if arr.shape[-1] in [1, 3]:
            if trial_index >= arr.shape[0]:
                raise IndexError(
                    f"trial_index {trial_index} out of range for video array with first dimension {arr.shape[0]}"
                )
            trial = arr[trial_index]   # -> (T, H, W, C)

        # (N, T, C, H, W)
        elif arr.shape[2] in [1, 3]:
            if trial_index >= arr.shape[0]:
                raise IndexError(
                    f"trial_index {trial_index} out of range for video array with first dimension {arr.shape[0]}"
                )
            trial = arr[trial_index]   # -> (T, C, H, W)
            trial = np.transpose(trial, (0, 2, 3, 1))  # -> (T, H, W, C)

        else:
            raise ValueError(f"Unsupported 5D subject video shape: {arr.shape}")

    # -----------------------------------------------------
    # Case 2: already one trial
    # -----------------------------------------------------
    elif arr.ndim == 4:
        # (T, H, W, C)
        if arr.shape[-1] in [1, 3]:
            trial = arr

        # (T, C, H, W)
        elif arr.shape[1] in [1, 3]:
            trial = np.transpose(arr, (0, 2, 3, 1))

        else:
            raise ValueError(f"Unsupported 4D trial shape: {arr.shape}")

    else:
        raise ValueError(f"Unsupported video array ndim: {arr.ndim}, shape: {arr.shape}")

    # -----------------------------------------------------
    # Sample frames
    # -----------------------------------------------------
    T = trial.shape[0]
    if T <= 0:
        raise ValueError(f"Trial has no frames. Trial shape: {trial.shape}")

    idxs = np.linspace(0, T - 1, num_frames, dtype=int)

    frames = []
    for i in idxs:
        frame = trial[i]
        frame = frame.astype(np.float32)

        if frame.max() > 1.0:
            frame = frame / 255.0

        if frame.shape[-1] == 1:
            frame = np.repeat(frame, 3, axis=-1)

        frame = cv2.resize(frame, (size, size))
        frames.append(frame)

    frames = np.stack(frames, axis=0)             # (T, H, W, C)
    frames = np.transpose(frames, (0, 3, 1, 2))  # (T, C, H, W)

    return frames.astype(np.float32)


# =========================================================
# LOAD SUBJECTS
# =========================================================
def load_subjects(subject_ids):
    all_X_eeg = []
    all_y = []
    all_metadata = []

    for subject_id in subject_ids:
        base_subject_dir = os.path.join(BASE_DATA_DIR, f"subject{subject_id}")
        video_dir = os.path.join(base_subject_dir, "Video")
        eeg_pkl_path = get_subject_eeg_pkl_path(subject_id)

        metadata = build_subject_metadata(subject_id, video_dir)
        X_eeg = load_subject_eeg_pkl_trials(eeg_pkl_path)
        y = np.array([row["label"] for row in metadata], dtype=np.int64)

        if X_eeg.shape[0] < len(metadata):
            raise ValueError(
                f"Subject {subject_id}: EEG pkl has only {X_eeg.shape[0]} trials but metadata expects {len(metadata)}"
            )

        # use only first 200 to match metadata/video
        X_eeg = X_eeg[:len(metadata)]

        all_X_eeg.append(X_eeg)
        all_y.append(y)
        all_metadata.extend(metadata)

    all_X_eeg = np.concatenate(all_X_eeg, axis=0)
    all_y = np.concatenate(all_y, axis=0)

    return all_X_eeg, all_y, all_metadata


# =========================================================
# EEG PREPROCESS
# =========================================================
def standardize_train_test_eeg(X_train, X_test):
    mean = X_train.mean(axis=(0, 2), keepdims=True)
    std = X_train.std(axis=(0, 2), keepdims=True)
    std[std < 1e-6] = 1e-6

    X_train = (X_train - mean) / std
    X_test = (X_test - mean) / std
    return X_train, X_test, mean, std


# =========================================================
# DATASET
# =========================================================
class EEGVideoDataset(Dataset):
    def __init__(self, X_eeg, y, metadata):
        self.X_eeg = torch.tensor(X_eeg, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)
        self.metadata = metadata
        self.video_cache = {}

    def __len__(self):
        return len(self.X_eeg)

    def _get_subject_video_array(self, pkl_path):
        if pkl_path not in self.video_cache:
            self.video_cache[pkl_path] = load_pkl_array(pkl_path, kind="Vision")
        return self.video_cache[pkl_path]

    def __getitem__(self, idx):
        eeg = self.X_eeg[idx]   # (C, T)
        label = self.y[idx]
        meta = self.metadata[idx]

        subject_video_array = self._get_subject_video_array(meta["vision_pkl_path"])

        video = extract_video_trial_from_subject_array(
            subject_video_array,
            trial_index=meta["trial_index"],
            num_frames=NUM_VIDEO_FRAMES,
            size=VIDEO_SIZE
        )

        video = torch.tensor(video, dtype=torch.float32)
        return eeg, video, label


# =========================================================
# EEG ENCODER
# =========================================================
class EEGEncoder(nn.Module):
    def __init__(self, num_channels=30, out_dim=128):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv1d(num_channels, 64, kernel_size=7, padding=3),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2),

            nn.Conv1d(64, 128, kernel_size=5, padding=2),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(2),

            nn.Conv1d(128, 256, kernel_size=5, padding=2),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.MaxPool1d(2),

            nn.Conv1d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU(),

            nn.AdaptiveAvgPool1d(1),
        )

        self.proj = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, out_dim),
            nn.ReLU()
        )

    def forward(self, x):
        x = self.features(x)
        x = self.proj(x)
        return x


# =========================================================
# VIDEO ENCODER
# =========================================================
class VideoEncoder(nn.Module):
    def __init__(self, out_dim=128):
        super().__init__()

        self.frame_cnn = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=5, stride=2, padding=2),
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
            nn.AdaptiveAvgPool2d((1, 1)),
        )

        self.proj = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, out_dim),
            nn.ReLU()
        )

    def forward(self, video):
        # video: (B, T, C, H, W)
        B, T, C, H, W = video.shape
        video = video.view(B * T, C, H, W)

        frame_feats = self.frame_cnn(video)
        frame_feats = self.proj(frame_feats)
        frame_feats = frame_feats.view(B, T, -1)

        video_feat = frame_feats.mean(dim=1)
        return video_feat


# =========================================================
# MULTIMODAL MODEL
# =========================================================
class EEGVideoFusionNet(nn.Module):
    def __init__(self, eeg_dim=128, video_dim=128, num_classes=5):
        super().__init__()

        self.eeg_encoder = EEGEncoder(num_channels=30, out_dim=eeg_dim)
        self.video_encoder = VideoEncoder(out_dim=video_dim)

        self.classifier = nn.Sequential(
            nn.Linear(eeg_dim + video_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes),
        )

    def forward(self, eeg, video):
        eeg_feat = self.eeg_encoder(eeg)
        video_feat = self.video_encoder(video)
        fused = torch.cat([eeg_feat, video_feat], dim=1)
        out = self.classifier(fused)
        return out


# =========================================================
# TRAIN / EVAL
# =========================================================
def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_count = 0

    for eeg_batch, video_batch, y_batch in loader:
        eeg_batch = eeg_batch.to(device)
        video_batch = video_batch.to(device)
        y_batch = y_batch.to(device)

        optimizer.zero_grad()
        logits = model(eeg_batch, video_batch)
        loss = criterion(logits, y_batch)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * eeg_batch.size(0)
        preds = logits.argmax(dim=1)
        total_correct += (preds == y_batch).sum().item()
        total_count += y_batch.size(0)

    return total_loss / total_count, total_correct / total_count


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_count = 0

    all_true = []
    all_pred = []

    for eeg_batch, video_batch, y_batch in loader:
        eeg_batch = eeg_batch.to(device)
        video_batch = video_batch.to(device)
        y_batch = y_batch.to(device)

        logits = model(eeg_batch, video_batch)
        loss = criterion(logits, y_batch)

        preds = logits.argmax(dim=1)

        total_loss += loss.item() * eeg_batch.size(0)
        total_correct += (preds == y_batch).sum().item()
        total_count += y_batch.size(0)

        all_true.extend(y_batch.cpu().numpy())
        all_pred.extend(preds.cpu().numpy())

    return total_loss / total_count, total_correct / total_count, np.array(all_true), np.array(all_pred)


# =========================================================
# RANDOM TEST FUNCTION
# =========================================================
@torch.no_grad()
def predict_random_trial(model, dataset, device):
    model.eval()

    idx = random.randint(0, len(dataset) - 1)
    eeg, video, label = dataset[idx]

    eeg = eeg.unsqueeze(0).to(device)
    video = video.unsqueeze(0).to(device)

    logits = model(eeg, video)
    pred = int(torch.argmax(logits, dim=1).item())
    true = int(label.item())
    meta = dataset.metadata[idx]

    print("\nRandom Test Trial")
    print("-" * 60)
    print("Subject ID   :", meta["subject_id"])
    print("Global index :", meta["global_idx"])
    print("Trial no     :", meta["trial_no"])
    print("Mode         :", meta["mode"])
    print("EEG pkl      :", meta["eeg_pkl_path"])
    print("Vision pkl   :", meta["vision_pkl_path"])
    print("Trial index  :", meta["trial_index"])
    print("True emotion :", INDEX_TO_EMOTION[true])
    print("Pred emotion :", INDEX_TO_EMOTION[pred])
    print("-" * 60)


# =========================================================
# SPLIT PREP
# =========================================================
def prepare_data():
    if SPLIT_MODE == "subject_independent":
        print("Split mode: subject_independent")
        print("Train subjects:", TRAIN_SUBJECT_IDS)
        print("Test subjects :", TEST_SUBJECT_IDS)

        X_train_eeg, y_train, metadata_train = load_subjects(TRAIN_SUBJECT_IDS)
        X_test_eeg, y_test, metadata_test = load_subjects(TEST_SUBJECT_IDS)

    elif SPLIT_MODE == "random_mixed":
        print("Split mode: random_mixed")
        print("Subjects:", ALL_SUBJECT_IDS)

        X_eeg, y, metadata = load_subjects(ALL_SUBJECT_IDS)

        indices = np.arange(len(y))
        train_idx, test_idx = train_test_split(
            indices,
            test_size=TEST_SIZE,
            random_state=RANDOM_STATE,
            stratify=y
        )

        X_train_eeg = X_eeg[train_idx]
        X_test_eeg = X_eeg[test_idx]
        y_train = y[train_idx]
        y_test = y[test_idx]

        metadata_train = [metadata[i] for i in train_idx]
        metadata_test = [metadata[i] for i in test_idx]

    else:
        raise ValueError(f"Unsupported SPLIT_MODE: {SPLIT_MODE}")

    X_train_eeg, X_test_eeg, mean, std = standardize_train_test_eeg(X_train_eeg, X_test_eeg)

    return X_train_eeg, X_test_eeg, y_train, y_test, metadata_train, metadata_test, mean, std


# =========================================================
# MAIN
# =========================================================
def main():
    print("Using device:", DEVICE)

    X_train_eeg, X_test_eeg, y_train, y_test, metadata_train, metadata_test, mean, std = prepare_data()

    print("Train EEG shape:", X_train_eeg.shape)
    print("Test EEG shape :", X_test_eeg.shape)

    train_dataset = EEGVideoDataset(X_train_eeg, y_train, metadata_train)
    test_dataset = EEGVideoDataset(X_test_eeg, y_test, metadata_test)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    model = EEGVideoFusionNet(eeg_dim=128, video_dim=128, num_classes=5).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    best_acc = 0.0
    best_state = None

    for epoch in range(1, EPOCHS + 1):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, DEVICE)
        test_loss, test_acc, _, _ = evaluate(model, test_loader, criterion, DEVICE)

        if test_acc > best_acc:
            best_acc = test_acc
            best_state = {k: v.cpu() for k, v in model.state_dict().items()}

        print(
            f"Epoch [{epoch}/{EPOCHS}] | "
            f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | "
            f"Test Loss: {test_loss:.4f} | Test Acc: {test_acc:.4f}"
        )

    print(f"\nBest Test Accuracy: {best_acc:.4f}")

    if best_state is not None:
        model.load_state_dict(best_state)

    _, _, y_true, y_pred = evaluate(model, test_loader, criterion, DEVICE)

    target_names = [INDEX_TO_EMOTION[i] for i in range(5)]
    print("\nClassification Report:\n")
    print(classification_report(y_true, y_pred, target_names=target_names, digits=4))

    print("Confusion Matrix:\n")
    print(confusion_matrix(y_true, y_pred))

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "mean": mean,
            "std": std,
            "split_mode": SPLIT_MODE,
            "train_subject_ids": TRAIN_SUBJECT_IDS if SPLIT_MODE == "subject_independent" else ALL_SUBJECT_IDS,
            "test_subject_ids": TEST_SUBJECT_IDS if SPLIT_MODE == "subject_independent" else "random_mixed",
            "emotion_to_index": EMOTION_TO_INDEX,
            "num_video_frames": NUM_VIDEO_FRAMES,
            "video_size": VIDEO_SIZE,
        },
        MODEL_SAVE_PATH,
    )
    print(f"\nSaved model: {MODEL_SAVE_PATH}")

    predict_random_trial(model, test_dataset, DEVICE)


if __name__ == "__main__":
    main()
