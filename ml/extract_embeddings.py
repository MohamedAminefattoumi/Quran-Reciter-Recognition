import os
from pathlib import Path
import shutil
import torch

# Fix: SpeechBrain uses symlinks by default which requires admin on Windows.
# We patch its internal function to use file copy instead.
import speechbrain.utils.fetching as _sb_fetch
def _copy_strategy(src, dst, local_strategy):
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    shutil.copy2(str(src), str(dst))
_sb_fetch.link_with_strategy = _copy_strategy
import torch.nn.functional as F
import librosa
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import Dataset, DataLoader
from speechbrain.inference.speaker import EncoderClassifier
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = str(ROOT / "dataset")
SAMPLE_RATE = 16000
AUDIO_LENGTH = 5 * SAMPLE_RATE
BATCH_SIZE = 16
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print("=" * 70)
print("ECAPA-TDNN - EXTRACT KNOWN RECITER EMBEDDINGS")
print("=" * 70)
print(f"Available device: {DEVICE}")

surahs = ["036", "055", "067", "078", "018"]

train_surahs, test_surahs = train_test_split(
    surahs,
    test_size=0.2,
    random_state=42
)

print("\nTrain surahs:", train_surahs)
print("Test surahs :", test_surahs)

X_train, y_train = [], []
X_test, y_test = [], []

for reciter in os.listdir(DATASET_DIR):
    reciter_folder = os.path.join(DATASET_DIR, reciter)

    if not os.path.isdir(reciter_folder):
        continue

    for filename in os.listdir(reciter_folder):
        if not filename.lower().endswith(".wav"):
            continue

        path = os.path.join(reciter_folder, filename)

        if any(filename.startswith(f"s{s}") for s in train_surahs):
            X_train.append(path)
            y_train.append(reciter)
        elif any(filename.startswith(f"s{s}") for s in test_surahs):
            X_test.append(path)
            y_test.append(reciter)

print(f"\nTrain clips : {len(X_train)}")
print(f"Test clips  : {len(X_test)}")

label_encoder = LabelEncoder()
y_train_encoded = label_encoder.fit_transform(y_train)
y_test_encoded = label_encoder.transform(y_test)

print(f"\nClasses ({len(label_encoder.classes_)}):")
for index, class_name in enumerate(label_encoder.classes_):
    print(f"{index:2d} -> {class_name}")


class AudioDataset(Dataset):
    def __init__(self, paths, labels):
        self.paths = paths
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        audio, _ = librosa.load(
            self.paths[idx],
            sr=SAMPLE_RATE,
            mono=True
        )

        audio = torch.tensor(audio, dtype=torch.float32)

        if len(audio) < AUDIO_LENGTH:
            audio = F.pad(
                audio,
                (0, AUDIO_LENGTH - len(audio))
            )
        else:
            audio = audio[:AUDIO_LENGTH]

        audio = (audio - audio.mean()) / (audio.std() + 1e-7)

        return audio, self.labels[idx]


train_loader = DataLoader(
    AudioDataset(X_train, y_train_encoded),
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0
)

test_loader = DataLoader(
    AudioDataset(X_test, y_test_encoded),
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0
)

print("\n" + "=" * 70)
print("LOADING ECAPA-TDNN")
print("=" * 70)

encoder = EncoderClassifier.from_hparams(
    source="speechbrain/spkrec-ecapa-voxceleb",
    savedir=str(ROOT / "pretrained_models" / "ecapa"),
    run_opts={"device": DEVICE},
    local_strategy="copy"
)

encoder.eval()

print("ECAPA-TDNN loaded.")


def extract_embeddings(dataloader, description):
    all_embeddings = []
    all_labels = []

    for audio, labels in tqdm(dataloader, desc=description):
        audio = audio.to(DEVICE)

        with torch.no_grad():
            embeddings = encoder.encode_batch(audio)
            embeddings = embeddings.squeeze(1)
            embeddings = F.normalize(
                embeddings,
                p=2,
                dim=1
            )

        all_embeddings.append(embeddings.cpu())
        all_labels.append(labels)

    return (
        torch.cat(all_embeddings),
        torch.cat(all_labels)
    )


print("\n" + "=" * 70)
print("EXTRACTING TRAIN EMBEDDINGS")
print("=" * 70)

train_embeddings, train_labels = extract_embeddings(
    train_loader,
    "Train"
)

print("\n" + "=" * 70)
print("EXTRACTING TEST EMBEDDINGS")
print("=" * 70)

test_embeddings, test_labels = extract_embeddings(
    test_loader,
    "Test"
)

print(f"\nTrain embeddings : {train_embeddings.shape}")
print(f"Test embeddings  : {test_embeddings.shape}")

print("\n" + "=" * 70)
print("1-NN EVALUATION")
print("=" * 70)

similarities = test_embeddings @ train_embeddings.T

best_indices = similarities.argmax(dim=1)

predictions = train_labels[best_indices]

scores = similarities.max(dim=1).values

accuracy = (
    predictions == test_labels
).float().mean().item()

print(f"ECAPA-TDNN 1-NN accuracy: {accuracy * 100:.2f}%")

print(f"Mean similarity: {scores.mean().item():.4f}")
print(f"Median similarity: {scores.median().item():.4f}")
print(f"Minimum similarity: {scores.min().item():.4f}")
print(f"Maximum similarity: {scores.max().item():.4f}")

print("\n" + "=" * 70)
print("SAVING EMBEDDINGS")
print("=" * 70)

torch.save(
    {
        "train_embeddings": train_embeddings,
        "train_labels": train_labels,
        "test_embeddings": test_embeddings,
        "test_labels": test_labels,
        "classes": label_encoder.classes_,
        "model": "speechbrain/spkrec-ecapa-voxceleb",
        "sample_rate": SAMPLE_RATE,
        "audio_length": AUDIO_LENGTH,
        "method": "ECAPA-TDNN + L2 normalization"
    },
    str(ROOT / "backend" / "models" / "embeddings.pt")
)

print("Saved: embeddings.pt")

print("\n" + "=" * 70)
print("EXTRACTION FINISHED")
print("=" * 70)
