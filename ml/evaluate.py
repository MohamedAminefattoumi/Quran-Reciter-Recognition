import os
from pathlib import Path
import shutil
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import torchaudio
import soundfile as sf
from torch.utils.data import Dataset, DataLoader
import speechbrain.utils.fetching as _sb_fetch
from speechbrain.inference.speaker import EncoderClassifier

def _copy_strategy(src, dst, local_strategy):
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    shutil.copy2(str(src), str(dst))

_sb_fetch.link_with_strategy = _copy_strategy

ROOT = Path(__file__).resolve().parent.parent
UNKNOWN_DATASET_DIR = str(ROOT / "unknown_dataset")
EMBEDDINGS_FILE = str(ROOT / "backend" / "models" / "embeddings.pt")
RECITER_MODEL_FILE = str(ROOT / "backend" / "models" / "reciter_model.pt")
SAMPLE_RATE = 16000
AUDIO_LENGTH = 5 * SAMPLE_RATE
BATCH_SIZE = 16
TARGET_UNKNOWN_REJECTION = 0.99
THRESHOLD = None

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("=" * 70)
print("QURAN RECITER RECOGNITION - ECAPA OPEN SET TEST")
print("=" * 70)
print(f"Device : {DEVICE}")
print(f"Target unknown rejection : {TARGET_UNKNOWN_REJECTION * 100:.2f}%")

print("\n" + "=" * 70)
print("LOADING KNOWN EMBEDDINGS")
print("=" * 70)

data = torch.load(EMBEDDINGS_FILE, weights_only=False)

train_embeddings = F.normalize(
    data["train_embeddings"].float(),
    p=2,
    dim=1
)
test_embeddings = F.normalize(
    data["test_embeddings"].float(),
    p=2,
    dim=1
)
train_labels = data["train_labels"]
test_labels = data["test_labels"]
classes = data["classes"]

print(f"Train embeddings : {train_embeddings.shape}")
print(f"Test embeddings  : {test_embeddings.shape}")

print("\n" + "=" * 70)
print("KNOWN RECITER TEST - 1-NN COSINE")
print("=" * 70)

known_similarities = test_embeddings @ train_embeddings.T
known_best_indices = known_similarities.argmax(dim=1)
known_predictions = train_labels[known_best_indices]
known_scores = known_similarities.max(dim=1).values

known_accuracy = (
    known_predictions == test_labels
).float().mean().item()

print(f"Known-reciter accuracy : {known_accuracy * 100:.2f}%")
print(f"Mean score (known)     : {known_scores.mean().item():.4f}")
print(f"Median score (known)   : {known_scores.median().item():.4f}")
print(f"Minimum score (known)  : {known_scores.min().item():.4f}")
print(f"Maximum score (known)  : {known_scores.max().item():.4f}")

class UnknownAudioDataset(Dataset):
    def __init__(self, root_dir):
        self.files = []
        for root, _, files in os.walk(root_dir):
            for file in files:
                if file.lower().endswith(".wav"):
                    self.files.append(os.path.join(root, file))
        self.files.sort()

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        path = self.files[idx]
        audio, sample_rate = sf.read(path, dtype="float32")
        audio = torch.tensor(audio, dtype=torch.float32)

        if audio.ndim > 1:
            audio = audio.mean(dim=1)

        if sample_rate != SAMPLE_RATE:
            audio = torchaudio.functional.resample(
                audio,
                sample_rate,
                SAMPLE_RATE
            )

        if len(audio) < AUDIO_LENGTH:
            audio = F.pad(
                audio,
                (0, AUDIO_LENGTH - len(audio))
            )
        else:
            audio = audio[:AUDIO_LENGTH]

        audio = (audio - audio.mean()) / (audio.std() + 1e-7)
        return audio, path

print("\n" + "=" * 70)
print("UNKNOWN DATASET")
print("=" * 70)

unknown_dataset = UnknownAudioDataset(UNKNOWN_DATASET_DIR)

if len(unknown_dataset) == 0:
    raise RuntimeError(
        f"No WAV files found in '{UNKNOWN_DATASET_DIR}'."
    )

unknown_loader = DataLoader(
    unknown_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0
)

print(f"Unknown audio files: {len(unknown_dataset)}")

print("\n" + "=" * 70)
print("LOADING ECAPA-TDNN")
print("=" * 70)

encoder = EncoderClassifier.from_hparams(
    source="speechbrain/spkrec-ecapa-voxceleb",
    savedir=str(ROOT / "pretrained_models" / "ecapa"),
    run_opts={"device": str(DEVICE)},
    local_strategy="copy"
)

encoder.eval()
print("ECAPA-TDNN loaded.")

print("\n" + "=" * 70)
print("EXTRACTING UNKNOWN EMBEDDINGS")
print("=" * 70)

unknown_embeddings_list = []
unknown_paths = []

with torch.no_grad():
    for batch_index, (audio, paths) in enumerate(unknown_loader):
        audio = audio.to(DEVICE)

        embeddings = encoder.encode_batch(audio)
        embeddings = embeddings.squeeze(1)
        embeddings = F.normalize(
            embeddings,
            p=2,
            dim=1
        )

        unknown_embeddings_list.append(embeddings.cpu())
        unknown_paths.extend(paths)

        if (
            (batch_index + 1) % 20 == 0
            or (batch_index + 1) == len(unknown_loader)
        ):
            processed = min(
                (batch_index + 1) * BATCH_SIZE,
                len(unknown_dataset)
            )
            print(
                f"Processed {processed}/{len(unknown_dataset)} "
                f"unknown files"
            )

unknown_embeddings = torch.cat(unknown_embeddings_list)

print(f"\nUnknown embeddings : {unknown_embeddings.shape}")

print("\n" + "=" * 70)
print("OPEN SET RECOGNITION")
print("=" * 70)

unknown_similarities = unknown_embeddings @ train_embeddings.T
unknown_best_indices = unknown_similarities.argmax(dim=1)
unknown_scores = unknown_similarities.max(dim=1).values
unknown_predictions = train_labels[unknown_best_indices]

known_scores_np = known_scores.cpu().numpy()
known_predictions_np = known_predictions.cpu().numpy()
test_labels_np = test_labels.cpu().numpy()
unknown_scores_np = unknown_scores.cpu().numpy()
unknown_predictions_np = unknown_predictions.cpu().numpy()

unknown_count = len(unknown_paths)

print(f"Total unknown clips : {unknown_count}")
print(
    f"Mean unknown score  : "
    f"{unknown_scores_np.mean():.4f}"
)
print(
    f"Median unknown score: "
    f"{np.median(unknown_scores_np):.4f}"
)
print(
    f"Minimum unknown score: "
    f"{unknown_scores_np.min():.4f}"
)
print(
    f"Maximum unknown score: "
    f"{unknown_scores_np.max():.4f}"
)

print("\n" + "=" * 70)
print("THRESHOLD ANALYSIS - KNOWN + UNKNOWN")
print("=" * 70)

threshold_results = []

for threshold in np.arange(0.30, 1.001, 0.01):
    threshold = round(float(threshold), 2)

    known_accepted = known_scores_np >= threshold
    known_rejected = ~known_accepted

    unknown_accepted = unknown_scores_np >= threshold
    unknown_rejected = ~unknown_accepted

    known_accepted_count = int(known_accepted.sum())
    known_rejected_count = int(known_rejected.sum())
    unknown_accepted_count = int(unknown_accepted.sum())
    unknown_rejected_count = int(unknown_rejected.sum())

    correct_known_accepted = (
        known_accepted
        & (known_predictions_np == test_labels_np)
    )

    correct_known_accepted_count = int(
        correct_known_accepted.sum()
    )

    wrong_known_accepted_count = (
        known_accepted_count
        - correct_known_accepted_count
    )

    known_coverage = (
        known_accepted_count / len(known_scores_np)
    )

    known_recognition_rate = (
        correct_known_accepted_count
        / len(known_scores_np)
    )

    known_precision = (
        correct_known_accepted_count
        / known_accepted_count
        if known_accepted_count > 0
        else 0.0
    )

    known_false_rejection_rate = (
        known_rejected_count / len(known_scores_np)
    )

    unknown_rejection_rate = (
        unknown_rejected_count / unknown_count
    )

    false_acceptance_rate = (
        unknown_accepted_count / unknown_count
    )

    threshold_results.append({
        "threshold": threshold,
        "known_accepted": known_accepted_count,
        "known_rejected": known_rejected_count,
        "known_coverage": known_coverage,
        "known_correct_accepted": correct_known_accepted_count,
        "known_wrong_accepted": wrong_known_accepted_count,
        "known_recognition_rate": known_recognition_rate,
        "known_precision": known_precision,
        "known_false_rejection_rate": known_false_rejection_rate,
        "unknown_accepted": unknown_accepted_count,
        "unknown_rejected": unknown_rejected_count,
        "unknown_rejection_rate": unknown_rejection_rate,
        "false_acceptance_rate": false_acceptance_rate
    })

    print(
        f"{threshold:.2f} | "
        f"Known coverage: {known_coverage * 100:6.2f}% | "
        f"Known precision: {known_precision * 100:6.2f}% | "
        f"Unknown rejection: {unknown_rejection_rate * 100:6.2f}% | "
        f"False acceptance: {false_acceptance_rate * 100:6.2f}%"
    )

threshold_df = pd.DataFrame(threshold_results)

eligible_thresholds = threshold_df[
    threshold_df["unknown_rejection_rate"]
    >= TARGET_UNKNOWN_REJECTION
]

if len(eligible_thresholds) > 0:
    best_row = eligible_thresholds.sort_values(
        by=[
            "known_recognition_rate",
            "known_precision",
            "threshold"
        ],
        ascending=[False, False, True]
    ).iloc[0]

    THRESHOLD = float(best_row["threshold"])
else:
    best_row = threshold_df.sort_values(
        by="unknown_rejection_rate",
        ascending=False
    ).iloc[0]

    THRESHOLD = float(best_row["threshold"])

print("\n" + "=" * 70)
print("RECOMMENDED THRESHOLD")
print("=" * 70)

print(f"Recommended threshold : {THRESHOLD:.2f}")
print(
    f"Known coverage        : "
    f"{best_row['known_coverage'] * 100:.2f}%"
)
print(
    f"Known recognition rate: "
    f"{best_row['known_recognition_rate'] * 100:.2f}%"
)
print(
    f"Known precision       : "
    f"{best_row['known_precision'] * 100:.2f}%"
)
print(
    f"Known false rejection : "
    f"{best_row['known_false_rejection_rate'] * 100:.2f}%"
)
print(
    f"Unknown rejection     : "
    f"{best_row['unknown_rejection_rate'] * 100:.2f}%"
)
print(
    f"False acceptance      : "
    f"{best_row['false_acceptance_rate'] * 100:.2f}%"
)

print("\n" + "=" * 70)
print("FINAL OPEN SET TEST")
print("=" * 70)

known_is_accepted = known_scores_np >= THRESHOLD
unknown_is_accepted = unknown_scores_np >= THRESHOLD

known_accepted_count = int(known_is_accepted.sum())
known_rejected_count = int((~known_is_accepted).sum())

known_correct_accepted = (
    known_is_accepted
    & (known_predictions_np == test_labels_np)
)

known_correct_accepted_count = int(
    known_correct_accepted.sum()
)

known_wrong_accepted_count = (
    known_accepted_count
    - known_correct_accepted_count
)

unknown_accepted_count = int(
    unknown_is_accepted.sum()
)

unknown_rejected_count = int(
    (~unknown_is_accepted).sum()
)

print(f"Threshold : {THRESHOLD:.2f}")
print()
print("KNOWN DATASET")
print(f"Total clips                  : {len(known_scores_np)}")
print(f"Accepted as known            : {known_accepted_count}")
print(f"Rejected as unknown          : {known_rejected_count}")
print(
    f"Known coverage               : "
    f"{known_accepted_count / len(known_scores_np) * 100:.2f}%"
)
print(
    f"Correctly recognized         : "
    f"{known_correct_accepted_count}"
)
print(
    f"Wrongly recognized           : "
    f"{known_wrong_accepted_count}"
)
print(
    f"Known recognition rate       : "
    f"{known_correct_accepted_count / len(known_scores_np) * 100:.2f}%"
)
print(
    f"Known precision              : "
    f"{known_correct_accepted_count / known_accepted_count * 100:.2f}%"
    if known_accepted_count > 0
    else "Known precision              : 0.00%"
)

print("\nUNKNOWN DATASET")
print(f"Total clips                  : {unknown_count}")
print(f"Rejected as unknown          : {unknown_rejected_count}")
print(f"Incorrectly accepted as known: {unknown_accepted_count}")
print(
    f"Unknown rejection rate       : "
    f"{unknown_rejected_count / unknown_count * 100:.2f}%"
)
print(
    f"False acceptance rate        : "
    f"{unknown_accepted_count / unknown_count * 100:.2f}%"
)

print("\n" + "=" * 70)
print("UNKNOWN SIMILARITY STATISTICS")
print("=" * 70)

print(f"Mean similarity    : {unknown_scores_np.mean():.4f}")
print(f"Median similarity  : {np.median(unknown_scores_np):.4f}")
print(f"Minimum similarity : {unknown_scores_np.min():.4f}")
print(f"Maximum similarity : {unknown_scores_np.max():.4f}")
print(f"Mean score (known) : {known_scores_np.mean():.4f}")
print(f"Median score (known): {np.median(known_scores_np):.4f}")

print("\n" + "=" * 70)
print("TOP 20 MOST SIMILAR UNKNOWN FILES")
print("=" * 70)

top_indices = np.argsort(unknown_scores_np)[::-1][:20]

for rank, idx in enumerate(top_indices, 1):
    predicted_name = classes[
        unknown_predictions_np[idx]
    ]

    open_set_label = (
        predicted_name
        if unknown_is_accepted[idx]
        else "Unknown"
    )

    print(
        f"{rank:2d}. "
        f"Score={unknown_scores_np[idx]:.4f} | "
        f"Prediction={open_set_label} | "
        f"{unknown_paths[idx]}"
    )

print("\n" + "=" * 70)
print("SAVING RESULTS")
print("=" * 70)

results = pd.DataFrame({
    "path": unknown_paths,
    "predicted_reciter": [
        classes[p]
        if unknown_is_accepted[i]
        else "Unknown"
        for i, p in enumerate(unknown_predictions_np)
    ],
    "similarity": unknown_scores_np,
    "threshold": THRESHOLD
})

results.to_csv(
    str(ROOT / "outputs" / "unknown_reciter_results.csv"),
    index=False,
    encoding="utf-8"
)

threshold_df.to_csv(
    str(ROOT / "outputs" / "threshold_analysis.csv"),
    index=False,
    encoding="utf-8"
)

torch.save(
    {
        "threshold": THRESHOLD,
        "known_accuracy_without_threshold": known_accuracy,
        "known_coverage": float(
            best_row["known_coverage"]
        ),
        "known_recognition_rate": float(
            best_row["known_recognition_rate"]
        ),
        "known_precision": float(
            best_row["known_precision"]
        ),
        "known_false_rejection_rate": float(
            best_row["known_false_rejection_rate"]
        ),
        "unknown_rejection_rate": float(
            best_row["unknown_rejection_rate"]
        ),
        "false_acceptance_rate": float(
            best_row["false_acceptance_rate"]
        ),
        "train_embeddings_shape": tuple(
            train_embeddings.shape
        ),
        "sample_rate": SAMPLE_RATE,
        "audio_length_seconds": 5,
        "method": (
            "ECAPA-TDNN + L2 normalization "
            "+ 1-NN cosine similarity + open-set threshold"
        )
    },
    RECITER_MODEL_FILE
)

print("Saved: unknown_reciter_results.csv")
print("Saved: threshold_analysis.csv")
print("Saved: reciter_model.pt")

print("\n" + "=" * 70)
print("OPEN SET TEST FINISHED")
print("=" * 70)
print(f"FINAL THRESHOLD : {THRESHOLD:.2f}")
print(
    f"KNOWN RECOGNITION RATE : "
    f"{best_row['known_recognition_rate'] * 100:.2f}%"
)
print(
    f"UNKNOWN REJECTION RATE : "
    f"{best_row['unknown_rejection_rate'] * 100:.2f}%"
)
print(
    f"FALSE ACCEPTANCE RATE : "
    f"{best_row['false_acceptance_rate'] * 100:.2f}%"
)
