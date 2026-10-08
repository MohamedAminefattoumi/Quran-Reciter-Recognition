import torch
import soundfile as sf
import torch.nn.functional as F
from speechbrain.inference.speaker import EncoderClassifier

from backend.config import (
    EMBEDDINGS_PATH,
    SIMILARITY_THRESHOLD
)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

classifier = EncoderClassifier.from_hparams(
    source="speechbrain/spkrec-ecapa-voxceleb",
    savedir="pretrained_models/ecapa",
    run_opts={"device": DEVICE}
)

embeddings_data = torch.load(
    EMBEDDINGS_PATH,
    map_location="cpu",
    weights_only=False
)

train_embeddings = embeddings_data["train_embeddings"]
train_labels = embeddings_data["train_labels"]
classes = embeddings_data["classes"]

train_embeddings = F.normalize(
    train_embeddings,
    p=2,
    dim=1
)

def extract_embedding(audio_path):
    audio, sample_rate = sf.read(audio_path)

    if len(audio.shape) > 1:
        audio = audio.mean(axis=1)

    signal = torch.tensor(
        audio,
        dtype=torch.float32
    ).unsqueeze(0)

    if sample_rate != 16000:
        raise ValueError(
            f"Expected 16000 Hz audio, got {sample_rate} Hz"
        )

    with torch.no_grad():
        embedding = classifier.encode_batch(signal)

    embedding = embedding.squeeze()

    embedding = F.normalize(
        embedding,
        p=2,
        dim=0
    )

    return embedding.cpu()

def predict_reciter(audio_path):
    embedding = extract_embedding(audio_path)

    similarities = torch.matmul(
        train_embeddings,
        embedding
    )

    best_index = torch.argmax(similarities).item()
    best_score = similarities[best_index].item()

    best_label = train_labels[best_index].item()
    reciter_name = classes[best_label]

    if best_score >= SIMILARITY_THRESHOLD:
        prediction = reciter_name
    else:
        prediction = "Unknown"

    return {
        "prediction": prediction,
        "score": round(best_score, 4)
    }


def aggregate_predictions(results):
    total_segments = len(results)

    if total_segments == 0:
        return {
            "prediction": "Unknown",
            "segments_total": 0,
            "segments_known": 0,
            "accepted_ratio": 0.0,
            "top_reciter_segments": 0,
            "top_reciter_ratio": 0.0,
            "average_score": 0.0
        }

    votes = {}
    scores = {}

    for result in results:
        prediction = result["prediction"]

        if prediction == "Unknown":
            continue

        votes[prediction] = votes.get(prediction, 0) + 1

        if prediction not in scores:
            scores[prediction] = []

        scores[prediction].append(result["score"])

    if not votes:
        return {
            "prediction": "Unknown",
            "segments_total": total_segments,
            "segments_known": 0,
            "accepted_ratio": 0.0,
            "top_reciter_segments": 0,
            "top_reciter_ratio": 0.0,
            "average_score": 0.0
        }

    best_reciter = max(votes, key=votes.get)
    best_votes = votes[best_reciter]

    accepted_segments = sum(votes.values())

    accepted_ratio = accepted_segments / total_segments
    top_reciter_ratio = best_votes / total_segments

    average_score = sum(scores[best_reciter]) / len(scores[best_reciter])

    if top_reciter_ratio >= 0.50:
        final_prediction = best_reciter
    else:
        final_prediction = "Unknown"

    return {
        "prediction": final_prediction,
        "segments_total": total_segments,
        "segments_known": accepted_segments,
        "accepted_ratio": round(accepted_ratio, 4),
        "top_reciter_segments": best_votes,
        "top_reciter_ratio": round(top_reciter_ratio, 4),
        "average_score": round(average_score, 4)
    }


