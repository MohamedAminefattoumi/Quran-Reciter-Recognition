
import os
from fastapi import APIRouter, UploadFile, File

from backend.config import UPLOAD_DIR, PROCESSED_DIR
from backend.services.audio import preprocess_audio
from backend.services.video import extract_audio
from backend.services.predictor import (
    predict_reciter,
    aggregate_predictions
)

router = APIRouter()

AUDIO_EXTENSIONS = {
    ".mp3",
    ".wav",
    ".m4a",
    ".aac",
    ".flac"
}

VIDEO_EXTENSIONS = {
    ".mp4",
    ".mov",
    ".avi",
    ".mkv",
    ".webm"
}

@router.post("/predict")
async def predict(file: UploadFile = File(...)):
    filename = os.path.basename(file.filename)
    extension = os.path.splitext(filename)[1].lower()

    if extension not in AUDIO_EXTENSIONS and extension not in VIDEO_EXTENSIONS:
        raise ValueError(
            f"Unsupported file type: {extension}"
        )

    file_path = os.path.join(
        UPLOAD_DIR,
        filename
    )

    content = await file.read()

    with open(file_path, "wb") as f:
        f.write(content)

    base_name = os.path.splitext(filename)[0]

    output_dir = os.path.join(
        PROCESSED_DIR,
        base_name
    )

    os.makedirs(output_dir, exist_ok=True)

    audio_path = file_path

    if extension in VIDEO_EXTENSIONS:
        audio_path = os.path.join(
            output_dir,
            "extracted_audio.wav"
        )

        extract_audio(
            file_path,
            audio_path
        )

    segments = preprocess_audio(
        audio_path,
        output_dir
    )

    results = []

    for segment_path in segments:
        result = predict_reciter(segment_path)

        results.append({
            "segment": os.path.basename(segment_path),
            "prediction": result["prediction"],
            "score": result["score"]
        })

    if os.path.exists(file_path):
        os.remove(file_path)

    if extension in VIDEO_EXTENSIONS and os.path.exists(audio_path):
        os.remove(audio_path)

    final_prediction = aggregate_predictions(results)

    return {
        "filename": filename,
        "segments_created": len(segments),
        "prediction": final_prediction["prediction"],
        "summary": final_prediction,
        "results": results
    }