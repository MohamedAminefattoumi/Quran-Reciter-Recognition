
import os
import librosa
import soundfile as sf

from backend.config import (
    TARGET_SAMPLE_RATE,
    SEGMENT_SAMPLES
)

def preprocess_audio(file_path, output_dir):
    audio, _ = librosa.load(
        file_path,
        sr=TARGET_SAMPLE_RATE,
        mono=True
    )

    os.makedirs(output_dir, exist_ok=True)

    segments = []

    for start in range(0, len(audio), SEGMENT_SAMPLES):
        segment = audio[start:start + SEGMENT_SAMPLES]

        if len(segment) < SEGMENT_SAMPLES:
            break

        segment_number = len(segments) + 1

        segment_path = os.path.join(
            output_dir,
            f"segment_{segment_number:03d}.wav"
        )

        sf.write(
            segment_path,
            segment,
            TARGET_SAMPLE_RATE
        )

        segments.append(segment_path)

    return segments

