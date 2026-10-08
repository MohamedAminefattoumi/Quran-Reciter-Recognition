
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

UPLOAD_DIR = os.path.join(BASE_DIR, "uploads")
PROCESSED_DIR = os.path.join(BASE_DIR, "processed")

MODELS_DIR = os.path.join(BASE_DIR, "backend", "models")

EMBEDDINGS_PATH = os.path.join(MODELS_DIR, "embeddings.pt")
RECITER_MODEL_PATH = os.path.join(MODELS_DIR, "reciter_model.pt")

TARGET_SAMPLE_RATE = 16000
SEGMENT_DURATION = 5
SEGMENT_SAMPLES = TARGET_SAMPLE_RATE * SEGMENT_DURATION

SIMILARITY_THRESHOLD = 0.64

os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(PROCESSED_DIR, exist_ok=True)
