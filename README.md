# 🕌 Quran Reciter Recognition

Identify Quran reciters from audio/video files using **ECAPA-TDNN** speaker embeddings and **open-set recognition** (the system can recognize known reciters or reject unknown voices).

Includes both an end-to-end **Machine Learning pipeline** and a **FastAPI + Web UI interface** with audio/video file processing and segmentation.

---

## 📁 Project Structure

```
Reciter/
├── ml/                          # Training, extraction & evaluation scripts
│   ├── prepare_dataset.py       # Download & segment the 20 known reciters
│   ├── prepare_unknown_dataset.py  # Download & segment 5 unknown reciters
│   ├── extract_embeddings.py    # Extract ECAPA-TDNN embeddings → embeddings.pt
│   ├── evaluate.py              # Open-set evaluation & threshold calibration
│   └── explore.py               # Exploratory data analysis
│
├── backend/                     # FastAPI backend
│   ├── main.py                  # App entry point & static files mounting
│   ├── config.py                # Configuration (paths, threshold, sample rate)
│   ├── api/
│   │   └── routes.py            # /predict API endpoint
│   ├── services/
│   │   ├── predictor.py         # Embedding extraction & cosine voting
│   │   ├── audio.py             # 5s audio chunking & preprocessing
│   │   └── video.py             # Audio extraction from video via FFmpeg
│   └── models/                  # Saved embeddings (gitignored)
│       └── .gitkeep
│
├── frontend/                    # Web Interface
│   ├── index.html
│   ├── style.css
│   └── script.js
│
├── dataset/                     # Known reciters audio (gitignored)
├── unknown_dataset/             # Unknown reciters audio (gitignored)
├── processed/                   # Audio segments cache (gitignored)
├── uploads/                     # Uploaded files cache (gitignored)
├── pretrained_models/           # ECAPA-TDNN weights cache (gitignored)
└── outputs/                     # Benchmark CSV results (gitignored)
```

---

## 🌐 Running the Web Application

To launch the web interface and API:

```bash
uvicorn backend.main:app --reload --port 8000
```

Then open your browser at **`http://localhost:8000`**. You can upload an audio (`.mp3`, `.wav`, etc.) or video (`.mp4`, `.mkv`, etc.) file to get reciter predictions segment by segment.

---

## 🚀 ML Pipeline

If you want to reproduce the dataset and benchmark from scratch:

```bash
# 1. Download and prepare the known dataset (20 reciters, 5 surahs each)
python ml/prepare_dataset.py

# 2. Download and prepare the unknown dataset (5 unseen reciters)
python ml/prepare_unknown_dataset.py

# 3. Extract ECAPA-TDNN embeddings from the known dataset
python ml/extract_embeddings.py
#    → Saves: backend/models/embeddings.pt

# 4. Run open-set evaluation: calibrate threshold & benchmark unknown rejection
python ml/evaluate.py
#    → Saves: backend/models/reciter_model.pt
#    → Saves: outputs/threshold_analysis.csv
#    → Saves: outputs/unknown_reciter_results.csv
```

---

## 🧠 How It Works

```
Input Audio / Extracted from Video (16kHz mono)
                    ↓
Split into 5-second segments
                    ↓
ECAPA-TDNN (speechbrain/spkrec-ecapa-voxceleb)
                    ↓
Speaker Embedding [192 dimensions, L2-normalized]
                    ↓
Cosine Similarity vs Known Reference Embeddings
                    ↓
Segment Prediction (Best Score ≥ Threshold ? Reciter : "Unknown")
                    ↓
Aggregation & Majority Voting across all segments
```

---

## 📦 Installation

```bash
pip install -r requirements.txt
```

> **FFmpeg prerequisite:** Ensure `ffmpeg` is installed and available in your system `PATH` for video audio extraction.

---

## 📊 Dataset Overview

- **20 known reciters** — Hafs recitation across 5 surahs (Ya-Sin, Al-Rahman, Al-Mulk, An-Naba, Al-Kahf)
- **5-second clips** sampled at 16kHz mono
- **5 unseen reciters** used strictly for open-set false acceptance / rejection benchmarking
- Data source: [mp3quran.net API](https://www.mp3quran.net/api/v3/reciters?language=eng)
