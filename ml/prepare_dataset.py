import os
from pathlib import Path
import re
import time

import librosa
import requests
import soundfile as sf
import subprocess

# ----------------------------- Configuration -----------------------------
API_URL = "https://www.mp3quran.net/api/v3/reciters?language=eng"
OUT_DIR = str(Path(__file__).resolve().parent.parent / "dataset")
TMP_DIR = str(Path(__file__).resolve().parent.parent / "tmp_audio")
FFMPEG = r"C:\Users\Msi\AppData\Local\Microsoft\WinGet\Packages\Gyan.FFmpeg_Microsoft.Winget.Source_8wekyb3d8bbwe\ffmpeg-9.0.2-full_build\bin\ffmpeg.exe"

SR = 16000                       # target sample rate
CLIP_SEC = 5                     # clip length in seconds
CLIP_LEN = SR * CLIP_SEC
SURAHS = [36, 55, 67, 78, 18]    # surahs to download per reciter
MAX_CLIPS = 600                  # max clips kept per reciter
MAX_RECITERS = 20                # start small to test, then raise it
SILENCE_RMS = 0.005              # clips below this energy are skipped
# ------------------------------------------------------------------------


def slugify(text):
    return re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")


def count_clips(folder):
    return len([f for f in os.listdir(folder) if f.endswith(".wav")])


def get_reciters():
    """Return (id, name, server) for reciters with a Hafs recitation
    that contains all the surahs we need."""
    resp = requests.get(API_URL, timeout=30)
    resp.raise_for_status()
    data = resp.json()

    reciters = []
    for r in data["reciters"]:
        for m in r["moshaf"]:
            available = set(m["surah_list"].split(","))
            if "hafs" in m["name"].lower() and all(str(s) in available for s in SURAHS):
                reciters.append((r["id"], r["name"], m["server"]))
                break
    return reciters


def download_file(url, dest, retries=3):
    """Stream the download and retry on failure."""
    headers = {"User-Agent": "Mozilla/5.0"}
    for attempt in range(1, retries + 1):
        try:
            # (connect timeout, read timeout between chunks)
            with requests.get(url, stream=True, timeout=(15, 120), headers=headers) as resp:
                resp.raise_for_status()
                with open(dest, "wb") as f:
                    for chunk in resp.iter_content(chunk_size=1024 * 256):
                        f.write(chunk)
            return True
        except Exception as e:
            print(f"  attempt {attempt}/{retries} failed for {url}: {e}")
            if os.path.exists(dest):
                os.remove(dest)
            time.sleep(3 * attempt)  # wait longer after each failure
    return False


def split_and_save(mp3_path, out_dir, prefix, budget):
    """Convert MP3 to WAV with FFmpeg, then cut it into fixed-length clips."""

    wav_path = mp3_path.replace(".mp3", ".wav")

    try:
        # Convert MP3 to 16 kHz mono WAV
        subprocess.run(
            [
                FFMPEG,
                "-y",
                "-i", mp3_path,
                "-ar", str(SR),
                "-ac", "1",
                wav_path
            ],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL
        )

        # Load the converted WAV
        y, _ = librosa.load(wav_path, sr=SR, mono=True)

        saved = 0

        for i in range(len(y) // CLIP_LEN):
            if saved >= budget:
                break

            clip = y[i * CLIP_LEN:(i + 1) * CLIP_LEN]

            # Skip near-silent clips
            if librosa.feature.rms(y=clip).mean() < SILENCE_RMS:
                continue

            sf.write(
                os.path.join(out_dir, f"{prefix}_{i:04d}.wav"),
                clip,
                SR
            )

            saved += 1

        return saved

    finally:
        # Remove temporary converted WAV
        if os.path.exists(wav_path):
            os.remove(wav_path)

def main():
    os.makedirs(TMP_DIR, exist_ok=True)
    os.makedirs(OUT_DIR, exist_ok=True)

    reciters = get_reciters()[:MAX_RECITERS]
    print(f"{len(reciters)} reciters selected")

    for rid, name, server in reciters:
        out_dir = os.path.join(OUT_DIR, f"{rid}_{slugify(name)}")
        os.makedirs(out_dir, exist_ok=True)

        # Resume support: skip reciters that are already complete
        if count_clips(out_dir) >= MAX_CLIPS:
            print(f"{name}: already done, skipping")
            continue

        total = 0
        for s in SURAHS:
            if total >= MAX_CLIPS:
                break

            url = f"{server}{s:03d}.mp3"
            tmp = os.path.join(TMP_DIR, f"{rid}_{s:03d}.mp3")

            if not download_file(url, tmp):
                print(f"  skipping {url}")
                continue

            try:
                total += split_and_save(
                    tmp,
                    out_dir,
                    f"s{s:03d}",
                    MAX_CLIPS - total
                )
            except Exception as e:
                print(f"  failed to process {tmp}: {e}")
            finally:
                if os.path.exists(tmp):
                    os.remove(tmp)

        print(f"{name}: {total} clips")

    # After all reciters are processed
    print("Done.")

    # Delete temporary folder if it is empty
    if os.path.exists(TMP_DIR) and not os.listdir(TMP_DIR):
        os.rmdir(TMP_DIR)


if __name__ == "__main__":
    main()