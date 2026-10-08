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

OUT_DIR = str(Path(__file__).resolve().parent.parent / "unknown_dataset")

TMP_DIR = str(Path(__file__).resolve().parent.parent / "tmp_audio")

FFMPEG = r"C:\Users\Msi\AppData\Local\Microsoft\WinGet\Packages\Gyan.FFmpeg_Microsoft.Winget.Source_8wekyb3d8bbwe\ffmpeg-9.0.2-full_build\bin\ffmpeg.exe"

SR = 16000
CLIP_SEC = 5
CLIP_LEN = SR * CLIP_SEC

SURAHS = [36, 55, 67, 78, 18]

MAX_CLIPS = 600

# Number of NEW reciters for the unknown dataset
MAX_UNKNOWN_RECITERS = 5

SILENCE_RMS = 0.005


# ------------------------------------------------------------------------
# IMPORTANT:
# These are the 20 reciters already present in your KNOWN dataset.
# They must NOT appear in the unknown dataset.
# ------------------------------------------------------------------------

KNOWN_RECITERS = {
    "100_majed_al_enezi",
    "102_maher_al_meaqli",
    "106_mohammad_al_tablaway",
    "107_mohammed_al_lohaidan",
    "108_mohammed_al_muhasny",
    "109_mohammed_ayyub",
    "10_akram_alalaqmi",
    "110_mohammad_saleh_alim_shah",
    "111_mohammed_jibreel",
    "112_mohammed_siddiq_al_minshawi",
    "115_mohammad_abdullkarem",
    "118_mahmoud_khalil_al_hussary",
    "121_mahmoud_ali_albanna",
    "123_mishary_alafasi",
    "125_mustafa_ismail",
    "126_mustafa_al_lahoni",
    "127_mustafa_raad_alazawy",
    "129_muftah_alsaltany",
    "12_idrees_abkr",
    "1_ibrahim_al_akdar",
}


# ------------------------------------------------------------------------


def slugify(text):
    return re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")


def count_clips(folder):

    if not os.path.exists(folder):
        return 0

    return len([
        f
        for f in os.listdir(folder)
        if f.endswith(".wav")
    ])


def get_reciters():

    """
    Get reciters from MP3Quran who have a Hafs recitation
    containing all required surahs.
    """

    print("Getting reciters from MP3Quran API...")

    resp = requests.get(API_URL,timeout=30)

    resp.raise_for_status()

    data = resp.json()

    reciters = []

    for r in data["reciters"]:

        for m in r["moshaf"]:

            available = set(m["surah_list"].split(","))
            if("hafs" in m["name"].lower()and all(str(s) in available for s in SURAHS)):
                reciters.append(
                    (r["id"],r["name"],m["server"])
                )
                break
    return reciters


def download_file(url, dest, retries=3):

    """
    Download MP3 with retries.
    """

    headers = {
        "User-Agent": "Mozilla/5.0"
    }

    for attempt in range(
        1,
        retries + 1
    ):

        try:

            with requests.get(
                url,
                stream=True,
                timeout=(15, 120),
                headers=headers
            ) as resp:

                resp.raise_for_status()

                with open(
                    dest,
                    "wb"
                ) as f:

                    for chunk in resp.iter_content(
                        chunk_size=1024 * 256
                    ):

                        f.write(chunk)

            return True

        except Exception as e:

            print(
                f"  attempt "
                f"{attempt}/{retries} failed: {e}"
            )

            if os.path.exists(dest):
                os.remove(dest)

            time.sleep(3 * attempt)

    return False


def split_and_save(
    mp3_path,
    out_dir,
    prefix,
    budget
):

    """
    Convert MP3 to 16 kHz mono WAV,
    then split into 5-second clips.
    """

    wav_path = mp3_path.replace(
        ".mp3",
        ".wav"
    )

    try:

        # ------------------------------------------------
        # Convert MP3 -> 16 kHz mono WAV
        # ------------------------------------------------

        subprocess.run(
            [
                FFMPEG,
                "-y",
                "-i",
                mp3_path,
                "-ar",
                str(SR),
                "-ac",
                "1",
                wav_path
            ],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL
        )

        # ------------------------------------------------
        # Load WAV
        # ------------------------------------------------

        y, _ = librosa.load(
            wav_path,
            sr=SR,
            mono=True
        )

        saved = 0

        # ------------------------------------------------
        # Split into 5-second clips
        # ------------------------------------------------

        for i in range(
            len(y) // CLIP_LEN
        ):

            if saved >= budget:
                break

            clip = y[
                i * CLIP_LEN:
                (i + 1) * CLIP_LEN
            ]

            # --------------------------------------------
            # Skip near-silent clips
            # --------------------------------------------

            if (
                librosa.feature.rms(
                    y=clip
                ).mean()
                < SILENCE_RMS
            ):
                continue

            output_path = os.path.join(
                out_dir,
                f"{prefix}_{i:04d}.wav"
            )

            sf.write(
                output_path,
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

    # ----------------------------------------------------
    # Create folders
    # ----------------------------------------------------

    os.makedirs(
        TMP_DIR,
        exist_ok=True
    )

    os.makedirs(
        OUT_DIR,
        exist_ok=True
    )

    # ----------------------------------------------------
    # Get all suitable reciters
    # ----------------------------------------------------

    all_reciters = get_reciters()

    print(
        f"\nTotal suitable reciters found: "
        f"{len(all_reciters)}"
    )

    # ----------------------------------------------------
    # Remove the 20 known reciters
    # ----------------------------------------------------

    unknown_candidates = []

    for rid, name, server in all_reciters:

        reciter_folder = (
            f"{rid}_{slugify(name)}"
        )

        if reciter_folder in KNOWN_RECITERS:

            print(
                f"KNOWN - excluded: {reciter_folder}"
            )

            continue

        unknown_candidates.append(
            (
                rid,
                name,
                server
            )
        )

    print(
        f"\nUnknown candidates available: "
        f"{len(unknown_candidates)}"
    )

    # ----------------------------------------------------
    # Select first 5 unknown reciters
    # ----------------------------------------------------

    selected_reciters = (
        unknown_candidates[
            :MAX_UNKNOWN_RECITERS
        ]
    )

    print(
        "\n" + "=" * 70
    )

    print(
        "SELECTED UNKNOWN RECITERS"
    )

    print(
        "=" * 70
    )

    for rid, name, server in selected_reciters:

        print(
            f"{rid} - {name}"
        )

    print(
        "=" * 70
    )

    # ----------------------------------------------------
    # Download and process each unknown reciter
    # ----------------------------------------------------

    for rid, name, server in selected_reciters:

        folder_name = (
            f"{rid}_{slugify(name)}"
        )

        out_dir = os.path.join(
            OUT_DIR,
            folder_name
        )

        os.makedirs(
            out_dir,
            exist_ok=True
        )

        # ------------------------------------------------
        # Resume support
        # ------------------------------------------------

        existing = count_clips(
            out_dir
        )

        if existing >= MAX_CLIPS:

            print(
                f"\n{name}: "
                f"already complete "
                f"({existing} clips), skipping."
            )

            continue

        print(
            "\n" + "=" * 70
        )

        print(
            f"Processing UNKNOWN reciter: {name}"
        )

        print(
            f"Folder: {folder_name}"
        )

        print(
            f"Existing clips: {existing}"
        )

        print(
            "=" * 70
        )

        total = existing

        # ------------------------------------------------
        # Download the five surahs
        # ------------------------------------------------

        for s in SURAHS:

            if total >= MAX_CLIPS:
                break

            url = (
                f"{server}"
                f"{s:03d}.mp3"
            )

            tmp = os.path.join(
                TMP_DIR,
                f"{rid}_{s:03d}.mp3"
            )

            print(
                f"\nDownloading Surah {s}..."
            )

            if not download_file(
                url,
                tmp
            ):

                print(
                    f"  Failed to download: "
                    f"{url}"
                )

                continue

            try:

                saved = split_and_save(
                    tmp,
                    out_dir,
                    f"s{s:03d}",
                    MAX_CLIPS - total
                )

                total += saved

                print(
                    f"  Saved: {saved} clips"
                )

                print(
                    f"  Total: {total}/{MAX_CLIPS}"
                )

            except Exception as e:

                print(
                    f"  Failed to process "
                    f"{tmp}: {e}"
                )

            finally:

                if os.path.exists(tmp):

                    os.remove(tmp)

        print(
            f"\n{name}: "
            f"{total} clips created."
        )

    # ----------------------------------------------------
    # Cleanup temporary directory
    # ----------------------------------------------------

    print(
        "\n" + "=" * 70
    )

    print("Done.")

    print(
        "=" * 70
    )

    if (
        os.path.exists(TMP_DIR)
        and not os.listdir(TMP_DIR)
    ):

        os.rmdir(TMP_DIR)

        print(
            "Temporary folder removed."
        )

    print(
        f"\nUnknown dataset location:"
        f" {OUT_DIR}"
    )


if __name__ == "__main__":

    main()