import librosa
import librosa.display
import os
import matplotlib.pyplot as plt
import numpy as np 

# ----------------------------- Configuration -----------------------------

DATASET_DIR = "dataset"

EXPECTED_SR = 16000
EXPECTED_DURATION = 5

# --------------------------------------------------------------------------


def get_audio_files(folder):
    """Return all WAV files inside a folder."""
    return [
        os.path.join(folder, file)
        for file in os.listdir(folder)
        if file.lower().endswith(".wav")
    ]


def check_dataset():
    """Check the number of clips in each reciter folder."""

    print("\n========== DATASET OVERVIEW ==========\n")

    reciter_folders = [
        folder
        for folder in os.listdir(DATASET_DIR)
        if os.path.isdir(os.path.join(DATASET_DIR, folder))
    ]

    total_clips = 0

    for folder in sorted(reciter_folders):
        folder_path = os.path.join(DATASET_DIR, folder)

        audio_files = get_audio_files(folder_path)

        number_of_clips = len(audio_files)
        total_clips += number_of_clips

        print(f"{folder:<40} : {number_of_clips}")

    print("\n--------------------------------------")
    print(f"Number of reciters : {len(reciter_folders)}")
    print(f"Total clips        : {total_clips}")


def inspect_audio():
    """Inspect the first few audio files."""

    print("\n========== AUDIO CHECK ==========\n")

    reciter_folders = sorted(os.listdir(DATASET_DIR))

    checked = 0

    for folder in reciter_folders:

        folder_path = os.path.join(DATASET_DIR, folder)

        if not os.path.isdir(folder_path):
            continue

        audio_files = get_audio_files(folder_path)

        for audio_file in audio_files[:2]:

            try:
                y, sr = librosa.load(
                    audio_file,
                    sr=None,
                    mono=True
                )

                duration = len(y) / sr

                print(f"File     : {audio_file}")
                print(f"Sample rate : {sr} Hz")
                print(f"Duration    : {duration:.2f} sec")
                print(f"Samples     : {len(y)}")
                print()

                checked += 1

            except Exception as e:

                print(f"ERROR: {audio_file}")
                print(f"       {e}")
                print()

        if checked >= 10:
            break


def find_bad_files():
    """Find files that do not match the expected audio properties."""

    print("\n========== BAD FILE CHECK ==========\n")

    bad_files = []

    for folder in os.listdir(DATASET_DIR):

        folder_path = os.path.join(DATASET_DIR, folder)

        if not os.path.isdir(folder_path):
            continue

        for audio_file in get_audio_files(folder_path):

            try:

                y, sr = librosa.load(
                    audio_file,
                    sr=None,
                    mono=True
                )

                duration = len(y) / sr

                if sr != EXPECTED_SR:
                    bad_files.append(
                        (audio_file, f"wrong sample rate: {sr}")
                    )

                if abs(duration - EXPECTED_DURATION) > 0.1:
                    bad_files.append(
                        (audio_file, f"wrong duration: {duration:.2f}s")
                    )

                if len(y) == 0:
                    bad_files.append(
                        (audio_file, "empty audio")
                    )

            except Exception as e:

                bad_files.append(
                    (audio_file, f"cannot read: {e}")
                )

    if not bad_files:

        print("No problematic files found.")

    else:

        print(f"{len(bad_files)} problematic files found:\n")

        for file, reason in bad_files[:30]:

            print(file)
            print(f"  -> {reason}")


def plot_mel_spectrogram():

    """Display Mel-spectrograms from a few reciters."""

    print("\n========== MEL-SPECTROGRAM ==========\n")

    reciter_folders = sorted(os.listdir(DATASET_DIR))

    plotted = 0

    for folder in reciter_folders:

        folder_path = os.path.join(DATASET_DIR, folder)

        if not os.path.isdir(folder_path):
            continue

        audio_files = get_audio_files(folder_path)

        if not audio_files:
            continue

        audio_file = audio_files[0]

        y, sr = librosa.load(
            audio_file,
            sr=EXPECTED_SR,
            mono=True
        )

        mel = librosa.feature.melspectrogram(
            y=y,
            sr=sr,
            n_mels=128
        )

        mel_db = librosa.power_to_db(
            mel,
            ref=np.max
        )

        plt.figure(figsize=(10, 4))

        librosa.display.specshow(
            mel_db,
            sr=sr,
            x_axis="time",
            y_axis="mel"
        )

        plt.colorbar(format="%+2.0f dB")

        plt.title(f"Mel-Spectrogram - {folder}")

        plt.tight_layout()

        plt.show()

        plotted += 1

        if plotted >= 3:
            break


def main():

    if not os.path.exists(DATASET_DIR):

        print(
            f"Dataset folder '{DATASET_DIR}' does not exist."
        )

        return

    check_dataset()

    inspect_audio()

    find_bad_files()

    plot_mel_spectrogram()


if __name__ == "__main__":
    main()