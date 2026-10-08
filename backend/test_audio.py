
import os
from urllib.request import urlretrieve

URL = "https://cdn.mp3quran.net/audio/abdulbasit-abdulsamad/r3/018.mp3"

DOWNLOAD_DIR = os.path.join(
    os.path.expanduser("~"),
    "Downloads"
)

OUTPUT_FILE = os.path.join(
    DOWNLOAD_DIR,
    "abdulbasit_test.mp3"
)

print("Downloading the audio...")

urlretrieve(URL, OUTPUT_FILE)

print("Download completed!")
print(f"File: {OUTPUT_FILE}")
