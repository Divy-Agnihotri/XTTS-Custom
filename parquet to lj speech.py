import os
import io
import pandas as pd
import soundfile as sf
from pydub import AudioSegment

PARQUET_FILE = r"C:\Users\Owner\Downloads\train-00000-of-00008.parquet"
OUTPUT_DIR = r"D:\TTS\ljspeech_output"

WAV_DIR = os.path.join(OUTPUT_DIR, "wavs")
METADATA_FILE = os.path.join(OUTPUT_DIR, "metadata.txt")

os.makedirs(WAV_DIR, exist_ok=True)

df = pd.read_parquet(PARQUET_FILE)

with open(METADATA_FILE, "w", encoding="utf-8") as f:

    for i, row in df.iterrows():

        transcript = str(row["sentence"]).strip()

        audio_bytes = row["audio"]["bytes"]

        # -----------------------------
        # Decode audio bytes
        # -----------------------------
        audio = AudioSegment.from_file(io.BytesIO(audio_bytes))

        audio = audio.set_channels(1)  # mono
        audio = audio.set_frame_rate(22050)

        audio_array = audio.get_array_of_samples()

        filename = f"{i:06d}.wav"
        wav_path = os.path.join(WAV_DIR, filename)

        sf.write(wav_path, audio_array, audio.frame_rate)

        # LJSpeech format
        f.write(f"{filename}|{transcript}|{transcript}\n")

print("DONE ✔ Dataset converted to LJSpeech format")
