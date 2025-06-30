import os
import numpy as np
import sounddevice as sd
import soundfile as sf

# === 録音設定 ===
RATE = 16000
DURATION = 1.0
RECORD_DIR = "recordings_formant"

def record_vowels():
    vowels = ['a', 'i', 'u', 'e', 'o']
    os.makedirs(RECORD_DIR, exist_ok=True)
    for vowel in vowels:
        for i in range(10):
            print(f"\n→「{vowel}」を発音してください（{i+1}/10）")
            sd.sleep(1000)
            audio = sd.rec(int(RATE * DURATION), samplerate=RATE, channels=1)
            sd.wait()
            sf.write(f"{RECORD_DIR}/{vowel}_{i+1}.wav", audio, RATE)
            print("✅ 録音完了")

if __name__ == "__main__":
    record_vowels()
