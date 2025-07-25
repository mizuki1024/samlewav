import whisper
import librosa
import numpy as np
import matplotlib.pyplot as plt
import os
import sounddevice as sd
from scipy.io.wavfile import write

def record_audio(filename, duration=5, fs=16000):
    print(f"{duration}秒間録音します。話してください...")
    recording = sd.rec(int(duration * fs), samplerate=fs, channels=1, dtype='int16')
    sd.wait()
    write(filename, fs, recording)
    print(f"録音完了: {filename}")

def analyze_loudness(filename):
    y, sr = librosa.load(filename, sr=None)
    frame_length = 2048
    hop_length = 512
    rms = librosa.feature.rms(y=y, frame_length=frame_length, hop_length=hop_length)[0]
    t = librosa.frames_to_time(range(len(rms)), sr=sr, hop_length=hop_length)
    return rms, t, len(y)/sr, y, sr

def get_word_segments_whisper(filename, language='ja', model_size='small'):
    model = whisper.load_model(model_size)
    result = model.transcribe(filename, language=language, word_timestamps=True, verbose=False)
    words = []
    for seg in result['segments']:
        for w in seg['words']:
            words.append({
                'word': w['word'],
                'start': w['start'],
                'end': w['end']
            })
    return words, result['text']

def get_gradient_color(norm_value, cmap_name='coolwarm'):
    cmap = plt.get_cmap(cmap_name)
    rgba = cmap(norm_value)
    r, g, b = [int(255*x) for x in rgba[:3]]
    return f'#{r:02x}{g:02x}{b:02x}'

def colorize_words_by_loudness(words, rms, t, y, sr, cmap_name='coolwarm'):
    rms_min, rms_max = np.min(rms), np.max(rms)
    colored_words = []
    for w in words:
        start_sample = int(w['start'] * sr)
        end_sample = int(w['end'] * sr)
        if end_sample > len(y):
            end_sample = len(y)
        if end_sample > start_sample:
            segment = y[start_sample:end_sample]
            seg_rms = np.sqrt(np.mean(segment**2))
        else:
            seg_rms = rms_min
        norm = (seg_rms - rms_min) / (rms_max - rms_min + 1e-6)
        color = get_gradient_color(norm, cmap_name)
        colored_words.append({
            'word': w['word'],
            'start': w['start'],
            'end': w['end'],
            'color': color
        })
    return colored_words

def save_html_with_audio(words, audio_file, lang, filename="result.html"):
    html = f"""
    <html>
    <head>
    <meta charset=\"utf-8\">
    <title>強弱付き認識結果（グラデーション, Whisper）</title>
    </head>
    <body>
    <h2>強弱付き認識結果（グラデーション, Whisper, {lang}）</h2>
    <div style=\"width:300px; height:30px; background: linear-gradient(to right, #3b4cc0, #e0ecf4, #f7b89c, #b40426); margin-bottom:5px;\"></div>
    <div style=\"display:flex; justify-content:space-between; width:300px; margin-bottom:20px;\">
      <span>弱い</span>
      <span>普通</span>
      <span>強い</span>
    </div>
    <audio id=\"audio\" src=\"{audio_file}\" preload=\"auto\"></audio>
    <p style=\"font-size:2em;\">
    """
    for w in words:
        color = w['color']
        word = w['word']
        start = w['start']
        end = w['end']
        html += f'<span style="color:{color};cursor:pointer;" data-start="{start}" data-end="{end}" onclick="playPart(this)">{word}</span>'
    html += "</p>"
    html += """
    <script>
    function playPart(elem) {
        var audio = document.getElementById('audio');
        var start = parseFloat(elem.getAttribute('data-start'));
        var end = parseFloat(elem.getAttribute('data-end'));
        audio.currentTime = start;
        audio.play();
        var handler = function() {
            if (audio.currentTime >= end) {
                audio.pause();
                audio.removeEventListener('timeupdate', handler);
            }
        };
        audio.addEventListener('timeupdate', handler);
    }
    </script>
    </body>
    </html>
    """
    with open(filename, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"結果を {filename} に保存しました。")
    import webbrowser
    webbrowser.open('file://' + os.path.realpath(filename))

if __name__ == "__main__":
    print("1: 録音して解析")
    print("2: ファイルを指定して解析")
    choice = input("選択してください（1 or 2）: ")

    if choice == "1":
        filename = "user_input.wav"
        duration = input("録音時間（秒, デフォルト5）: ")
        duration = float(duration) if duration else 5
        record_audio(filename, duration=duration)
    elif choice == "2":
        filename = input("解析したい音声ファイルのパスを入力してください: ")
        if not os.path.exists(filename):
            print("ファイルが見つかりません。")
            exit()
    else:
        print("無効な選択です。")
        exit()

    print("言語を選択してください (ja: 日本語, en: 英語)")
    lang = input("言語 (ja/en, デフォルトja): ")
    if lang.strip() == 'en':
        lang = 'en'
    else:
        lang = 'ja'

    words, text = get_word_segments_whisper(filename, language=lang)
    rms, t, audio_duration, y, sr = analyze_loudness(filename)
    colored_words = colorize_words_by_loudness(words, rms, t, y, sr, cmap_name='coolwarm')
    save_html_with_audio(colored_words, filename, lang) 