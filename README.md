# MFCC-based Japanese Vowel Recognition System

日本語5母音（あ、い、う、え、お）のリアルタイム認識システム。MFCC（メル周波数ケプストラム係数）とフォルマント分析を使用した音声認識プログラムです。

## 概要

このプロジェクトは複数のPythonプログラムで構成されています：

1. **recording.py** - 母音サンプルの録音ツール
2. **MFCC34simple.py** - 簡略版の母音認識システム（MFCC3,4使用）
3. **vowel_recognition_mfcc13.py** - 13次元MFCC使用の母音認識システム

## システムの流れ

```
recording.py → 学習データ作成(録音)
    ↓
recordings_formant/ ディレクトリに保存
    ↓
MFCC34.py, または vowel_recognition_mfcc13.py で認識
```

##　プロジェクトのクローン

```
git clone https://github.com/mizuki1024/mfcc-vowel-recognition.git
cd mfcc-vowel-recognition
```
