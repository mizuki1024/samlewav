# MFCC-based Japanese Vowel Recognition System

日本語5母音（あ、い、う、え、お）のリアルタイム認識システム。MFCC（メル周波数ケプストラム係数）とフォルマント分析を使用した音声認識プログラムです。

## 概要

このプロジェクトは複数のPythonプログラムで構成されています：

1. **recording.py** - 母音サンプルの録音ツール
2. **MFCC34.py** - フル機能版の母音認識システム（MFCC3,4使用）
3. **MFCC34simple.py** - 簡略版の母音認識システム（MFCC3,4使用）
4. **vowel_recognition_mfcc13.py** - 13次元MFCC使用の高精度認識システム

## システムの流れ

```
recording.py → 学習データ作成
    ↓
recordings_formant/ ディレクトリに保存
    ↓
MFCC34.py, MFCC34simple.py, または vowel_recognition_mfcc13.py で認識
```

## 各ファイルの詳細

### 1. recording.py - 母音サンプル録音ツール

#### 概要
学習データとなる母音サンプルを録音するためのツールです。

#### 機能
- 各母音（あ、い、う、え、お）を30サンプルずつ録音
- 1サンプルあたり1秒間の録音
- カウントダウン表示で録音タイミングを案内
- `recordings_formant/`ディレクトリに自動保存

#### 使用方法
```bash
python recording.py
```

#### 出力ファイル形式
- `recordings_formant/a_1.wav` から `a_30.wav`
- `recordings_formant/i_1.wav` から `i_30.wav`
- 以下同様に各母音30ファイル

### 2. MFCC34.py - フル機能版母音認識システム

#### 概要
MFCC3,4係数とフォルマント分析を組み合わせた高精度な母音認識システムです。

#### 主要機能
- **リアルタイム母音認識**: 1秒ごとに音声を録音し、即座に認識
- **MFCC特徴抽出**: MFCC3,4（インデックス2,3）を使用
- **フォルマント分析**: LPC分析によるF1,F2,F3の抽出
- **i/e判別強化**: 第2フォルマント（F2）を重視した判別アルゴリズム

#### 可視化機能（3つの図）
1. **MFCC空間図**
   - 2次元空間での母音分布を表示
   - リアルタイムで認識結果をプロット
   - 曖昧母音（ə）の理論位置も表示

2. **フォルマントチャート**
   - F1-F2平面での母音分布
   - 音響音声学の慣例に従った軸設定

3. **伝統的母音図**
   - 舌の位置による母音分類
   - 前舌-後舌、高-低の2次元表示
   - リアルタイムで推定位置を表示

#### パラメータ設定
```python
IE_DISTINGUISH = True    # i/e判別機能の有効化
IE_BIAS = 0.8           # iをeより優先する度合い
VERY_SIMILAR_THRESHOLD = 0.85  # 高類似度の閾値
```

#### 使用方法
```bash
python MFCC34.py
```

### 3. MFCC34simple.py - 簡略版母音認識システム

#### 概要
MFCC34.pyの軽量版。コア機能は同じですが、表示を最小限にして処理速度を向上させています。

#### 主要機能
- MFCC34.pyと同じ認識アルゴリズム
- MFCC3,4係数とフォルマント分析を使用
- 個別サンプルとの比較を無効化して高速化

#### 可視化機能（1つの図のみ）
- **MFCC空間図**のみを表示
- フォルマントチャートと母音図は削除

#### 使用シーン
- シンプルな認識結果だけが必要な場合
- 処理速度を重視する場合
- 画面スペースが限られている場合

#### 使用方法
```bash
python MFCC34simple.py
```

### 4. vowel_recognition_mfcc13.py - 13次元MFCC高精度認識システム

#### 概要
全13次元のMFCC特徴量を使用した高精度な母音認識システムです。より多くの音響特徴を捉えることで、認識精度を向上させています。

#### 主要機能
- **13次元MFCC使用**: MFCC0〜MFCC12のすべての係数を活用
- **リアルタイム認識**: 1秒ごとに音声を録音し、即座に認識
- **MFCC重要度分析**: 各係数の母音識別への貢献度を自動分析
- **テンプレートマッチング**: ユークリッド距離による分類

#### 可視化機能（2つの図）
1. **MFCCプロファイル**
   - 13次元すべてのMFCC値をリアルタイム表示
   - 各母音のテンプレートとの比較
   
2. **距離バーチャート**
   - 各母音との距離を可視化
   - 認識結果をハイライト表示

#### 分析機能
- 母音ごとのMFCCプロファイル比較
- 各MFCC係数の識別力（分散）分析
- 最も重要なMFCC係数の自動特定
- ヒートマップによる可視化

#### 使用シーン
- 最高精度の認識が必要な場合
- 音響特徴の詳細な分析が必要な場合
- 研究・教育目的での使用

#### 使用方法
```bash
python vowel_recognition_mfcc13.py
```

## 必要な環境

### Pythonバージョン
- Python 3.7以上

### 必要なライブラリ
```bash
pip install numpy
pip install sounddevice
pip install soundfile
pip install librosa
pip install matplotlib
pip install scikit-learn
pip install scipy
```

### 動作環境
- macOS, Windows, Linux
- マイク入力が必要

## クイックスタート（初めての方向け）

### 1. Gitのインストール確認
```bash
# Gitがインストールされているか確認
git --version

# インストールされていない場合
# Mac: brew install git
# Windows: https://git-scm.com/download/win
# Linux: sudo apt-get install git
```

### 2. リポジトリの取得
```bash
# ホームディレクトリに移動
cd ~

# リポジトリをクローン（ダウンロード）
git clone https://github.com/mizuki1024/mfcc-vowel-recognition.git

# プロジェクトディレクトリに移動
cd mfcc-vowel-recognition

# ファイル一覧を確認
ls -la
```

### 3. Python環境の準備
```bash
# Pythonのバージョン確認（3.7以上が必要）
python3 --version

# 仮想環境の作成（推奨）
python3 -m venv venv

# 仮想環境の有効化
# Mac/Linux:
source venv/bin/activate
# Windows:
# venv\Scripts\activate

# 必要なライブラリをインストール
pip install -r requirements.txt
```

### 4. 学習データの作成
```bash
# 録音プログラムを実行
python3 recording.py

# 画面の指示に従って各母音を30回ずつ録音
# 合計150個の音声ファイルが作成される
```

### 5. 母音認識を実行
```bash
# 方法1: フル機能版（3つのグラフ表示）
python3 MFCC34.py

# 方法2: 簡易版（1つのグラフ表示）
python3 MFCC34simple.py

# 方法3: 高精度版（13次元MFCC使用）
python3 vowel_recognition_mfcc13.py
```

## 詳細なセットアップ手順

### 前提条件
- Git
- Python 3.7以上
- マイク（内蔵または外付け）
- 約100MBの空き容量

### ステップ1: リポジトリのクローン
```bash
# HTTPSでクローン（推奨）
git clone https://github.com/mizuki1024/mfcc-vowel-recognition.git

# またはSSHでクローン（GitHubにSSHキー登録済みの場合）
git clone git@github.com:mizuki1024/mfcc-vowel-recognition.git

# クローンしたディレクトリに移動
cd mfcc-vowel-recognition
```

### ステップ2: Python環境のセットアップ
```bash
# 仮想環境を作成（プロジェクト専用の環境）
python3 -m venv venv

# 仮想環境を有効化
source venv/bin/activate  # Mac/Linux
# venv\Scripts\activate   # Windows

# pipを最新版に更新
pip install --upgrade pip

# 依存ライブラリをインストール
pip install -r requirements.txt

# インストール確認
pip list
```

### ステップ3: 初回セットアップ
```bash
# 録音ディレクトリの確認
ls -la recordings_formant/

# ディレクトリがない場合は作成される
python3 recording.py
```

## 各ファイルの詳細な動作方法

### recording.py - 学習データの録音

#### 起動方法
```bash
# 仮想環境が有効化されていることを確認
# (venv) が表示されているはず

# プログラムを実行
python3 recording.py
```

#### 動作の流れ
1. **起動時の表示**
   ```
   🎤 母音録音ツール
   各母音を30回ずつ録音します
   母音 'a' の録音を開始します
   ```

2. **録音プロセス**
   ```
   サンプル 1/30
   準備してください...
   3
   2
   1
   🔴 録音中...
   ✅ 録音完了!
   ```

3. **録音のコツ**
   - カウントダウン中に息を整える
   - 「録音中...」が表示されたらすぐに発音
   - 1秒間同じ母音を継続
   - 各母音30回×5母音＝計150回録音

4. **録音ファイルの確認**
   ```bash
   # 録音されたファイルを確認
   ls recordings_formant/
   # a_1.wav, a_2.wav, ..., o_30.wav が表示される
   
   # 録音数を確認
   ls recordings_formant/*.wav | wc -l
   # 150 と表示されれば成功
   ```

### MFCC34.py - フル機能版母音認識

#### 起動方法
```bash
# プログラムを実行
python3 MFCC34.py
```

#### 初回起動時の動作
```
📦 テンプレート読み込み中...
🔍 曖昧母音の理論フォルマント値: F1=642Hz, F2=1502Hz
📊 MFCC成分強度分析を 'mfcc_strength_analysis.png' に保存しました
🔊 母音図を生成中...
🟢 リアルタイム母音認識を開始します（Ctrl+Cで停止）
```

#### 画面表示
1. **3つのウィンドウが開く**
   - 左: MFCC3-4空間図（リアルタイム更新）
   - 中央: フォルマントチャート（静的）
   - 右: 伝統的母音図（リアルタイム更新）

2. **認識サイクル**
   ```
   🎤 発音してください...
   [1秒間録音]
   
   === 判定結果 ===
   🗣 推定: 「a」 / 距離スコア: 8.45
   📊 あなたのフォルマント: F1=720Hz, F2=1250Hz, F3=2800Hz
   📊 類似度ランキング:
     1. a（距離: 8.45）
     2. o（距離: 15.23）
     3. e（距離: 18.67）
   ✅ 発音は良好です！
   ```

3. **終了方法**
   - Ctrl+C を押す
   - 「🛑 終了しました。」と表示される

### vowel_recognition_mfcc13.py - 高精度版（13次元MFCC）

#### 起動方法
```bash
# プログラムを実行
python3 vowel_recognition_mfcc13.py
```

#### 起動時の分析表示
```
🎵 13次元MFCCを使用した母音認識システム
📦 テンプレート読み込み中...

📊 13次元MFCCによる母音識別分析:
  使用しているMFCC係数: MFCC0〜MFCC12 (全13次元)
  サンプル数: 15

📊 母音識別に重要なMFCC係数（分散が大きい順）:
  1. MFCC1: 分散=245.632
  2. MFCC2: 分散=198.456
  3. MFCC0: 分散=156.789
  4. MFCC3: 分散=98.234
  5. MFCC4: 分散=87.123

📊 13次元MFCC分析結果を 'mfcc13_analysis.png' に保存しました

📊 分析が完了しました。Enterキーを押してリアルタイム認識を開始...
```

#### リアルタイム認識モード
1. **2つのグラフウィンドウ**
   - 左: 13次元MFCCプロファイル（リアルタイム）
   - 右: 各母音との距離（バーチャート）

2. **認識結果の表示**
   ```
   🎤 発音してください...
   
   === 判定結果 ===
   🗣 推定: 「i」 / 距離スコア: 4.23
   📊 13次元MFCC使用
   📊 類似度ランキング:
     1. i（距離: 4.23）
     2. e（距離: 8.56）
     3. a（距離: 12.34）
   ⭐ 良い発音です！
   ```

### MFCC34simple.py - 簡易版

#### 起動方法
```bash
python3 MFCC34simple.py
```

#### 特徴
- MFCC34.pyと同じ認識エンジン
- 表示はMFCC空間図のみ（1ウィンドウ）
- 起動が速い
- 画面が小さい環境に最適

### 認識精度を上げるコツ

1. **録音環境**
   - 静かな環境で録音する
   - マイクから適切な距離（15-30cm）を保つ
   - 一定の音量で発音する

2. **発音のポイント**
   - はっきりと母音を発音する
   - 1秒間同じ母音を継続する
   - 各母音の特徴を意識する：
     - 「あ」：口を大きく開ける
     - 「い」：口を横に引く
     - 「う」：唇をすぼめる
     - 「え」：口角を上げる
     - 「お」：唇を丸くする

3. **学習データの質**
   - recording.pyで多様な発音を録音する
   - 異なる声の高さで録音する
   - 安定した発音を心がける

### トラブルシューティング

#### 「テンプレートがありません」エラー
```bash
# エラーメッセージ
❌ テンプレートがありません。recordings_formant フォルダを確認してください。

# 解決方法
# 1. ディレクトリの確認
ls -la recordings_formant/

# 2. ディレクトリがない、または空の場合
python3 recording.py
# 各母音を30回ずつ録音

# 3. 録音ファイルの確認
ls recordings_formant/*.wav | wc -l
# 150と表示されればOK
```

#### マイクが認識されない
```bash
# macOSの場合
# システム設定 > プライバシーとセキュリティ > マイク
# Terminalまたは使用中のアプリにマイクアクセスを許可

# 音声デバイスの確認（Python内で）
python3
>>> import sounddevice as sd
>>> sd.query_devices()
# 利用可能なデバイスリストが表示される
>>> exit()
```

#### 認識精度が低い場合
1. **環境の改善**
   ```bash
   # 録音時のノイズレベルを確認
   # recording.pyを実行して背景ノイズを測定
   ```

2. **学習データの質を向上**
   ```bash
   # 既存データをバックアップ
   mv recordings_formant recordings_formant_backup
   
   # 新しく録音
   python3 recording.py
   # より静かな環境で、はっきりと発音
   ```

3. **プログラムの選択**
   ```bash
   # より高精度な認識が必要な場合
   python3 vowel_recognition_mfcc13.py
   # 13次元すべてを使用するため精度が向上
   ```

#### Pythonライブラリのエラー
```bash
# NumPyのバージョンエラーの場合
pip uninstall numpy
pip install numpy==1.21.0

# librosaのエラーの場合
pip uninstall librosa
pip install librosa==0.9.2

# すべての依存関係を再インストール
pip install -r requirements.txt --force-reinstall
```

## 技術的な詳細

### MFCC（メル周波数ケプストラム係数）
- 人間の聴覚特性を考慮した音声特徴量
- MFCC34.py/MFCC34simple.py: MFCC3,4を使用（中域周波数の情報）
- vowel_recognition_mfcc13.py: 13次元すべてを使用（全周波数帯域の情報）
- フォルマント周波数と高い相関

### フォルマント分析
- LPC（線形予測符号化）による分析
- F1: 第1フォルマント（舌の高さに対応）
- F2: 第2フォルマント（舌の前後位置に対応）
- F3: 第3フォルマント（補助的情報）

### 認識アルゴリズム
1. 音声録音（1秒間）
2. 無音区間除去
3. MFCC特徴抽出
4. フォルマント抽出
5. テンプレートマッチング
6. 距離計算と分類

## プロジェクトの特徴

- **リアルタイム処理**: 1秒ごとに認識結果を更新
- **高精度なi/e判別**: フォルマント情報を活用
- **視覚的フィードバック**: 複数の可視化手法
- **発音アドバイス機能**: 各母音の発音方法を表示
- **拡張性**: 新しい母音や言語への対応が可能

## ライセンス

MIT License

## 作者

Mizuki Murata

## 謝辞

このプロジェクトは音声認識と音響音声学の研究成果を基に開発されました。