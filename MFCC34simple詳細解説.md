# MFCC34simple.py 完全解説

## 概要
MFCC34simple.pyは、MFCC34.pyの簡略版で、**MFCC4,5（第5,6係数）のみを使用**した日本語5母音のリアルタイム認識システムです。母音図（舌位置）の可視化機能を除去し、MFCC空間での可視化に特化しています。

## 主な変更点（MFCC34.pyとの違い）

### 1. 使用するMFCC係数の変更
```python
# MFCC34.py: MFCC3,4を使用
X_mfcc.append(mfcc_mean[3:5])  # インデックス3,4

# MFCC34simple.py: MFCC4,5を使用
X_mfcc.append(mfcc_mean[3:5])  # インデックス3,4（実際は4,5番目）
```

**注意**: コードのコメントでは「MFCC4,5」と記載されていますが、実装は同じインデックス[3:5]を使用しています。

### 2. 削除された機能
- 伝統的な母音図（舌位置マッピング）関連のコード
- フォルマントチャートの作成機能
- 母音図へのリアルタイムプロット機能

### 3. 簡略化された構造
- 不要な定数定義（VOWEL_POSITIONS、IPA_SYMBOLS）を削除
- 母音図関連の関数を削除
- メイン処理ループの簡素化

## 技術的詳細

### 1. MFCC4,5の特性
```python
# MFCC4（第5係数）: 約1500-2500Hz領域のスペクトル構造
# MFCC5（第6係数）: 約2000-3500Hz領域のスペクトル構造
```

これらの係数は主に**第2フォルマント（F2）の情報**を強く反映し、特に前舌母音（i, e）と後舌母音（u, o）の区別に有効です。

### 2. フォルマント抽出（57-162行目）
```python
def extract_formants(y, sr, n_formants=3):
    """LPC分析によるフォルマント抽出"""
```

#### 処理フロー：
1. **プリエンファシス**: 高周波成分を強調（係数0.97）
2. **フレーム分割**: 25msフレーム、10msシフト
3. **窓関数適用**: ハミング窓でスペクトル漏れを防ぐ
4. **LPC分析**: 
   - 自己相関計算
   - Levinson-Durbin法でLPC係数を求める
   - LPC次数 = sr/1000 + 4（16kHzなら20次）
5. **スペクトル包絡計算**: freqz関数で周波数応答
6. **ピーク検出**: 200-5000Hz範囲でフォルマント候補を探索
7. **中央値計算**: 全フレームの結果から安定した値を抽出

### 3. 特徴抽出（166-199行目）
```python
def extract_features():
    """各母音サンプルからMFCCとフォルマントを抽出"""
```

各母音について3つのサンプルファイルを処理：
- ファイル形式: `{母音}_{番号}.wav`（例: a_1.wav）
- MFCC: 13次元計算後、4,5番目のみ使用
- フォルマント: F1, F2, F3を抽出
- 個別サンプルのフォルマントも保存（後の類似度計算用）

### 4. 分類アルゴリズム（257-405行目）
```python
def classify(user_mfcc, user_formants, templates, formant_templates, all_formants=None):
    """ユーザー入力を母音に分類"""
```

#### 4.1 距離計算の詳細

**MFCC距離**（259行目）:
```python
mfcc_distances = {vowel: np.linalg.norm(user_mfcc - vec) for vowel, vec in templates.items()}
```
- ユークリッド距離で計算
- 2次元ベクトル間の直線距離

**フォルマント距離**（268-288行目）:
```python
# 正規化された距離計算
f1_dist = abs(user_formants[0] - template_formants[0]) / max_formant
f2_dist = abs(user_formants[1] - template_formants[1]) / max_formant

# 母音別の重み付け
if vowel in ['i', 'e']:
    formant_dist = 0.3 * f1_dist + 0.7 * f2_dist  # F2重視
else:
    formant_dist = 0.5 * f1_dist + 0.5 * f2_dist  # 均等

# 類似度スコア（0-1の範囲）
similarity = 1.0 / (1.0 + 10 * formant_dist)
```

#### 4.2 個別サンプルとの比較（290-329行目）
録音済みの各サンプルと直接比較し、最も類似度の高いものを選択：

```python
# 各サンプルとの類似度を計算
for sample_formant in vowel_samples:
    # F1, F2の距離から類似度を計算
    sim = 1.0 / (1.0 + 10 * dist)
    
# 「い」の優先処理
if vowel == 'i':
    max_sim *= (1.0 + (IE_BIAS - 0.5) * 0.5)  # 類似度を上げる
elif vowel == 'e':
    max_sim *= (1.0 - (IE_BIAS - 0.5) * 0.5)  # 類似度を下げる
```

#### 4.3 合成距離の計算（331-349行目）
```python
# 「い」「え」の場合
combined_distances[vowel] = 0.6 * mfcc_distances[vowel] + 0.4 * formant_distances[vowel] * 100

# その他の母音
combined_distances[vowel] = 0.8 * mfcc_distances[vowel] + 0.2 * formant_distances[vowel] * 100
```

フォルマント距離に100を掛けてスケールを調整しています。

#### 4.4 特別な「い」「え」判定（361-402行目）
上位2つが「い」と「え」の場合、第2フォルマントで詳細判定：

```python
# F2の相対位置を計算（0=e寄り、1=i寄り）
rel_pos = (user_f2 - e_f2) / abs(i_f2 - e_f2)

# バイアスを適用
rel_pos = rel_pos * (2 * IE_BIAS) + (1 - IE_BIAS) * 2 - 1

# 判定
if rel_pos > 0.5:
    # 「い」を優先
else:
    # 「え」を優先
```

### 5. 可視化（413-453行目）
```python
def init_3d_plot(X, y, pca, templates):
    """2次元MFCC空間の可視化"""
```

特徴：
- X軸: MFCC5（第6係数）
- Y軸: MFCC4（第5係数）
- 各母音をカラーコードで表示
- 曖昧母音（ə）も理論値として表示
- 軸を入れ替えて表示（視覚的な配置のため）

### 6. リアルタイム処理（514-603行目）

#### 6.1 初期化フェーズ
1. テンプレート読み込み
2. 曖昧母音の理論値計算（全母音の平均）
3. フォルマント情報の表示
4. 可視化ウィンドウの初期化

#### 6.2 メインループ
```python
while True:
    # 1. 録音（1秒間）
    record_audio(audio_path)
    
    # 2. 特徴抽出
    user_mfcc, user_formants = extract_user_features(audio_path)
    
    # 3. 分類
    results = classify(...)
    
    # 4. 結果表示
    - 推定母音と距離スコア
    - フォルマント値（F1, F2, F3）
    - サンプルとの類似度
    - 類似度ランキング
    
    # 5. 可視化更新
    update_user_point(...)
```

### 7. パラメータ調整

#### 7.1 主要パラメータ
```python
IE_DISTINGUISH = True    # i/e判別機能の有効化
IE_BIAS = 0.8           # iの優先度（0.5=中立、1.0=最大）
VERY_SIMILAR_THRESHOLD = 0.85  # 高類似度の閾値
```

#### 7.2 調整のガイドライン
- **IE_BIASを上げる**: 「い」をより認識しやすく
- **IE_BIASを下げる**: 「え」をより認識しやすく
- **VERY_SIMILAR_THRESHOLDを上げる**: より厳密な判定
- **VERY_SIMILAR_THRESHOLDを下げる**: より緩い判定

### 8. 発音評価システム（475-484行目）
```python
def show_advice(vowel, score):
    """距離スコアに基づく発音評価"""
```

評価基準：
- **score < 15**: 発音良好
- **15 ≤ score < 30**: 少しズレている
- **score ≥ 30**: かなりズレている

各母音に対する具体的なアドバイスも提供。

## 実行時の動作

### 1. 起動時
```
📦 テンプレート読み込み中...
🔍 曖昧母音の理論フォルマント値: F1=XXXHz, F2=XXXHz
🔍 母音のフォルマント情報:
  「a」: F1=XXXHz, F2=XXXHz
  ...
```

### 2. 認識時
```
🎤 発音してください...
=== 判定結果 ===
🗣 推定: 「a」 / 距離スコア: XX.XX
📊 あなたのフォルマント: F1=XXXHz, F2=XXXHz, F3=XXXHz
📊 類似度ランキング:
  1. a（距離: XX.XX）
  2. o（距離: XX.XX）
  ...
```

### 3. 特別な判定
```
⭐ サンプルとの高い類似度検出: 「i」 (類似度: 0.XXX)
🔍 フォルマント分析による「i」/「e」判別:
  i のF2: XXXXHz, e のF2: XXXXHz, あなたのF2: XXXXHz
  相対位置: X.XX (0=e寄り, 1=i寄り), バイアス: 0.8
```

## まとめ

MFCC34simple.pyは、MFCC34.pyから母音図機能を除去し、コア機能に集中した実装です。主な特徴：

1. **MFCC4,5のみ使用**: 計算効率と認識精度のバランス
2. **フォルマント分析統合**: 特に「い」「え」の判別精度向上
3. **個別サンプル比較**: テンプレートマッチングより精密な判定
4. **リアルタイム処理**: 1秒ごとの連続認識
5. **視覚的フィードバック**: MFCC空間での位置表示

この簡略版は、母音認識の本質的な機能を保持しながら、コードの理解と保守を容易にしています。