#MFCC45
# === ライブラリのインポート ===
import os
import numpy as np
import sounddevice as sd
import soundfile as sf
import librosa
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.decomposition import PCA
from matplotlib.patches import Ellipse
from scipy.linalg import eigh
from scipy.signal import find_peaks
from time import sleep

# === フォント設定（Mac対応） ===
plt.rcParams['font.family'] = 'AppleGothic'  # Mac用フォント（Windowsの場合は 'MS Gothic'）

# === 音声処理設定 ===
RATE = 16000  # サンプリングレート（16kHz）
DURATION = 1.0  # 録音時間（秒）
N_MFCC = 13  # MFCCの次元数
RECORDINGS_DIR = "recordings_formant"  # 録音ファイルの保存先
VOWELS = ['a', 'i', 'u', 'e', 'o']  # 対象とする母音
VOWELS_WITH_SCHWA = ['a', 'i', 'u', 'e', 'o', 'ə']  # 曖昧母音を含む母音リスト
SAMPLES_PER_VOWEL = 3  # 母音ごとのサンプル数

# iとeの判別のためのパラメータ
IE_DISTINGUISH = True  # iとeの判別機能を有効にする
IE_BIAS = 0.8  # iをeより優先する度合い (0.5 = 中立, 1.0 = 最大)
VERY_SIMILAR_THRESHOLD = 0.85  # 「すごく近い」と判断する類似度閾値

# === 母音の色マッピング ===
COLOR_MAP = {
    'a': 'red',
    'i': 'blue',
    'u': 'green',
    'e': 'purple',
    'o': 'orange',
    'ə': 'gray'  # 曖昧母音（シュワー）
}

# === 母音の舌位置マッピング（前舌-後舌, 高-低） ===
# 値は (x, y) 座標で、x: 0(前舌)→1(後舌), y: 0(高)→1(低)
VOWEL_POSITIONS = {
    'i': (0.1, 0.1),   # 前舌・高
    'e': (0.2, 0.4),   # 前舌・中高
    'a': (0.6, 0.9),   # 中舌・低
    'o': (0.8, 0.4),   # 後舌・中高
    'u': (0.9, 0.1),   # 後舌・高
    'ə': (0.5, 0.5)    # 中舌・中（曖昧母音）
}

# === 国際音声記号（IPA）の対応 ===
IPA_SYMBOLS = {
    'i': 'i',
    'e': 'e',
    'a': 'a',
    'o': 'o',
    'u': 'u',
    'ə': 'ə'
}

# === 母音ごとの発音アドバイス ===
ADVICE_MAP = {
    'a': "口を大きく縦に開け、舌は下に落としましょう。",
    'i': "口を横に引いて、舌は前に出すようにしましょう。",
    'u': "唇をすぼめて、舌を後ろに引きましょう。",
    'e': "口角を少し上げ、舌をやや前に出しましょう。",
    'o': "唇を丸く突き出し、舌を後ろに引きましょう。",
    'ə': "口と舌をリラックスさせ、力を抜いて発音しましょう。"
}

# === フォルマント抽出機能 ===
def extract_formants(y, sr, n_formants=3):
    """音声波形からフォルマント周波数を抽出する（LPC分析使用）"""
    from scipy.signal import lfilter, freqz
    from scipy.signal.windows import hamming
    
    # 音声信号の前処理
    # プリエンファシス（高周波成分の強調）
    pre_emphasis = 0.97
    emphasized = np.append(y[0], y[1:] - pre_emphasis * y[:-1])
    
    # フレーム分割のパラメータ
    frame_size = int(0.025 * sr)  # 25msのフレーム
    frame_shift = int(0.010 * sr)  # 10msのシフト
    
    formants_list = []
    
    # フレームごとに処理
    for start in range(0, len(emphasized) - frame_size, frame_shift):
        frame = emphasized[start:start + frame_size]
        
        # ハミング窓を適用
        windowed = frame * hamming(len(frame))
        
        # LPC分析
        # LPC次数は (サンプリングレート/1000) + 2 が目安
        lpc_order = int(sr / 1000) + 4
        
        try:
            # 自己相関法によるLPC係数の計算
            autocorr = np.correlate(windowed, windowed, mode='full')
            autocorr = autocorr[len(autocorr)//2:]
            
            # Levinson-Durbin再帰でLPC係数を計算
            lpc_coeffs = solve_lpc(autocorr, lpc_order)
            
            # LPCスペクトル包絡を計算
            w, h = freqz([1], lpc_coeffs, worN=8192, fs=sr)
            
            # スペクトル包絡からピークを検出
            magnitude = 20 * np.log10(np.abs(h) + 1e-15)
            
            # ピーク検出（フォルマント候補）
            peaks, properties = find_peaks(magnitude, distance=int(300/(sr/len(w))))
            
            # ピークの周波数を取得
            peak_freqs = w[peaks]
            
            # 有効なフォルマント範囲でフィルタリング（200Hz-5000Hz）
            valid_peaks = [(f, magnitude[peaks[i]]) for i, f in enumerate(peak_freqs) 
                          if 200 < f < 5000]
            
            # 強度でソートして上位n個を選択
            valid_peaks.sort(key=lambda x: x[1], reverse=True)
            
            # 周波数でソート（低い順）してフォルマントとする
            if valid_peaks:
                formant_freqs = sorted([f for f, _ in valid_peaks[:n_formants*2]])[:n_formants]
            else:
                formant_freqs = []
            
            # 必要な数に満たない場合は0で埋める
            formant_freqs = formant_freqs + [0] * (n_formants - len(formant_freqs))
            
        except Exception:
            # エラーが発生した場合はデフォルト値
            formant_freqs = [0] * n_formants
        
        formants_list.append(formant_freqs)
    
    # 全フレームのフォルマントの中央値を計算
    formants_array = np.array(formants_list)
    # 0以外の値のみで中央値を計算
    median_formants = []
    for i in range(n_formants):
        valid_values = formants_array[:, i][formants_array[:, i] > 0]
        if len(valid_values) > 0:
            median_formants.append(np.median(valid_values))
        else:
            median_formants.append(0)
    
    return np.array(median_formants)

def solve_lpc(autocorr, order):
    """Levinson-Durbin再帰によるLPC係数の計算"""
    # 初期化
    error = autocorr[0]
    lpc = np.zeros(order + 1)
    lpc[0] = 1.0
    
    for i in range(1, order + 1):
        # 反射係数の計算
        k = -np.sum(lpc[:i] * autocorr[i:0:-1]) / error
        
        # LPC係数の更新
        lpc_temp = lpc.copy()
        lpc[i] = k
        for j in range(1, i):
            lpc[j] = lpc_temp[j] + k * lpc_temp[i - j]
        
        # 予測誤差の更新
        error *= (1 - k * k)
        
        if error <= 0:
            break
    
    return lpc

# === フォルマント値を舌位置に変換する関数 ===
def formants_to_tongue_position(f1, f2):
    """
    フォルマント周波数から舌の位置を推定
    
    Parameters:
    - f1: 第1フォルマント (Hz)
    - f2: 第2フォルマント (Hz)
    
    Returns:
    - (x, y): 母音図上の座標 (x: 0→1=前→後, y: 0→1=高→低)
    """
    # F1は主に口の開き（高さ）に対応: 低F1=高舌位、高F1=低舌位
    # 典型的なF1の範囲: 高母音=300Hz, 低母音=800Hz
    y = min(1.0, max(0.0, (f1 - 200) / 800))
    
    # F2は主に舌の前後位置に対応: 高F2=前舌、低F2=後舌
    # 典型的なF2の範囲: 前舌=2300Hz, 後舌=800Hz
    x = min(1.0, max(0.0, 1.0 - (f2 - 700) / 1800))
    
    return (x, y)

# === 音声ファイルからMFCC特徴量とフォルマントを抽出 ===
def extract_features():
    X_mfcc, X_formants, y = [], [], []
    all_formants = {}  # 母音ごとの個別サンプルのフォルマントを保存
    
    for vowel in VOWELS:
        all_formants[vowel] = []  # この母音のサンプルフォルマントリスト
        
        for i in range(1, SAMPLES_PER_VOWEL + 1):
            filepath = os.path.join(RECORDINGS_DIR, f"{vowel}_{i}.wav")
            if os.path.exists(filepath):
                # 音声読み込み
                y_data, sr = librosa.load(filepath, sr=RATE)
                
                # 無音区間除去
                y_data, _ = librosa.effects.trim(y_data, top_db=20)
                
                # MFCC抽出
                mfcc = librosa.feature.mfcc(y=y_data, sr=sr, n_mfcc=N_MFCC)
                mfcc_mean = np.mean(mfcc, axis=1)
                # MFCC3とMFCC4（インデックス3と4）のみを使用
                X_mfcc.append(mfcc_mean[3:5])  # 4番目と5番目の係数のみ
                
                # フォルマント抽出
                formants = extract_formants(y_data, sr)
                X_formants.append(formants)
                
                # 個別サンプルのフォルマントを保存
                all_formants[vowel].append(formants)
                
                y.append(vowel)
            else:
                print(f"⚠️ ファイルが見つかりません: {filepath}")
    
    return np.array(X_mfcc), np.array(X_formants), np.array(y), all_formants

# === 母音ごとにMFCCとフォルマントの平均テンプレートを作成 ===
def build_templates(X_mfcc, X_formants, y):
    templates = {}
    formant_templates = {}
    
    for vowel in VOWELS:
        indices = y == vowel
        if np.any(indices):
            templates[vowel] = np.mean(X_mfcc[indices], axis=0)
            formant_templates[vowel] = np.mean(X_formants[indices], axis=0)
    
    # 曖昧母音（シュワー）の理論値を計算
    if len(formant_templates) == len(VOWELS):
        # 5母音のフォルマント値から中心値を計算
        all_formants = np.array(list(formant_templates.values()))
        schwa_formants = np.mean(all_formants, axis=0)
        
        # 曖昧母音のMFCCは全母音の平均
        all_mfcc = np.array(list(templates.values()))
        schwa_mfcc = np.mean(all_mfcc, axis=0)
        
        # テンプレートに追加
        templates['ə'] = schwa_mfcc
        formant_templates['ə'] = schwa_formants
        
        print(f"🔍 曖昧母音の理論フォルマント値: F1={schwa_formants[0]:.0f}Hz, F2={schwa_formants[1]:.0f}Hz")
    
    return templates, formant_templates

# === ユーザーの発音を録音し、ファイルに保存 ===
def record_audio(path):
    print("🎤 発音してください...")
    audio = sd.rec(int(RATE * DURATION), samplerate=RATE, channels=1)
    sd.wait()
    sf.write(path, audio, RATE)

# === 録音された音声から特徴を抽出 ===
def extract_user_features(filepath):
    # 音声読み込み
    y_data, sr = librosa.load(filepath, sr=RATE)
    
    # 無音区間除去
    y_data, _ = librosa.effects.trim(y_data, top_db=20)
    
    # MFCC抽出
    mfcc = librosa.feature.mfcc(y=y_data, sr=sr, n_mfcc=N_MFCC)
    mfcc_mean = np.mean(mfcc, axis=1)
    # MFCC3とMFCC4（インデックス3と4）のみを使用
    mfcc_features = mfcc_mean[3:5]  # 4番目と5番目の係数のみ
    
    # フォルマント抽出
    formant_features = extract_formants(y_data, sr)
    
    return mfcc_features, formant_features

# === ユーザーのMFCCとフォルマントをテンプレートと比較して分類 ===
def classify(user_mfcc, user_formants, templates, formant_templates, all_formants=None):
    # MFCC距離の計算
    mfcc_distances = {vowel: np.linalg.norm(user_mfcc - vec) for vowel, vec in templates.items()}
    
    # フォルマント距離の計算（正規化）
    max_formant = 4000  # 正規化のための最大フォルマント周波数
    formant_distances = {}
    
    # フォルマントの類似度スコア（1に近いほど類似）
    formant_similarity = {}
    
    for vowel, template_formants in formant_templates.items():
        # 第1と第2フォルマントの距離（正規化）
        if len(template_formants) >= 2 and len(user_formants) >= 2:
            # 各フォルマントの距離を計算（周波数で正規化）
            f1_dist = abs(user_formants[0] - template_formants[0]) / max_formant
            f2_dist = abs(user_formants[1] - template_formants[1]) / max_formant
            
            # iとeの場合は第2フォルマントを重視
            if vowel in ['i', 'e']:
                formant_dist = 0.3 * f1_dist + 0.7 * f2_dist
            else:
                formant_dist = 0.5 * f1_dist + 0.5 * f2_dist
                
            # 類似度スコア（距離の逆数、近いほど高い）
            similarity = 1.0 / (1.0 + 10 * formant_dist)  # 0-1の範囲に正規化
        else:
            formant_dist = 1.0  # フォルマントが取れない場合は最大距離
            similarity = 0.0
        
        formant_distances[vowel] = formant_dist
        formant_similarity[vowel] = similarity
    
    # サンプルの個別フォルマントとの比較（類似度計算）
    sample_similarities = {}
    very_similar_vowel = None
    
    if all_formants is not None:
        for vowel in VOWELS:
            vowel_samples = all_formants.get(vowel, [])
            if vowel_samples:
                # 各サンプルとの類似度を計算
                similarities = []
                for sample_formant in vowel_samples:
                    if len(sample_formant) >= 2 and len(user_formants) >= 2:
                        # F1とF2の距離を計算
                        f1_dist = abs(user_formants[0] - sample_formant[0]) / max_formant
                        f2_dist = abs(user_formants[1] - sample_formant[1]) / max_formant
                        
                        # 距離から類似度を計算
                        if vowel in ['i', 'e']:
                            dist = 0.3 * f1_dist + 0.7 * f2_dist
                        else:
                            dist = 0.5 * f1_dist + 0.5 * f2_dist
                        
                        sim = 1.0 / (1.0 + 10 * dist)
                        similarities.append(sim)
                
                # 最も類似度の高いサンプルを選択
                if similarities:
                    max_sim = max(similarities)
                    
                    # iの場合、類似度を優先的に高める（iをeより認識しやすく）
                    if vowel == 'i':
                        max_sim = min(1.0, max_sim * (1.0 + (IE_BIAS - 0.5) * 0.5))
                    elif vowel == 'e':
                        max_sim = max(0.0, max_sim * (1.0 - (IE_BIAS - 0.5) * 0.5))
                    
                    sample_similarities[vowel] = max_sim
                    
                    # 非常に高い類似度の場合、その母音を記録
                    if max_sim > VERY_SIMILAR_THRESHOLD and (very_similar_vowel is None or max_sim > sample_similarities.get(very_similar_vowel, 0)):
                        very_similar_vowel = vowel
    
    # 合成距離の計算（MFCCとフォルマントを組み合わせる）
    combined_distances = {}
    
    for vowel in templates.keys():
        # MFCCとフォルマントの距離を組み合わせる（重み付け）
        if vowel in ['i', 'e'] and IE_DISTINGUISH:
            # iとeの場合はフォルマントをより重視
            combined_distances[vowel] = 0.6 * mfcc_distances[vowel] + 0.4 * formant_distances[vowel] * 100
            
            # iをeよりも優先する調整を適用
            if vowel == 'i':
                # iの距離を短く（優先度を高く）調整
                combined_distances[vowel] *= (1.0 - (IE_BIAS - 0.5) * 0.6)
            elif vowel == 'e':
                # eの距離を長く（優先度を低く）調整
                combined_distances[vowel] *= (1.0 + (IE_BIAS - 0.5) * 0.6)
        else:
            # その他の母音はMFCCを重視
            combined_distances[vowel] = 0.8 * mfcc_distances[vowel] + 0.2 * formant_distances[vowel] * 100
    
    # 距離が近い順にソート
    sorted_distances = sorted(combined_distances.items(), key=lambda x: x[1])
    
    # サンプルとの類似度が非常に高い場合、その母音を優先
    if very_similar_vowel is not None:
        print(f"\n⭐ サンプルとの高い類似度検出: 「{very_similar_vowel}」 (類似度: {sample_similarities[very_similar_vowel]:.3f})")
        # 該当の母音を先頭に持ってくる
        sorted_distances = [(very_similar_vowel, combined_distances[very_similar_vowel])] + \
                          [d for d in sorted_distances if d[0] != very_similar_vowel]
    
    # 上位2つが「i」と「e」の場合、フォルマントで再判定
    elif IE_DISTINGUISH and len(sorted_distances) >= 2:
        first, first_dist = sorted_distances[0]
        second, second_dist = sorted_distances[1]
        
        if set([first, second]) == set(['i', 'e']):
            # 判別マージンを計算
            margin = abs(first_dist - second_dist)
            
            # iとeの第2フォルマントを比較
            i_f2 = formant_templates['i'][1]
            e_f2 = formant_templates['e'][1]
            user_f2 = user_formants[1]
            
            # 第2フォルマントの相対位置を計算
            # 0なら完全にeに近く、1なら完全にiに近い
            i_e_range = abs(i_f2 - e_f2)
            if i_e_range > 0:
                if i_f2 > e_f2:  # 通常はiの方が第2フォルマントが高い
                    rel_pos = min(1.0, max(0.0, (user_f2 - e_f2) / i_e_range))
                else:  # 万が一逆の場合
                    rel_pos = min(1.0, max(0.0, (e_f2 - user_f2) / i_e_range))
                
                # IE_BIASを適用（rel_posを調整）
                rel_pos = rel_pos * (2 * IE_BIAS) + (1 - IE_BIAS) * 2 - 1
                
                # 相対位置に基づいて判定
                if rel_pos > 0.5:  # iに近い
                    sorted_distances = [('i', combined_distances['i'])] + [d for d in sorted_distances if d[0] != 'i']
                else:  # eに近い
                    sorted_distances = [('e', combined_distances['e'])] + [d for d in sorted_distances if d[0] != 'e']
            else:
                # フォルマントが同じ場合はiを優先
                if IE_BIAS > 0.5:
                    sorted_distances = [('i', combined_distances['i'])] + [d for d in sorted_distances if d[0] != 'i']
                else:
                    sorted_distances = [('e', combined_distances['e'])] + [d for d in sorted_distances if d[0] != 'e']
            
            # フォルマント情報を表示
            print(f"\n🔍 フォルマント分析による「i」/「e」判別:")
            print(f"  i のF2: {i_f2:.0f}Hz, e のF2: {e_f2:.0f}Hz, あなたのF2: {user_f2:.0f}Hz")
            print(f"  相対位置: {rel_pos:.2f} (0=e寄り, 1=i寄り), バイアス: {IE_BIAS:.1f}")
    
    # 判別結果とともに各種距離情報を返す
    return sorted_distances, mfcc_distances, formant_distances, sample_similarities if 'sample_similarities' in locals() else {}

# === PCAによる次元圧縮（MFCC3,4のみなので2次元） ===
def fit_pca_with_data(X):
    # MFCC3とMFCC4の2次元なのでPCAは不要だが、互換性のため残す
    pca = PCA(n_components=2)  # 2次元に設定
    return pca, pca.fit_transform(X)

# === 2次元プロットの初期化（MFCC3,4のみ使用） ===
def init_3d_plot(X, y, pca, templates):
    # PCAを使わず直接MFCC3,4をプロット
    X_plot = X  # すでにMFCC3,4の2次元
    
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111)
    
    # 各母音クラスタをプロット（X軸とY軸を入れ替え）
    for vowel in VOWELS:
        cluster = X_plot[y == vowel]
        if len(cluster) > 0:
            color = COLOR_MAP.get(vowel, 'gray')
            ax.scatter(cluster[:, 1], cluster[:, 0],  # X軸とY軸を入れ替え
                      label=vowel, s=100, color=color, alpha=0.7, edgecolor='black')
            
            # クラスタの中心に母音ラベルを表示
            center = np.mean(cluster, axis=0)
            ax.text(center[1], center[0], vowel,  # X軸とY軸を入れ替え
                    fontsize=16, weight='bold', color=color, ha='center', va='center')
    
    # 曖昧母音の理論位置をプロット
    if 'ə' in templates:
        schwa_point = templates['ə']  # すでに2次元
        color = COLOR_MAP.get('ə', 'gray')
        ax.scatter(schwa_point[1], schwa_point[0],  # X軸とY軸を入れ替え
                  label='ə', s=150, color=color, alpha=0.7, marker='s', edgecolor='black')
        ax.text(schwa_point[1], schwa_point[0], 'ə',  # X軸とY軸を入れ替え
                fontsize=16, weight='bold', color=color, ha='center', va='center')
    
    # グラフの設定（軸ラベルも入れ替え）
    ax.set_title("🎯 MFCC3-4母音空間（フォルマント情報に対応）", fontsize=16)
    ax.set_xlabel("MFCC4（第5係数）")  # X軸とY軸を入れ替え
    ax.set_ylabel("MFCC3（第4係数）")  # X軸とY軸を入れ替え
    ax.legend(loc='upper right')
    ax.grid(True, linestyle='--', linewidth=0.5, alpha=0.7)
    
    # 軸の範囲を調整
    ax.axis('equal')
    
    return fig, ax, X_plot

# === ユーザーの母音点をプロットに追加・更新する ===
def update_user_point(ax, pca, user_vec, predicted_label, score, prev_scatter=None):
    # 以前の点を削除
    if prev_scatter:
        prev_scatter.remove()
    
    # ユーザーの特徴ベクトル（すでに2次元）
    user_point = user_vec
    
    # ユーザーの点をプロット（大きい赤い点、X軸とY軸を入れ替え）
    scatter = ax.scatter(user_point[1], user_point[0],  # X軸とY軸を入れ替え
                         color='red', s=200, marker='*', edgecolor='white', linewidth=2)
    
    # 予測結果とアドバイスをタイトルに表示
    advice = ADVICE_MAP.get(predicted_label, "練習を続けましょう。")
    ax.set_title(f"推定された母音: 「{predicted_label}」 (距離: {score:.2f})\n💡 {advice}", fontsize=14)
    
    plt.pause(0.01)  # 更新を即座に反映
    return scatter

# === 発音スコアに応じたアドバイス表示 ===
def show_advice(vowel, score):
    print("\n🧪 発音評価:")
    if score < 15:
        print("✅ 発音は良好です！")
    elif score < 30:
        print("⚠ 少しズレています。もう一度意識して発音してみましょう。")
    else:
        print("❌ 発音がかなりズレています。以下の改善点を参考にしてください。")
        print(f"🗣 「{vowel}」の発音アドバイス: {ADVICE_MAP.get(vowel, '練習を続けましょう。')}")

# === 主成分の寄与率を計算して表示 ===
def show_pca_contribution(pca):
    # MFCC3,4のみを使用しているため、PCAは実質的に不要
    print("\n📊 使用中の特徴量:")
    print("  MFCC3（第4係数）: 中域スペクトル構造")
    print("  MFCC4（第5係数）: 中域スペクトル構造")
    print("\n📖 MFCC3,4の特性:")
    print("  ・フォルマント周波数（特にF1,F2）と高い相関")
    print("  ・母音の音響的特徴を効率的に表現")
    print("  ・スペクトルの中域（約1000-3000Hz）の情報を主に反映")

# === 各母音のMFCC特徴量を可視化 ===
def visualize_mfcc_by_vowel(templates):
    """各母音のMFCC特徴量を可視化（MFCC3,4のみ）"""
    # データを準備
    vowels = list(templates.keys())
    mfcc_data = np.array([templates[v] for v in vowels])
    
    # 散布図でMFCC3,4の分布を表示
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # 各母音をプロット（X軸とY軸を入れ替え）
    for i, vowel in enumerate(vowels):
        color = COLOR_MAP.get(vowel, 'gray')
        ax.scatter(mfcc_data[i, 1], mfcc_data[i, 0],  # X軸とY軸を入れ替え
                  s=200, color=color, label=vowel, edgecolor='black', alpha=0.7)
        ax.text(mfcc_data[i, 1], mfcc_data[i, 0], vowel,  # X軸とY軸を入れ替え
                fontsize=14, ha='center', va='center', weight='bold', color='white')
    
    # 軸の設定（ラベルも入れ替え）
    ax.set_xlabel('MFCC4（第5係数）', fontsize=14)  # X軸とY軸を入れ替え
    ax.set_ylabel('MFCC3（第4係数）', fontsize=14)  # X軸とY軸を入れ替え
    ax.set_title('各母音のMFCC3-4分布', fontsize=16)
    ax.grid(True, linestyle='--', alpha=0.5)
    ax.legend(loc='best')
    
    plt.tight_layout()
    plt.savefig('mfcc34_scatter.png')
    print("📊 MFCC3-4の分布図を 'mfcc34_scatter.png' に保存しました")
    
    # 各母音のMFCC3,4の値を表示
    print("\n📊 各母音のMFCC3,4の値:")
    for i, vowel in enumerate(vowels):
        print(f"  {vowel}: MFCC3={mfcc_data[i, 0]:.2f}, MFCC4={mfcc_data[i, 1]:.2f}")
    
    # MFCC3と4の母音識別力を強調表示
    print("\n📊 使用中のMFCC係数（MFCC3,4）の母音識別力:")
    print(f"  MFCC3: 中域スペクトル構造（フォルマント情報）")
    print(f"  MFCC4: 中域スペクトル構造（フォルマント情報）")
    print("\n  ※ MFCC3,4はフォルマント周波数と相関が高く、母音識別に重要な特徴です")

# === フォルマント情報の表示 ===
def display_formant_info(formant_templates):
    """各母音のフォルマント情報を表示"""
    print("\n🔍 母音のフォルマント情報:")
    for vowel, formants in formant_templates.items():
        if len(formants) >= 2:
            print(f"  「{vowel}」: F1={formants[0]:.0f}Hz, F2={formants[1]:.0f}Hz")
            
            # 特にiとeの違いを詳細表示
            if vowel in ['i', 'e']:
                print(f"    👉 「{vowel}」の特徴: {'高い第2フォルマント' if vowel == 'i' else '中程度の第2フォルマント'}")

# === フォルマントチャートの作成 ===
def create_formant_chart(formant_templates):
    """F1-F2平面でのフォルマントチャートを作成"""
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # 各母音のフォルマントをプロット
    for vowel, formants in formant_templates.items():
        if len(formants) >= 2:
            # F1とF2をプロット（母音音響学の慣例に従い、F1は上下反転、F2は左右反転）
            color = COLOR_MAP.get(vowel, 'gray')
            ax.scatter(formants[1], formants[0], s=100, color=color, label=vowel, edgecolor='black')
            ax.text(formants[1], formants[0], vowel, fontsize=16, ha='center', va='center', weight='bold')
    
    # 軸の設定（慣例に従い反転）
    ax.set_xlim(3000, 500)  # F2は高い周波数から低い周波数へ
    ax.set_ylim(1000, 200)  # F1は高い周波数から低い周波数へ
    
    ax.set_xlabel('第2フォルマント (F2) [Hz]')
    ax.set_ylabel('第1フォルマント (F1) [Hz]')
    ax.set_title('母音フォルマントチャート', fontsize=16)
    ax.grid(True, linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    
    # ファイルに保存
    plt.savefig('formant_chart.png')
    print("📊 フォルマントチャートを 'formant_chart.png' に保存しました")
    
    return fig

# === 伝統的な母音図の作成 ===
def create_vowel_chart():
    """
    伝統的な母音図（舌の位置に基づく）を作成
    
    Returns:
    - fig: matplotlib図オブジェクト
    - ax: 軸オブジェクト
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # 背景に母音四角形を描画
    # - 輪郭
    ax.plot([0.1, 0.1, 0.9, 0.9, 0.1], [0.1, 0.9, 0.9, 0.1, 0.1], 'k-', alpha=0.3)
    # - 水平線（高、中、低）
    ax.plot([0.1, 0.9], [0.1, 0.1], 'k--', alpha=0.2)  # 高
    ax.plot([0.1, 0.9], [0.5, 0.5], 'k--', alpha=0.2)  # 中
    ax.plot([0.1, 0.9], [0.9, 0.9], 'k--', alpha=0.2)  # 低
    # - 垂直線（前、中、後）
    ax.plot([0.1, 0.1], [0.1, 0.9], 'k--', alpha=0.2)  # 前
    ax.plot([0.5, 0.5], [0.1, 0.9], 'k--', alpha=0.2)  # 中
    ax.plot([0.9, 0.9], [0.1, 0.9], 'k--', alpha=0.2)  # 後
    
    # 各母音の位置をプロット（曖昧母音を含む）
    for vowel, (x, y) in VOWEL_POSITIONS.items():
        color = COLOR_MAP.get(vowel, 'gray')
        # 曖昧母音は特別なマーカーで表示
        if vowel == 'ə':
            ax.scatter(x, y, s=200, color=color, alpha=0.7, edgecolor='black', 
                      zorder=10, marker='s')  # 四角マーカー
        else:
            ax.scatter(x, y, s=200, color=color, alpha=0.7, edgecolor='black', zorder=10)
        ax.text(x, y, IPA_SYMBOLS[vowel], fontsize=16, ha='center', va='center', 
                color='white', weight='bold', zorder=11)
    
    # 軸ラベルと設定
    ax.set_xlim(0, 1)
    ax.set_ylim(1, 0)  # 上下反転（上が高、下が低）
    
    # 軸ラベル
    ax.text(0.5, -0.05, '前舌 ← → 後舌', ha='center', va='center', fontsize=14)
    ax.text(-0.05, 0.5, '高 ↑\n↓ 低', ha='center', va='center', fontsize=14)
    
    # 象限ラベル
    ax.text(0.25, 0.2, '前舌・高', ha='center', va='center', alpha=0.5)
    ax.text(0.25, 0.8, '前舌・低', ha='center', va='center', alpha=0.5)
    ax.text(0.75, 0.2, '後舌・高', ha='center', va='center', alpha=0.5)
    ax.text(0.75, 0.8, '後舌・低', ha='center', va='center', alpha=0.5)
    
    # タイトルと軸の表示設定
    ax.set_title('母音図（舌の位置による分類）', fontsize=16)
    ax.set_xticks([])
    ax.set_yticks([])
    
    # 枠を非表示
    for spine in ax.spines.values():
        spine.set_visible(False)
    
    return fig, ax

# === ユーザーの発音を母音図にプロット ===
def plot_user_vowel(ax, f1, f2, predicted_label, prev_scatter=None, prev_text=None):
    """
    ユーザーの母音を母音図にプロット
    
    Parameters:
    - ax: 母音図の軸オブジェクト
    - f1, f2: ユーザーの第1, 第2フォルマント
    - predicted_label: 予測された母音
    - prev_scatter: 前回のプロット（削除用）
    - prev_text: 前回のテキスト（削除用）
    
    Returns:
    - scatter: 新しいプロットオブジェクト
    - text: 新しいテキストオブジェクト
    """
    # 前回のプロットを削除
    if prev_scatter:
        prev_scatter.remove()
    if prev_text:
        prev_text.remove()
    
    # フォルマントから舌位置に変換
    x, y = formants_to_tongue_position(f1, f2)
    
    # ユーザーの母音をプロット
    scatter = ax.scatter(x, y, s=300, color='red', marker='*', 
                         edgecolor='white', linewidth=1.5, zorder=20)
    
    # 推定結果をテキストで表示
    text = ax.text(x, y - 0.1, f'推定: {predicted_label}', color='red', 
                   fontsize=12, ha='center', va='center', weight='bold',
                   bbox=dict(facecolor='white', alpha=0.7, boxstyle='round'),
                   zorder=21)
    
    # 母音図のタイトルを更新（アドバイス付き）
    advice = ADVICE_MAP.get(predicted_label, "練習を続けましょう。")
    ax.set_title(f'母音図 - 推定: 「{predicted_label}」\n💡 {advice}', fontsize=14)
    
    plt.pause(0.01)  # 表示を更新
    return scatter, text


# === メイン処理 ===
def main():
    print("📦 テンプレート読み込み中...")
    X_mfcc, X_formants, y, all_formants = extract_features()
    
    if len(X_mfcc) == 0:
        print("❌ テンプレートがありません。recordings フォルダを確認してください。")
        return

    # テンプレート作成
    templates, formant_templates = build_templates(X_mfcc, X_formants, y)
    
    # フォルマント情報の表示
    display_formant_info(formant_templates)
    
    # 各母音のMFCC特徴量を可視化
    visualize_mfcc_by_vowel(templates)
    
    # フォルマントチャートの作成
    formant_fig = create_formant_chart(formant_templates)
    
    # 伝統的な母音図の作成
    print("🔊 母音図を生成中...")
    vowel_chart_fig, vowel_chart_ax = create_vowel_chart()
    vowel_chart_scatter = None
    vowel_chart_text = None
    
    
    # PCAモデル学習（MFCCのみ使用）
    pca, _ = fit_pca_with_data(X_mfcc)
    show_pca_contribution(pca)

    plt.ion()  # 対話モードON
    fig, ax, _ = init_3d_plot(X_mfcc, y, pca, templates)

    print(f"\n🔧 「i」と「e」の判別: {'フォルマント分析を使用' if IE_DISTINGUISH else '無効'}")
    print(f"💫 個別サンプルとの類似度比較: 有効 (閾値: {VERY_SIMILAR_THRESHOLD})")
    print(f"🎯 「i」優先度: {IE_BIAS:.1f} (0.5=中立, 1.0=最大)")
    print("\n📌 MFCC3,4のみを使用した母音認識モード")
    print("  → MFCC3,4は中域のスペクトル構造を表し、フォルマント情報と相関")
    print("🟢 リアルタイム母音認識を開始します（Ctrl+Cで停止）")
    prev_scatter = None

    try:
        while True:
            audio_path = "user_input.wav"
            record_audio(audio_path)  # 録音
            
            # 特徴抽出（MFCCとフォルマント）
            user_mfcc, user_formants = extract_user_features(audio_path)
            
            # 分類
            results, mfcc_distances, formant_distances, sample_similarities = classify(
                user_mfcc, user_formants, templates, formant_templates, all_formants)
            
            predicted, dist = results[0]  # 最も近い母音と距離

            # 結果表示
            print("\n=== 判定結果 ===")
            print(f"🗣 推定: 「{predicted}」 / 距離スコア: {dist:.2f}")
            
            # フォルマント情報の表示
            f1, f2 = user_formants[0], user_formants[1]
            print(f"📊 あなたのフォルマント: F1={f1:.0f}Hz, F2={f2:.0f}Hz, F3={user_formants[2]:.0f}Hz")
            
            
            # サンプルとの類似度情報表示
            if sample_similarities:
                print("📊 サンプルとの類似度:")
                for vowel, sim in sorted(sample_similarities.items(), key=lambda x: x[1], reverse=True)[:3]:
                    print(f"  「{vowel}」: {sim:.3f}" + (" ⭐" if vowel == 'i' and sim > 0.7 else ""))
            
            print("📊 類似度ランキング:")
            for i, (v, d) in enumerate(results):
                print(f"  {i+1}. {v}（距離: {d:.2f}）")

            show_advice(predicted, dist)  # アドバイス表示
            
            # PCA空間にプロットを更新
            prev_scatter = update_user_point(ax, pca, user_mfcc, predicted, dist, prev_scatter)
            
            # === 母音図を更新 ===
            vowel_chart_scatter, vowel_chart_text = plot_user_vowel(
                vowel_chart_ax, f1, f2, predicted, vowel_chart_scatter, vowel_chart_text)
            
            
            # 2Dプロットなので視点回転は不要
            
            sleep(0.2)

    except KeyboardInterrupt:
        print("\n🛑 終了しました。")
        plt.ioff()  # 対話モードOFF
        plt.close('all')

# === エントリーポイント ===
if __name__ == "__main__":
    # recordingsディレクトリの存在確認
    if not os.path.exists(RECORDINGS_DIR):
        print(f"📁 {RECORDINGS_DIR}ディレクトリを作成します...")
        os.makedirs(RECORDINGS_DIR)
        print(f"⚠️ {RECORDINGS_DIR}ディレクトリに母音サンプルを追加してください")
    else:
        main()