"""
MFCC13次元を使用した母音認識システム
全13次元のMFCC特徴量を活用して高精度な母音認識を実現
"""

# === ライブラリのインポート ===
import os
import numpy as np
import sounddevice as sd
import soundfile as sf
import librosa
import matplotlib.pyplot as plt
from time import sleep

# === フォント設定（Mac対応） ===
plt.rcParams['font.family'] = 'AppleGothic'  # Mac用フォント

# === 音声処理設定 ===
RATE = 16000  # サンプリングレート（16kHz）
DURATION = 1.0  # 録音時間（秒）
N_MFCC = 13  # MFCCの次元数（全13次元使用）
RECORDINGS_DIR = "recordings_formant"  # 録音ファイルの保存先
VOWELS = ['a', 'i', 'u', 'e', 'o']  # 対象とする母音
SAMPLES_PER_VOWEL = 3  # 母音ごとのサンプル数

# === 母音の色マッピング ===
COLOR_MAP = {
    'a': 'red',
    'i': 'blue',
    'u': 'green',
    'e': 'purple',
    'o': 'orange'
}

# === 母音ごとの発音アドバイス ===
ADVICE_MAP = {
    'a': "口を大きく縦に開け、舌は下に落としましょう。",
    'i': "口を横に引いて、舌は前に出すようにしましょう。",
    'u': "唇をすぼめて、舌を後ろに引きましょう。",
    'e': "口角を少し上げ、舌をやや前に出しましょう。",
    'o': "唇を丸く突き出し、舌を後ろに引きましょう。"
}

# === 音声ファイルからMFCC特徴量を抽出 ===
def extract_features():
    """音声ファイルから13次元MFCCを抽出"""
    X_mfcc, y = [], []
    
    for vowel in VOWELS:
        for i in range(1, SAMPLES_PER_VOWEL + 1):
            filepath = os.path.join(RECORDINGS_DIR, f"{vowel}_{i}.wav")
            if os.path.exists(filepath):
                # 音声読み込み
                y_data, sr = librosa.load(filepath, sr=RATE)
                
                # 無音区間除去
                y_data, _ = librosa.effects.trim(y_data, top_db=20)
                
                # MFCC抽出（13次元すべて）
                mfcc = librosa.feature.mfcc(y=y_data, sr=sr, n_mfcc=N_MFCC)
                X_mfcc.append(np.mean(mfcc, axis=1))  # 時間方向の平均
                y.append(vowel)
            else:
                print(f"⚠️ ファイルが見つかりません: {filepath}")
    
    return np.array(X_mfcc), np.array(y)

# === 母音ごとにMFCCの平均テンプレートを作成 ===
def build_templates(X_mfcc, y):
    """各母音のMFCCテンプレートを作成"""
    templates = {}
    
    for vowel in VOWELS:
        indices = y == vowel
        if np.any(indices):
            templates[vowel] = np.mean(X_mfcc[indices], axis=0)
    
    return templates

# === ユーザーの発音を録音 ===
def record_audio(path):
    """マイクから音声を録音"""
    print("🎤 発音してください...")
    audio = sd.rec(int(RATE * DURATION), samplerate=RATE, channels=1)
    sd.wait()
    sf.write(path, audio, RATE)

# === 録音された音声から特徴を抽出 ===
def extract_user_features(filepath):
    """ユーザーの音声から13次元MFCCを抽出"""
    # 音声読み込み
    y_data, sr = librosa.load(filepath, sr=RATE)
    
    # 無音区間除去
    y_data, _ = librosa.effects.trim(y_data, top_db=20)
    
    # MFCC抽出（13次元すべて）
    mfcc = librosa.feature.mfcc(y=y_data, sr=sr, n_mfcc=N_MFCC)
    mfcc_features = np.mean(mfcc, axis=1)
    
    return mfcc_features

# === ユーザーのMFCCをテンプレートと比較して分類 ===
def classify(user_mfcc, templates):
    """13次元MFCCによる母音分類"""
    # 各母音テンプレートとのユークリッド距離を計算
    distances = {}
    for vowel, template_mfcc in templates.items():
        # 13次元全体での距離を計算
        distance = np.linalg.norm(user_mfcc - template_mfcc)
        distances[vowel] = distance
    
    # 距離が近い順にソート
    sorted_distances = sorted(distances.items(), key=lambda x: x[1])
    
    return sorted_distances, distances

# === 発音スコアに応じたアドバイス表示 ===
def show_advice(vowel, score):
    """発音に対するアドバイスを表示"""
    print("\n🧪 発音評価:")
    if score < 5:
        print("✅ 非常に良い発音です！")
    elif score < 10:
        print("⭐ 良い発音です！")
    elif score < 15:
        print("⚠️ もう少し練習しましょう。")
    else:
        print("❌ 発音を改善しましょう。以下のアドバイスを参考にしてください。")
        print(f"🗣 「{vowel}」の発音アドバイス: {ADVICE_MAP.get(vowel, '練習を続けましょう。')}")

# === MFCC係数の重要度を分析・表示 ===
def analyze_mfcc_importance(X_mfcc, y, templates):
    """13次元MFCCの各係数の重要度を分析"""
    print("\n📊 13次元MFCCによる母音識別分析:")
    print(f"  使用しているMFCC係数: MFCC0〜MFCC12 (全13次元)")
    print(f"  サンプル数: {X_mfcc.shape[0]}")
    
    # 母音ごとの特徴を計算
    mfcc_by_vowel = {}
    for vowel in VOWELS:
        indices = y == vowel
        if np.any(indices):
            vowel_data = X_mfcc[indices]
            mfcc_by_vowel[vowel] = {
                'mean': np.mean(vowel_data, axis=0),
                'std': np.std(vowel_data, axis=0)
            }
    
    # 各MFCC係数の母音間分散を計算（識別力の指標）
    mfcc_variances = []
    for i in range(N_MFCC):
        values = [mfcc_by_vowel[v]['mean'][i] for v in VOWELS if v in mfcc_by_vowel]
        if values:
            mfcc_variances.append(np.var(values))
        else:
            mfcc_variances.append(0)
    
    # 重要度でソート
    importance_ranking = np.argsort(mfcc_variances)[::-1]
    
    print("\n📊 母音識別に重要なMFCC係数（分散が大きい順）:")
    for rank, idx in enumerate(importance_ranking[:5]):
        print(f"  {rank+1}. MFCC{idx}: 分散={mfcc_variances[idx]:.3f}")
    
    # 可視化
    create_mfcc_analysis_plots(mfcc_by_vowel, mfcc_variances, X_mfcc, y)

# === MFCC分析結果の可視化 ===
def create_mfcc_analysis_plots(mfcc_by_vowel, mfcc_variances, X_mfcc, y):
    """13次元MFCCの分析結果を可視化"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 1. 各母音のMFCCプロファイル
    ax1 = axes[0, 0]
    for vowel in VOWELS:
        if vowel in mfcc_by_vowel:
            color = COLOR_MAP.get(vowel, 'gray')
            mean_vals = mfcc_by_vowel[vowel]['mean']
            ax1.plot(range(N_MFCC), mean_vals, 
                    marker='o', label=vowel, color=color, linewidth=2)
    
    ax1.set_xlabel('MFCC係数')
    ax1.set_ylabel('平均値')
    ax1.set_title('各母音の13次元MFCCプロファイル')
    ax1.set_xticks(range(N_MFCC))
    ax1.set_xticklabels([f'{i}' for i in range(N_MFCC)])
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    
    # 2. MFCC係数の重要度（母音間分散）
    ax2 = axes[0, 1]
    bars = ax2.bar(range(N_MFCC), mfcc_variances)
    ax2.set_xlabel('MFCC係数')
    ax2.set_ylabel('母音間分散')
    ax2.set_title('各MFCC係数の母音識別力')
    ax2.set_xticks(range(N_MFCC))
    ax2.set_xticklabels([f'{i}' for i in range(N_MFCC)])
    ax2.grid(True, alpha=0.3, axis='y')
    
    # 重要度の高い係数を色分け
    max_var = np.max(mfcc_variances)
    for i, bar in enumerate(bars):
        if mfcc_variances[i] > max_var * 0.7:
            bar.set_color('darkred')
        elif mfcc_variances[i] > max_var * 0.4:
            bar.set_color('orange')
        else:
            bar.set_color('lightcoral')
    
    # 3. 母音ごとのMFCCヒートマップ
    ax3 = axes[1, 0]
    vowel_list = [v for v in VOWELS if v in mfcc_by_vowel]
    mfcc_matrix = np.array([mfcc_by_vowel[v]['mean'] for v in vowel_list])
    
    im = ax3.imshow(mfcc_matrix.T, cmap='RdBu_r', aspect='auto')
    ax3.set_xlabel('母音')
    ax3.set_ylabel('MFCC係数')
    ax3.set_title('母音別MFCC係数の値（ヒートマップ）')
    ax3.set_xticks(range(len(vowel_list)))
    ax3.set_xticklabels(vowel_list)
    ax3.set_yticks(range(N_MFCC))
    ax3.set_yticklabels([f'MFCC{i}' for i in range(N_MFCC)])
    
    # カラーバー
    cbar = plt.colorbar(im, ax=ax3, fraction=0.046, pad=0.04)
    cbar.set_label('MFCC値')
    
    # 4. 特定のMFCC係数での母音分布
    ax4 = axes[1, 1]
    # 最も識別力の高い2つのMFCC係数を選択
    top_indices = np.argsort(mfcc_variances)[::-1][:2]
    mfcc1_idx, mfcc2_idx = top_indices[0], top_indices[1]
    
    for vowel in VOWELS:
        indices = y == vowel
        if np.any(indices):
            vowel_data = X_mfcc[indices]
            color = COLOR_MAP.get(vowel, 'gray')
            ax4.scatter(vowel_data[:, mfcc1_idx], vowel_data[:, mfcc2_idx],
                       color=color, label=vowel, alpha=0.7, s=100)
    
    ax4.set_xlabel(f'MFCC{mfcc1_idx} 値')
    ax4.set_ylabel(f'MFCC{mfcc2_idx} 値')
    ax4.set_title(f'最も識別力の高いMFCC係数での母音分布')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('mfcc13_analysis.png', dpi=150)
    print("📊 13次元MFCC分析結果を 'mfcc13_analysis.png' に保存しました")
    plt.show()

# === リアルタイム可視化用のプロット初期化 ===
def init_realtime_plot(templates):
    """リアルタイム表示用のプロットを初期化"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # 左側：MFCCプロファイル表示
    ax1.set_xlabel('MFCC係数')
    ax1.set_ylabel('値')
    ax1.set_title('MFCCプロファイル（リアルタイム）')
    ax1.set_xlim(-0.5, N_MFCC-0.5)
    ax1.set_ylim(-30, 30)
    ax1.grid(True, alpha=0.3)
    ax1.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    
    # テンプレートをプロット
    for vowel, template in templates.items():
        color = COLOR_MAP.get(vowel, 'gray')
        ax1.plot(range(N_MFCC), template, 
                color=color, alpha=0.3, linewidth=1, linestyle='--')
    
    # ユーザーのMFCC用のライン（初期化）
    user_line, = ax1.plot([], [], 'r-', linewidth=2, marker='o', markersize=6)
    
    # 右側：距離バーチャート
    ax2.set_xlabel('母音')
    ax2.set_ylabel('距離')
    ax2.set_title('各母音との距離')
    ax2.set_ylim(0, 30)
    ax2.grid(True, alpha=0.3, axis='y')
    
    # 初期バー
    bars = ax2.bar(VOWELS, [0]*len(VOWELS), 
                   color=[COLOR_MAP[v] for v in VOWELS])
    
    plt.tight_layout()
    return fig, ax1, ax2, user_line, bars

# === リアルタイムプロットの更新 ===
def update_realtime_plot(user_mfcc, distances, predicted, user_line, bars, ax1, ax2):
    """リアルタイムプロットを更新"""
    # MFCCプロファイルを更新
    user_line.set_data(range(N_MFCC), user_mfcc)
    
    # 距離バーを更新
    dist_values = [distances[v] for v in VOWELS]
    for bar, dist in zip(bars, dist_values):
        bar.set_height(dist)
    
    # 予測結果をハイライト
    for i, vowel in enumerate(VOWELS):
        if vowel == predicted:
            bars[i].set_edgecolor('red')
            bars[i].set_linewidth(3)
        else:
            bars[i].set_edgecolor('black')
            bars[i].set_linewidth(1)
    
    # タイトル更新
    ax1.set_title(f'MFCCプロファイル - 推定: 「{predicted}」')
    
    plt.pause(0.01)

# === メイン処理 ===
def main():
    print("🎵 13次元MFCCを使用した母音認識システム")
    print("📦 テンプレート読み込み中...")
    
    # 特徴抽出
    X_mfcc, y = extract_features()
    
    if len(X_mfcc) == 0:
        print("❌ テンプレートがありません。recordings_formant フォルダを確認してください。")
        return
    
    # テンプレート作成
    templates = build_templates(X_mfcc, y)
    
    # 初回のみMFCC分析を表示（その後はスキップ）
    if not hasattr(main, 'analyzed'):
        analyze_mfcc_importance(X_mfcc, y, templates)
        main.analyzed = True
        input("\n📊 分析が完了しました。Enterキーを押してリアルタイム認識を開始...")
    
    # リアルタイムプロット初期化
    fig, ax1, ax2, user_line, bars = init_realtime_plot(templates)
    plt.ion()  # 対話モードON
    
    print("\n🟢 リアルタイム母音認識を開始します（Ctrl+Cで停止）")
    print("📊 13次元すべてのMFCC係数を使用して高精度な認識を行います")
    
    try:
        while True:
            audio_path = "user_input.wav"
            record_audio(audio_path)  # 録音
            
            # 特徴抽出（13次元MFCC）
            user_mfcc = extract_user_features(audio_path)
            
            # 分類
            results, distances = classify(user_mfcc, templates)
            predicted, dist = results[0]  # 最も近い母音と距離
            
            # 結果表示
            print("\n=== 判定結果 ===")
            print(f"🗣 推定: 「{predicted}」 / 距離スコア: {dist:.2f}")
            
            # MFCCの詳細情報
            print(f"📊 13次元MFCC使用")
            print("📊 類似度ランキング:")
            for i, (v, d) in enumerate(results):
                print(f"  {i+1}. {v}（距離: {d:.2f}）")
            
            # アドバイス表示
            show_advice(predicted, dist)
            
            # リアルタイムプロット更新
            update_realtime_plot(user_mfcc, distances, predicted, 
                               user_line, bars, ax1, ax2)
            
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
        print("💡 test3-1.py を実行してサンプルを録音できます")
    else:
        main()