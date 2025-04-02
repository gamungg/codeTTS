import librosa
import numpy as np
from pydub import AudioSegment
from pydub.silence import detect_silence
import pyloudnorm as pyln
import soundfile as sf
import whisper
from collections import defaultdict


# 1. 音频清晰度检测（背景噪声）
def detect_background_noise(audio_path, threshold_db=-40):
    """检测恒定背景噪声（频谱能量低于阈值）"""
    y, sr = librosa.load(audio_path, sr=None)
    S = librosa.amplitude_to_db(np.abs(librosa.stft(y)), ref=np.max)

    # 分析低频和高频噪声
    noise_bands = {
        'low_freq': np.mean(S[:50, :]),  # 0-500Hz (假设sr=22050)
        'high_freq': np.mean(S[-100:, :])  # 8kHz以上
    }

    issues = []
    for band, value in noise_bands.items():
        if value > threshold_db:
            issues.append(f"Background noise in {band} band: {value:.2f}dB")

    return issues


# 2. 静音段检测
def check_silence(audio_path, min_silence_len=50, silence_thresh=-40):
    """检测开头/结尾静音是否合规"""
    audio = AudioSegment.from_wav(audio_path)
    silence_ranges = detect_silence(audio, min_silence_len, silence_thresh)

    issues = []
    if len(silence_ranges) == 0:
        return ["No silence detected"]

    # 检查开头静音
    if silence_ranges[0][0] > 200:  # 开头静音超过200ms
        issues.append(f"Leading silence too long: {silence_ranges[0][0]}ms")

    # 检查结尾静音
    if audio.duration_seconds * 1000 - silence_ranges[-1][1] > 200:
        issues.append("Trailing silence too long")

    return issues


# 3. 音量标准化与削波检测
def check_loudness_clipping(audio_path):
    """检测响度合规性与削波"""
    y, sr = sf.read(audio_path)
    meter = pyln.Meter(sr)
    loudness = meter.integrated_loudness(y)

    issues = []
    # 响度范围检测 (-23 ± 2 LUFS)
    if not (-25 <= loudness <= -21):
        issues.append(f"Loudness out of range: {loudness:.2f} LUFS")

    # 削波检测
    if np.max(np.abs(y)) >= 1.0:
        issues.append("Clipping detected (amplitude >= 1.0)")

    return issues


# 4. 语速一致性检测
def check_speech_rate(audio_path, text):
    """通过ASR结果计算语速（字/秒）"""
    model = whisper.load_model("tiny")
    result = model.transcribe(audio_path)
    transcript = result["text"]

    # 计算实际字数与理论字数比
    actual_words = len(transcript)
    expected_words = len(text)
    ratio = actual_words / expected_words

    issues = []
    if not 0.9 <= ratio <= 1.1:
        issues.append(f"Speech/text mismatch: {ratio:.2%}")

    return issues


# 5. 技术参数检测
def check_technical_specs(audio_path):
    """检查采样率/位深/通道数"""
    try:
        with sf.SoundFile(audio_path) as f:
            issues = []
            if f.samplerate < 16000:
                issues.append(f"Low sample rate: {f.samplerate}Hz")
            if f.subtype not in ['PCM_16', 'PCM_24']:
                issues.append(f"Invalid bit depth: {f.subtype}")
            if f.channels != 1:
                issues.append(f"Not mono: {f.channels} channels")
            return issues
    except Exception as e:
        return [f"File corrupted: {str(e)}"]


# 6. 自动化检测整合
def audio_qc_pipeline(audio_path, text):
    """综合质量检测管道"""
    report = defaultdict(list)

    # 各检测项执行
    report["noise"] = detect_background_noise(audio_path)
    report["silence"] = check_silence(audio_path)
    report["loudness"] = check_loudness_clipping(audio_path)
    report["speech_rate"] = check_speech_rate(audio_path, text)
    report["tech_specs"] = check_technical_specs(audio_path)

    # 生成报告
    total_issues = sum(len(v) for v in report.values())
    report["summary"] = {
        "pass": total_issues == 0,
        "total_issues": total_issues
    }
    return report


# 运行示例
if __name__ == "__main__":
    report = audio_qc_pipeline("D:/Desktop/essay/4/hjy0001.wav", "我喜欢看电视")
    print("Audio Quality Report:")
    for category, issues in report.items():
        if category == "summary":
            print(f"\nSummary: {'PASS' if issues['pass'] else 'FAIL'} ({issues['total_issues']} issues)")
        else:
            print(f"\n{category.capitalize()}:")
            if not issues:
                print("  ✓ No issues detected")
            else:
                for issue in issues:
                    print(f"  ✗ {issue}")