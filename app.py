import librosa
import numpy as np
import os
import streamlit as st
import tempfile
import torch
import torchaudio
from scipy.signal import lfilter, butter
import pandas as pd
import soundfile as sf
import wave
import struct
from webrtcvad import Vad
import io

from silero_vad import load_silero_vad, get_speech_timestamps, read_audio

# GUI 界面
st.title("🔍 TTS 训练音频质检与标注系统")

# 侧边栏选择功能板块
st.sidebar.title("选择功能板块")
app_mode = st.sidebar.radio("请选择功能", ["音频质检", "文本标注"])

# 音频质检板块
if app_mode == "音频质检":
    st.header("音频质检")

    # 用户选择质检方式
    check_type = st.radio("选择质检方式", ("📂 批量检查文件夹", "📁 单个文件上传"))

    # 上传音频文件
    uploaded_files = []
    folder_path = ""
    if check_type == "📂 批量检查文件夹":
        folder_path = st.text_input("请输入文件夹路径（包含音频文件）")
        if folder_path:
            # 获取文件夹中的所有音频文件
            try:
                uploaded_files = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if
                                  f.endswith(('.mp3', '.wav'))]
                if not uploaded_files:
                    st.error("文件夹中没有找到有效的音频文件！")
            except FileNotFoundError:
                st.error("文件夹路径无效，请重新输入！")
    else:
        uploaded_files = st.file_uploader("上传音频文件（支持 WAV/MP3）", type=["wav", "mp3"], accept_multiple_files=True)

    # 质检参数设置
    snr_threshold = st.slider("信噪比阈值 (dB)", min_value=10, max_value=40, value=15)
    silence_threshold = st.slider("静音比例阈值 (%)", min_value=0, max_value=50, value=20)

    # 改进后的 SNR 检测函数
    def calculate_snr(y, sr, noise_duration=0.3):
        """
        Enhanced SNR Calculation using NLMS-based Noise Estimation and Improved Filtering.
        """
        # Step 1: Apply a wider bandpass filter (100 Hz - 8000 Hz)
        b, a = butter(6, [100 / (sr / 2), 8000 / (sr / 2)], btype='band')
        y = lfilter(b, a, y)

        # Step 2: Compute total signal power using RMS (Root Mean Square)
        signal_power = np.mean(y ** 2)

        # Step 3: Improved Noise Estimation using NLMS-based detection
        # 使用自适应滤波器分离噪声
        noise_sample = y[:int(noise_duration * sr)]
        d = y
        x = np.concatenate([np.zeros(100), noise_sample])  # 噪声作为参考信号
        e = nlms_filter(y, d, M=100, mu=0.1)

        # Step 4: Calculate noise power using RMS
        noise_power = np.mean(e ** 2)

        # 避免除以零的情况
        if noise_power < 1e-6:
            noise_power = 1e-6

        # Step 5: Calculate SNR in dB
        snr = 10 * np.log10(signal_power / noise_power)

        return snr

    # 自适应滤波器实现
    def nlms_filter(x, d, M=100, mu=0.1):
        """
        使用归一化最小均方（NLMS）算法实现自适应滤波器。
        """
        N = len(x)
        w = np.zeros(M)
        y = np.zeros(N)
        e = np.zeros(N)

        for n in range(N):
            if n >= M:
                x_n = x[n - M:n][::-1]
                y[n] = np.dot(w, x_n)
                e[n] = d[n] - y[n]
                x_power = np.sum(x_n ** 2)
                if x_power > 0:
                    w += mu / x_power * e[n] * x_n
            else:
                e[n] = d[n]  # 对于 n < M 的情况，直接将误差设为输入信号

        return e

# 改进后的 VAD 检测函数
model = load_silero_vad()  # 加载模型

def vad_check(file_path):
    """
    Advanced VAD Detection using Silero VAD Model with Improved Filtering.
    """
    try:
        wav = read_audio(file_path, sampling_rate=16000)
        # 应用自适应滤波器
        d = wav
        x = np.concatenate([np.zeros(100), wav[:int(0.3 * 16000)]])  # 噪声作为参考信号
        e = nlms_filter(wav, d, M=100, mu=0.1)
        wav = e  # 使用滤波后的信号进行VAD检测

        # 确保采样率正确
        speech_timestamps = get_speech_timestamps(wav, model=model, sampling_rate=16000)

        return len(speech_timestamps) >= 2, speech_timestamps

    except Exception as e:
        st.error(f"VAD 检测错误：{e}")
        return False, []

# 质检函数
def quality_check(file_path):
    results = {}

    # 音频格式检查
    results["格式检查"] = check_audio_format(file_path)

    # 音频时长检查
    results["时长检查"] = check_audio_duration(file_path)

    # 信噪比检查
    snr_status, snr_value = check_snr(file_path)
    results["信噪比检查"] = (snr_status, snr_value)

    # 噪声检查
    results["噪声检查"] = noise_check(file_path)

    # 语音活动检测
    vad_status, speech_timestamps = vad_check(file_path)
    results["语音活动检测"] = (vad_status, len(speech_timestamps))

    # 动态范围计算
    y, sr = librosa.load(file_path, sr=None)
    results["动态范围"] = calculate_dynamic_range(y)

    # 总谐波失真计算
    results["总谐波失真"] = calculate_thd(y, sr)

    return results

# 执行质检
if uploaded_files:
    all_results = []

    for uploaded_file in uploaded_files:
        if check_type == "📂 批量检查文件夹":
            file_path = uploaded_file
        else:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_file:
                temp_file.write(uploaded_file.read())
                file_path = temp_file.name

        # 执行质检
        qc_results = quality_check(file_path)

        # 整理结果
        file_name = os.path.basename(file_path)
        result_entry = {"文件名": file_name}

        if qc_results["格式检查"]:
            result_entry["格式检查"] = "✅ 合格"
        else:
            result_entry["格式检查"] = "❌ 不合格 - 格式错误"

        if qc_results["时长检查"]:
            result_entry["时长检查"] = "✅ 合格"
        else:
            result_entry["时长检查"] = "❌ 不合格 - 时长不合理"

        snr_status, snr_value = qc_results["信噪比检查"]
        if snr_status:
            result_entry["信噪比检查"] = f"✅ 合格 - {snr_value:.2f} dB"
        else:
            result_entry["信噪比检查"] = f"❌ 不合格 - 信噪比过低 ({snr_value:.2f} dB)"

        if qc_results["噪声检查"]:
            result_entry["噪声检查"] = "✅ 合格"
        else:
            result_entry["噪声检查"] = "❌ 不合格 - 噪声超标"

        vad_status, speech_count = qc_results["语音活动检测"]
        if vad_status:
            result_entry["语音活动检测"] = f"✅ 合格 - {speech_count} 个语音片段"
        else:
            result_entry["语音活动检测"] = "❌ 不合格 - 语音连贯性差"

        result_entry["动态范围"] = f"{qc_results['动态范围']:.2f} dB"
        result_entry["总谐波失真"] = f"{qc_results['总谐波失真']:.2f}"

        all_results.append(result_entry)

    # 显示质检结果
    st.write("🔎 **质检结果**")
    results_df = pd.DataFrame(all_results)
    st.dataframe(results_df)

    # 允许用户下载报告
    report_csv = results_df.to_csv(index=False, encoding="utf-8-sig").encode("utf-8-sig")
    st.download_button("📥 下载质检报告", report_csv, "audio_quality_report.csv", "text/csv")
# 文本标注板块
if app_mode == "文本标注":
    st.header("📝 文本标注")

    # 质检合格的音频文件
    if not hasattr(st.session_state, 'qc_results') or st.session_state.qc_results is None:
        st.error("请先进行音频质检，然后才能进行文本标注")
    else:
        qc_results = st.session_state.qc_results
        qualified_files = qc_results[qc_results['复检结果'] == '✅ 合格 - 复检通过']

        # 上传 Excel 文件
        uploaded_excel = st.file_uploader("📂 请选择包含文本内容的 Excel 文件", type=["xlsx", "xls"])

        if uploaded_excel:
            # 读取 Excel 数据
            df = pd.read_excel(uploaded_excel)

            # 检查 Excel 是否包含必要的列
            if '文件名' not in df.columns or '文本内容' not in df.columns:
                st.error("Excel 文件格式错误，必须包含 '文件名' 和 '文本内容' 两列")
            else:
                # 文本标注区域
                st.write("📖 **文本标注**")
                annotations = []

                for _, row in qualified_files.iterrows():
                    file_name = row['文件名']
                    file_path = row['播放']

                    # 初始化按钮状态
                    if 'quality_status' not in st.session_state:
                        st.session_state.quality_status = {}

                    # 播放音频
                    st.audio(file_path, format="audio/wav")
                    st.write(f"📄 **文件名**: {file_name}")

                    # 去掉文件扩展名，匹配 Excel 中的文件名
                    file_name_without_extension = os.path.splitext(file_name)[0]
                    matching_row = df[df['文件名'] == file_name_without_extension]
                    original_text = matching_row.iloc[0]['文本内容'] if not matching_row.empty else "无文本"

                    # 显示文本标注输入框
                    col1, col2 = st.columns(2)
                    with col1:
                        st.text_area("原始文本（不可编辑）", original_text, height=100, disabled=True, key=f"original_{file_name}")
                    with col2:
                        modified_text = st.text_area("修改文本（可编辑）", original_text, height=100, key=f"modified_{file_name}")

                    # 合格/不合格按钮
                    col3, col4 = st.columns(2)
                    with col3:
                        if st.button("合格", key=f"pass_{file_name}"):
                            st.session_state.quality_status[file_name] = "合格"
                    with col4:
                        if st.button("不合格", key=f"not_pass_{file_name}"):
                            st.session_state.quality_status[file_name] = "不合格"

                    # 显示按钮状态
                    if file_name in st.session_state.quality_status:
                        status = st.session_state.quality_status[file_name]
                        if status == "合格":
                            st.success(f"已标记为：{status}")
                        else:
                            st.error(f"已标记为：{status}")
                    else:
                        st.warning("未评估")

                    # 记录标注数据
                    quality_status = st.session_state.quality_status.get(file_name, "未评估")
                    annotations.append({
                        "文件名": file_name,
                        "原始文本": original_text,
                        "修改文本": modified_text,
                        "质量状态": quality_status
                    })

                # 允许下载标注结果
                if st.button("📥 下载标注结果"):
                    annotations_df = pd.DataFrame(annotations)
                    buffer = io.BytesIO()
                    with pd.ExcelWriter(buffer, engine='xlsxwriter') as writer:
                        annotations_df.to_excel(writer, index=False)
                    buffer.seek(0)
                    st.download_button(
                        label="📥 下载 Excel",
                        data=buffer,
                        file_name="text_annotations.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                    )