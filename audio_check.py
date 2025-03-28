import librosa
import numpy as np
import os
import streamlit as st
import tempfile
import soundfile as sf
import wave
import struct
from webrtcvad import Vad
import pandas as pd

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


    # 以下为检查函数（保持不变）

    def check_audio_format(file_path):
        """检查音频格式是否为 MP3 或 WAV"""
        valid_formats = ['.mp3', '.wav']
        file_extension = os.path.splitext(file_path)[1].lower()
        return file_extension in valid_formats


    def check_audio_duration(file_path, min_length=1.0, max_length=10.0):
        """检查音频时长是否合理"""
        y, sr = librosa.load(file_path, sr=None)
        duration = librosa.get_duration(y=y, sr=sr)
        return min_length <= duration <= max_length


    def check_snr(file_path, threshold=15):
        """检查信噪比是否满足要求"""
        y, sr = librosa.load(file_path, sr=None)
        signal_power = np.mean(y ** 2)
        noise_sample = y[:int(0.5 * sr)]
        noise_power = np.mean(noise_sample ** 2)

        if noise_power == 0:
            snr = float("inf")
        else:
            snr = 10 * np.log10(signal_power / noise_power)

        return snr >= threshold, snr  # 返回布尔值和信噪比值


    def load_audio(file_path):
        """加载音频并转换为16-bit PCM格式"""
        y, sr = librosa.load(file_path, sr=16000)  # 设置采样率为16kHz
        if len(y.shape) > 1:  # 如果是立体声，转换为单声道
            y = y.mean(axis=1)
        # 将音频转换为16-bit PCM
        y = np.int16(y / np.max(np.abs(y)) * 32767)  # 将浮动数据转换为16-bit整数
        # 检查音频数据是否为空
        if len(y) == 0:
            raise ValueError("音频数据为空")
        return y, sr


    def write_wave(file_path, audio_data, sample_rate):
        """保存音频为16-bit PCM WAV格式"""
        with wave.open(file_path, 'wb') as wf:
            wf.setnchannels(1)  # 单声道
            wf.setsampwidth(2)  # 16-bit
            wf.setframerate(sample_rate)
            wf.writeframes(audio_data)


    def vad_check(file_path):
        """语音活动检测（VAD）检查音频的连贯性和完整性"""
        vad = Vad(3)  # 设置 VAD 模式（0=低灵敏度，3=高灵敏度）

        # 加载音频并转换为16-bit PCM
        y, sr = load_audio(file_path)

        frame_duration = 30  # 每帧的时长（毫秒）
        frame_length = int(sr * frame_duration / 1000)  # 帧的长度（样本数）

        segments = []
        for start in range(0, len(y), frame_length):
            end = min(start + frame_length, len(y))
            frame = y[start:end]

            # 检查帧长度是否符合要求
            if len(frame) != frame_length:
                continue

            # 将数据转换为字节串
            frame_bytes = struct.pack("<" + "h" * len(frame), *frame)

            if vad.is_speech(frame_bytes, sr):
                segments.append((start, end))

        return segments  # 返回检测到的语音段落


    def noise_check(file_path):
        """简单的噪声检测，检查音频是否超过标准噪声阈值"""
        y, sr = librosa.load(file_path, sr=None)
        noise_sample = y[:int(0.5 * sr)]
        noise_power = np.mean(noise_sample ** 2)
        noise_threshold = 0.01  # 设定噪声阈值
        return noise_power <= noise_threshold


    def initial_check(file_path):
        """初检阶段：检查音频格式、时长、信噪比"""
        if not check_audio_format(file_path):
            return "❌ 不合格 - 格式错误", None

        if not check_audio_duration(file_path):
            return "❌ 不合格 - 时长不合理", None

        snr_status, snr_value = check_snr(file_path)
        if not snr_status:
            return f"❌ 不合格 - 信噪比过低（{snr_value:.2f} dB）", snr_value

        return "✅ 合格 - 初检通过", snr_value


    def review_check(file_path):
        """复检阶段：语音活动检测和噪声检测"""
        # 语音活动检测（VAD）
        segments = vad_check(file_path)
        if len(segments) < 2:
            return "❌ 不合格 - 语音连贯性差"

        # 噪声检测
        if not noise_check(file_path):
            return "❌ 不合格 - 噪声超标"

        return "✅ 合格 - 复检通过"


    # 多轮质检处理
    def multi_round_check(uploaded_files):
        all_results = []  # 用于存储所有质检结果
        for uploaded_file in uploaded_files:
            # 根据质检方式处理文件
            if check_type == "📂 批量检查文件夹":
                file_path = uploaded_file
            else:
                # 对于上传的文件，保存到临时文件
                with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_file:
                    temp_file.write(uploaded_file.read())
                    file_path = temp_file.name

            # 初检
            initial_status, snr_value = initial_check(file_path)
            # 复检
            if initial_status.startswith("✅"):
                review_status = review_check(file_path)
            else:
                review_status = "未进行复检"

            all_results.append({
                "文件名": os.path.basename(file_path),
                "初检结果": initial_status,
                "复检结果": review_status,
                "信噪比 (dB)": snr_value if snr_value is not None else "N/A",
                "播放": file_path  # 添加播放音频的路径
            })

        return all_results


    # 执行质检
    if uploaded_files:
        final_results = multi_round_check(uploaded_files)

        # 显示质检结果
        st.write("🔎 **质检结果**")
        results_df = pd.DataFrame(final_results)

        # 显示表格，添加播放按钮
        st.dataframe(results_df.drop(columns=["播放"]))  # 不显示播放列

        # 允许用户下载报告
        report_csv = results_df.drop(columns=["播放"]).to_csv(index=False, encoding="utf-8-sig").encode("utf-8-sig")
        st.download_button("📥 下载质检报告", report_csv, "audio_quality_report.csv", "text/csv")

        # 添加播放音频的按钮
        st.write("🎧 **播放音频**")
        for _, row in results_df.iterrows():
            file_path = row["播放"]
            st.audio(file_path, format="audio/wav")
            st.write(f"文件名: {row['文件名']}")

        # 保存质检结果以便在文本标注板块使用
        session_state = st.session_state
        session_state.qc_results = results_df


# 文本标注板块
if app_mode == "文本标注":
    st.header("📝 文本标注")

    # 质检合格的音频文件
    if not hasattr(st.session_state, 'qc_results'):
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
                    annotations_excel_path = "text_annotations.xlsx"
                    annotations_df.to_excel(annotations_excel_path, index=False)
                    with open(annotations_excel_path, "rb") as f:
                        st.download_button("📥 下载 Excel", f, file_name="text_annotations.xlsx",
                                           mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")