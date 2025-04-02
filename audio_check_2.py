import librosa
import numpy as np
import os
import streamlit as st
import tempfile
import pandas as pd
import wave
import struct
from webrtcvad import Vad

# 初始化会话状态
if 'qc_results' not in st.session_state:
    st.session_state.qc_results = None
if 'quality_status' not in st.session_state:
    st.session_state.quality_status = {}

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

    # 噪声段参数
    noise_start = st.number_input("噪声段起始位置（样本数）", min_value=0, value=0)
    noise_end = st.number_input("噪声段结束位置（样本数）", min_value=1, value=1000)


    # 检查函数
    def check_audio_format(file_path):
        """检查音频格式是否为 MP3 或 WAV"""
        valid_formats = ['.mp3', '.wav']
        file_extension = os.path.splitext(file_path)[1].lower()
        return file_extension in valid_formats


    def check_audio_duration(file_path, min_length=1.0, max_length=15.0):
        """检查音频时长是否合理"""
        y, sr = librosa.load(file_path, sr=None)
        duration = librosa.get_duration(y=y, sr=sr)
        return min_length <= duration <= max_length, duration


    def calculate_snr_with_ref(audio_path, noise_start=0, noise_end=1000):
        """通过预留的静音段计算SNR"""
        y, sr = librosa.load(audio_path, sr=None)
        if noise_end > len(y):
            return None, "噪声段超出音频长度"

        noise = y[noise_start:noise_end]  # 提取噪声段
        signal = y[noise_end:]  # 提取有效语音段

        power_signal = np.mean(signal ** 2)  # 信号功率
        power_noise = np.mean(noise ** 2)  # 噪声功率

        if power_noise == 0:
            return float('inf'), "信噪比无穷大（无噪声）"

        snr = 10 * np.log10(power_signal / power_noise)  # 计算SNR
        return snr, None  # 返回SNR值和错误信息


    def check_snr(file_path, noise_start=0, noise_end=1000, threshold=15):
        """检查信噪比是否满足要求"""
        snr, error = calculate_snr_with_ref(file_path, noise_start, noise_end)

        if error:
            return False, error

        if snr >= threshold:
            return True, snr
        else:
            return False, snr


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

        return len(segments) > 0  # 返回是否有语音段落


    def noise_check(file_path):
        """简单的噪声检测，检查音频是否超过标准噪声阈值"""
        y, sr = librosa.load(file_path, sr=None)
        noise_sample = y[:int(0.5 * sr)]
        noise_power = np.mean(noise_sample ** 2)
        noise_threshold = 0.01  # 设定噪声阈值
        return noise_power <= noise_threshold


    # 一轮质检处理
    def single_round_check(uploaded_files, noise_start=0, noise_end=1000):
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

            # 格式检查
            format_status = "✅ 合格" if check_audio_format(file_path) else "❌ 不合格"

            # 时长检查
            duration_status, duration_value = check_audio_duration(file_path)
            duration_status = "✅ 合格" if duration_status else "❌ 不合格"

            # 信噪比检查（基于预留噪声段）
            snr_status, snr_value = check_snr(file_path, noise_start, noise_end, snr_threshold)
            if isinstance(snr_value, str):
                snr_status = "❌ 不合格"
            else:
                snr_status = "✅ 合格" if snr_status else "❌ 不合格"

            # 语音活动检测
            vad_status = "✅ 合格" if vad_check(file_path) else "❌ 不合格"

            # 噪声检测
            noise_status = "✅ 合格" if noise_check(file_path) else "❌ 不合格"

            all_results.append({
                "文件名": os.path.basename(file_path),
                "格式检查": format_status,
                "时长检查": f"{duration_status} ({duration_value:.2f}秒)",
                "信噪比检查": f"{snr_status} ({snr_value} dB)" if isinstance(snr_value, (
                int, float)) else f"{snr_status} ({snr_value})",
                "语音活动检测": vad_status,
                "噪声检测": noise_status,
                "播放": file_path  # 添加播放音频的路径
            })

        return all_results


    # 执行质检
    if uploaded_files:
        final_results = single_round_check(uploaded_files, noise_start, noise_end)

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
        qualified_files = qc_results[qc_results['格式检查'] == '✅ 合格']

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
                        st.text_area("原始文本（不可编辑）", original_text, height=100, disabled=True,
                                     key=f"original_{file_name}")
                    with col2:
                        modified_text = st.text_area("修改文本（可编辑）", original_text, height=100,
                                                     key=f"modified_{file_name}")

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