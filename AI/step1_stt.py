import socket
import os
import dashscope
from dashscope.audio.asr import Recognition
from pydub import AudioSegment
from http import HTTPStatus

# ==========================================
#              用户配置区域
# ==========================================
# 1. 配置你的阿里云 API Key
# (注意：你刚才把Key贴出来了，为了安全，建议去阿里云后台删掉旧的重新生成一个，或者小心保管)
dashscope.api_key = "sk-26ac2fd867404de4846819e019ebfd92"

# 2. 网络端口 (保持不变)
UDP_IP = "0.0.0.0"
UDP_PORT = 12345


# ==========================================
#              核心功能函数
# ==========================================
MY_VOCAB_ID = "vocab-zhuomao-c0dcc1be55d7400888dada26fa09c4fe"
def run_aliyun_asr(wav_file_path):
    """
    调用阿里云 Paraformer 模型进行语音识别
    """
    print("   [阿里云] 正在上传音频并识别...")

    recognition = Recognition(
        model='paraformer-realtime-v2',  # 使用你测试成功的模型
        format='wav',
        sample_rate=16000,
        language_hints=['zh', 'en'],  # 提示支持中英文
        callback=None,
        vocabulary_id = MY_VOCAB_ID
    )

    # 这里的 call 是阻塞的，会等待识别完成
    response = recognition.call(wav_file_path)

    if response.status_code == HTTPStatus.OK:
        # --- 提取文字的核心逻辑 ---
        # 阿里云返回的是一个复杂的对象，我们需要剥洋葱
        sentences = response.get_sentence()
        if sentences and len(sentences) > 0:
            text = sentences[0]['text']
            return text
        else:
            return ""  # 没听到说话
    else:
        print(f"❌ 识别报错: {response.message}")
        return None


# ==========================================
#              主程序循环
# ==========================================

sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock.bind((UDP_IP, UDP_PORT))

print(f"🚀 茁猫的大脑(PC端)已启动，监听端口 {UDP_PORT}...")
print("waiting for ESP32...")

audio_data = bytearray()
is_receiving = False

while True:
    try:
        # 接收数据
        data, addr = sock.recvfrom(1024)

        # 1. 收到开始信号
        if data.startswith(b"START"):
            print("\n🔵 [状态] 正在录音...")
            audio_data = bytearray()
            is_receiving = True

        # 2. 收到结束信号 -> 开始处理
        elif data.startswith(b"END"):
            print("🔴 [状态] 录音结束，处理中...")
            is_receiving = False

            # --- A. 保存并转码 ---
            # 利用 pydub 将裸流 PCM 包装成 WAV (阿里云要求带文件头)
            if len(audio_data) > 0:
                raw_audio = AudioSegment(
                    data=audio_data,
                    sample_width=2,  # 16bit
                    frame_rate=16000,  # 16kHz
                    channels=1  # Mono
                )
                # (可选) 稍微放大一点声音，提高识别率
                raw_audio = raw_audio + 10

                raw_audio.export("temp_input.wav", format="wav")

                # --- B. 调用阿里云 ---
                text_result = run_aliyun_asr("temp_input.wav")

                # --- C. 输出结果 ---
                if text_result:
                    print("=" * 40)
                    print(f"🎤 听到用户说: {text_result}")
                    print("=" * 40)

                    # 💡 下一步：在这里把 text_result 传给大模型 (LLM)

                else:
                    print("⚠️ 似乎没听到声音 (或识别结果为空)")
            else:
                print("⚠️ 接收到的音频数据长度为0")

        # 3. 正在录音中 -> 拼接数据
        elif is_receiving:
            audio_data.extend(data)

    except KeyboardInterrupt:
        print("停止运行")
        break
    except Exception as e:
        print(f"发生错误: {e}")
