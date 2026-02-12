import socket
import os
import asyncio
import edge_tts
from pydub import AudioSegment
import dashscope
from dashscope.audio.asr import Recognition

# ================= 配置区域 =================
# 1. 务必填入正确的 SK-开头的 Key
dashscope.api_key = "sk-26ac2fd867404de4846819e019ebfd92"

# 2. 网络配置
UDP_IP = "0.0.0.0"
UDP_PORT = 12345
TTS_VOICE = "zh-CN-XiaoxiaoNeural"
# ============================================

# 检查 ffmpeg
current_dir = os.path.dirname(os.path.abspath(__file__))
os.environ["PATH"] += os.pathsep + current_dir

# 初始化 UDP
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock.bind((UDP_IP, UDP_PORT))

# 消息历史
history_messages = [
    {'role': 'system', 'content': '你是一只叫“茁猫”的电子宠物。说话可爱一点，简短一点，句尾带“喵”。'}
]


async def generate_tts(text, output_file):
    communicate = edge_tts.Communicate(text, TTS_VOICE)
    await communicate.save(output_file)


# ==========================================
#      👇 请用这段新代码替换原来的 process_audio_cloud 函数
# ==========================================

def process_audio_cloud(pcm_data):
    """全云端处理：SenseVoice -> Qwen -> Edge-TTS"""
    print("\n👂 [1/3] 正在上传语音至阿里云识别...")

    # 1. 预处理音频 (PCM -> WAV)
    temp_wav = "temp_input.wav"
    try:
        # 确保 pydub 导出的是标准的 WAV
        audio = AudioSegment(data=pcm_data, sample_width=2, frame_rate=16000, channels=1)
        audio.export(temp_wav, format="wav")
        # 检查一下文件是否存在
        if not os.path.exists(temp_wav):
            print("❌ WAV文件生成失败")
            return bytearray()
    except Exception as e:
        print(f"❌ 音频转码失败: {e}")
        return bytearray()

    # 2. 调用阿里云语音识别 (SenseVoice-v1)
    # 修正点：使用 sensevoice-v1 模型，这是专门用于识别录音文件的，非常稳定
    user_text = ""
    try:
        # 这里的 file 参数必须是完整路径 string
        file_path = os.path.abspath(temp_wav)

        rec_result = Recognition.call(
            model='sensevoice-v1',
            file=f'file://{file_path}',  # 关键修正：加上 file:// 前缀更保险
            format='wav',
        )

        # 3. 解析结果 (SenseVoice 的返回结构)
        if rec_result.status_code == 200:
            # SenseVoice 通常把结果放在 output.text 或 output.choices 中
            # 我们先打印一下原始结果方便调试，万一出错你看得见
            # print(f"DEBUG: {rec_result}")

            if hasattr(rec_result.output, 'text'):
                user_text = rec_result.output.text
            elif hasattr(rec_result.output, 'choices') and len(rec_result.output.choices) > 0:
                user_text = rec_result.output.choices[0]['message']['content']  # 这是一个容错写法
            else:
                # 尝试直接转字符串解析
                import json
                user_text = json.dumps(rec_result.output, ensure_ascii=False)

            # 清理可能的干扰字符
            import re
            # 有时候SenseVoice会返回 XML 标签，我们需要去掉
            user_text = re.sub(r'<[^>]+>', '', str(user_text)).strip()

            print(f"   >>> 识别结果: {user_text}")
        else:
            print(f"❌ 识别API报错: {rec_result.message}")
            return bytearray()

    except Exception as e:
        print(f"❌ 云端STT严重错误: {e}")
        return bytearray()

    if not user_text:
        print("⚠️ 未识别到有效文字")
        return bytearray()

    # 4. 调用大模型 (LLM)
    print("🧠 [2/3] 正在思考...")
    # 限制记忆长度，防止 token 爆炸
    if len(history_messages) > 10:
        history_messages.pop(1)  # 删掉最早的一条用户记录

    history_messages.append({'role': 'user', 'content': user_text})

    reply_text = "喵？"  # 默认回复
    try:
        response = dashscope.Generation.call(
            dashscope.Generation.Models.qwen_turbo,
            messages=history_messages,
            result_format='message',
        )

        if response.status_code == 200:
            reply_text = response.output.choices[0]['message']['content']
            print(f"   >>> 茁猫回: {reply_text}")
            history_messages.append({'role': 'assistant', 'content': reply_text})
        else:
            print(f"LLM错误: {response.code}")
            reply_text = "脑子宕机了喵..."
    except Exception as e:
        print(f"LLM网络错误: {e}")
        reply_text = "网络不好喵..."

    # 5. 生成语音 (TTS)
    print("🗣️ [3/3] 生成语音...")
    try:
        asyncio.run(generate_tts(reply_text, "../reply.mp3"))
        sound = AudioSegment.from_mp3("../reply.mp3")
        sound = sound.set_frame_rate(16000).set_channels(1).set_sample_width(2)
        sound = sound + 6  # 增加一点音量
        return sound.raw_data + (b'\x00' * 2000)  # 加一点静音尾巴防止被切断
    except Exception as e:
        print(f"❌ TTS错误: {e}")
        return bytearray()


# --- 主循环 (和之前一样) ---
print(f"🚀 全云端模式已启动！监听端口 {UDP_PORT}")
recv_buffer = bytearray()
is_recording = False

while True:
    try:
        data, addr = sock.recvfrom(1024)
        if data.startswith(b"START"):
            recv_buffer = bytearray()
            is_recording = True
        elif data.startswith(b"END"):
            is_recording = False
            if len(recv_buffer) > 0:
                reply = process_audio_cloud(recv_buffer)  # 使用新的云端处理函数
                if len(reply) > 0:
                    import time

                    for i in range(0, len(reply), 1024):
                        sock.sendto(reply[i:i + 1024], addr)
                        time.sleep(0.005)
        elif is_recording:
            recv_buffer.extend(data)
    except KeyboardInterrupt:
        break
    except Exception as e:
        print(e)
