import socket
import os
import asyncio
import edge_tts
from pydub import AudioSegment

# ==========================================
#              ⚡️ 强力修复区域 ⚡️
# ==========================================
# 获取当前脚本所在文件夹
current_dir = os.path.dirname(os.path.abspath(__file__))

# 1. 检查 ffmpeg 和 ffprobe 是否存在
ffmpeg_path = os.path.join(current_dir, "ffmpeg.exe")
ffprobe_path = os.path.join(current_dir, "ffprobe.exe")

print(f"当前工作目录: {current_dir}")
if os.path.exists(ffmpeg_path) and os.path.exists(ffprobe_path):
    print("✅ 检测到 ffmpeg.exe 和 ffprobe.exe，环境正常！")
    # 2. 关键操作：将当前目录临时加入环境变量 PATH
    # 这样 pydub 就能自动找到这两个文件了，不需要额外配置 converter
    os.environ["PATH"] += os.pathsep + current_dir
else:
    print("❌ 错误：请确保 ffmpeg.exe 和 ffprobe.exe 都在这个文件夹里！")
    # 如果没找到，打印出来提示用户
    if not os.path.exists(ffmpeg_path): print(f"   缺少文件: {ffmpeg_path}")
    if not os.path.exists(ffprobe_path): print(f"   缺少文件: {ffprobe_path}")

# ==========================================

# 网络配置
UDP_IP = "0.0.0.0"
UDP_PORT = 12345
VOICE = "zh-CN-XiaoxiaoNeural"  # 晓晓的声音

sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock.bind((UDP_IP, UDP_PORT))

print(f"🚀 服务器已启动，监听端口 {UDP_PORT}...")


async def run_tts(text, output_file):
    communicate = edge_tts.Communicate(text, VOICE)
    await communicate.save(output_file)


def process_audio(pcm_data):
    print("1. [收到] 正在保存音频 input.pcm...")
    with open("../input.pcm", "wb") as f:
        f.write(pcm_data)

    print("2. [思考] (模拟) 调用STT和LLM...")
    reply_text = "你好呀！金来嘎嘎！我是你的茁猫，是你的老婆猫哦~"
    print(f"   --> 准备回复: {reply_text}")

    print("3. [合成] 生成 Edge-TTS 语音...")
    try:
        asyncio.run(run_tts(reply_text, "../reply.mp3"))
        print("   --> reply.mp3 生成成功")
    except Exception as e:
        print(f"❌ TTS生成失败: {e}")
        return bytearray()

    print("4. [转码] 转换格式为 PCM (16k, 16bit, Mono)...")
    try:
        # pydub 加载 mp3
        sound = AudioSegment.from_mp3("../reply.mp3")
        # 转换参数
        sound = sound.set_frame_rate(16000).set_channels(1).set_sample_width(2)
        raw_audio = sound.raw_data
        print(f"   --> 转码成功！数据大小: {len(raw_audio)} 字节")
        return raw_audio
    except Exception as e:
        print(f"❌ 格式转换失败: {e}")
        print("   (提示：请确认 ffprobe.exe 也复制到了文件夹里)")
        return bytearray()


# 主循环
audio_data = bytearray()
is_receiving = False

while True:
    try:
        data, addr = sock.recvfrom(1024)

        if data.startswith(b"START"):
            print("\n🔵 开始接收 ESP32 语音...")
            audio_data = bytearray()
            is_receiving = True
        elif data.startswith(b"END"):
            print("🔴 语音接收完毕，开始处理...")
            is_receiving = False

            reply_pcm = process_audio(audio_data)

            if len(reply_pcm) > 0:
                print(f"5. [发送] 发送回 ESP32 ({len(reply_pcm)} bytes)...")
                chunk_size = 1024
                # 稍微加一点延时，防止UDP发太快 ESP32 接收不过来
                import time

                for i in range(0, len(reply_pcm), chunk_size):
                    chunk = reply_pcm[i:i + chunk_size]
                    sock.sendto(chunk, addr)
                    time.sleep(0.03)  # 微小的延时
                print("✅ 发送完成！等待下一轮...")
            else:
                print("⚠️ 音频为空，跳过发送。")

        elif is_receiving:
            audio_data.extend(data)

    except Exception as e:
        print(f"主循环错误: {e}")
