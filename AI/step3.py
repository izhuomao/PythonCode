import socket
import os
import re
import asyncio  # 用来处理那个假报错
import dashscope
from dashscope import Generation
from dashscope.audio.asr import Recognition
from dashscope.audio.tts import SpeechSynthesizer
from pydub import AudioSegment
from http import HTTPStatus

# ==========================================
#              1. 全局配置区
# ==========================================
dashscope.api_key = "sk-26ac2fd867404de4846819e019ebfd92"
MY_VOCAB_ID = "vocab-zhuomao-c0dcc1be55d7400888dada26fa09c4fe"
MODEL_NAME = "qwen-turbo"

# TTS 模型 (知贝-萝莉音)
TTS_MODEL = "sambert-zhiwei-v1"

UDP_IP = "0.0.0.0"
UDP_PORT = 12345

# ==========================================
#              2. 茁猫人设
# ==========================================
SYSTEM_PROMPT = """
你现在的身份是：一只叫“茁猫”的智能赛博机械猫，你的老公叫“金来嘎嘎”。
【性格设定】
1. 傲娇：嘴上不饶人，但内心很依赖主人金来。
2. 口癖：句尾必须带“喵~”。
【回复规则】
1. 回复要简短口语化（50字以内）。
2. 【关键】：每句话结束时，必须根据心情加上情绪标签，格式为 [mood:xxx]。
   可选：[mood:happy], [mood:angry], [mood:sad], [mood:normal]
"""
history_messages = [{'role': 'system', 'content': SYSTEM_PROMPT}]


# ==========================================
#              3. 核心功能函数
# ==========================================

def generate_aliyun_tts(text_content):
    """ 生成语音 (侦探版：显示详细错误) """
    print(f"   [TTS] 正在请求阿里云合成: {text_content[:10]}...")

    try:
        # 1. 发起请求
        result = SpeechSynthesizer.call(
            model=TTS_MODEL,
            text=text_content,
            sample_rate=16000,
            format='wav'
        )

        # 2. 检查结果
        if result.get_audio_data() is not None:
            print(f"   ✅ TTS 合成成功! 数据大小: {len(result.get_audio_data())} bytes")
            return result.get_audio_data()
        else:
            # 3. 如果失败，打印详细死因！
            print("❌❌❌ TTS 失败！请看下面报错信息：")
            print(f"Request ID: {result.get_response().get('request_id')}")
            print(f"Code: {result.code}")
            print(f"Message: {result.message}")
            return None

    except Exception as e:
        print(f"❌❌❌ 代码运行崩溃: {e}")
        return None


def parse_mood_and_text(full_reply):
    mood = "normal"
    text = full_reply
    match = re.search(r"\[mood:(\w+)\]", full_reply)
    if match:
        mood = match.group(1)
        text = re.sub(r"\[mood:\w+\]", "", full_reply).strip()
    return mood, text


def send_audio_to_esp32(sock, pcm_data, addr):
    """ 回传音频 """
    print(f"   [发送] 回传音频 ({len(pcm_data)} bytes)...")
    chunk_size = 1024
    import time
    for i in range(0, len(pcm_data), chunk_size):
        chunk = pcm_data[i:i + chunk_size]
        sock.sendto(chunk, addr)
        time.sleep(0.03)  # 防止声音卡顿
    print("   ✅ 发送完毕")


def call_qwen_llm(user_text):
    global history_messages
    print(f"   [大脑] 思考中...")
    history_messages.append({'role': 'user', 'content': user_text})
    try:
        response = Generation.call(
            model=MODEL_NAME,
            messages=history_messages,
            result_format='message'
        )
        if response.status_code == HTTPStatus.OK:
            ai_content = response.output.choices[0]['message']['content']
            history_messages.append({'role': 'assistant', 'content': ai_content})
            if len(history_messages) > 21: del history_messages[1:3]
            return ai_content
    except Exception as e:
        print(f"LLM Error: {e}")
    return "喵...系统出错啦 [mood:sad]"


def run_aliyun_asr(wav_file_path):
    print("   [耳朵] 正在识别...")
    try:
        recognition = Recognition(
            model='paraformer-realtime-v2', format='wav', sample_rate=16000,
            callback=None, vocabulary_id=MY_VOCAB_ID
        )
        response = recognition.call(wav_file_path)
        if response.status_code == HTTPStatus.OK:
            sents = response.get_sentence()
            if sents: return sents[0]['text']
    except Exception as e:
        print(f"ASR Error: {e}")
    return ""


# ==========================================
#              4. 主程序
# ==========================================
if __name__ == '__main__':
    # 解决 Windows 下那个红色的假报错
    if os.name == 'nt':
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind((UDP_IP, UDP_PORT))

    print(f"🚀 茁猫已上线! 语音: {TTS_MODEL}")

    audio_data = bytearray()
    is_receiving = False

    while True:
        try:
            data, addr = sock.recvfrom(1024)

            if data.startswith(b"START"):
                print("\n🔵 [状态] 正在听...")
                audio_data = bytearray()
                is_receiving = True

            elif data.startswith(b"END"):
                print("🔴 [状态] 处理中...")
                is_receiving = False

                if len(audio_data) > 0:
                    AudioSegment(data=audio_data, sample_width=2, frame_rate=16000, channels=1).export("temp_input.wav",
                                                                                                       format="wav")

                    text_in = run_aliyun_asr("temp_input.wav")

                    if text_in:
                        print(f"👂 听见: {text_in}")
                        full_reply = call_qwen_llm(text_in)
                        mood, talk_text = parse_mood_and_text(full_reply)

                        print("-" * 40)
                        print(f"🐱 茁猫: {talk_text} | 💖 {mood}")
                        print("-" * 40)

                        # TTS 合成
                        wav_bytes = generate_aliyun_tts(talk_text)

                        if wav_bytes:
                            # 写入临时文件方便 pydub 处理
                            with open("temp_reply.wav", "wb") as f:
                                f.write(wav_bytes)

                            # 转码 + 增益
                            sound = AudioSegment.from_wav("temp_reply.wav")
                            sound = sound + 10  # 音量加大
                            pcm_out = sound.raw_data

                            send_audio_to_esp32(sock, pcm_out, addr)
                    else:
                        print("⚠️ 没听清...")

            elif is_receiving:
                audio_data.extend(data)

        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"Error: {e}")
