import socket
import os
import re
import asyncio
import dashscope
from dashscope import Generation
from dashscope.audio.asr import Recognition
# ▼▼▼ 关键修改：引入 v2 版本的 TTS ▼▼▼
from dashscope.audio.tts_v2 import SpeechSynthesizer, AudioFormat
from pydub import AudioSegment
from http import HTTPStatus

# ==========================================
#              1. 全局配置区
# ==========================================
dashscope.api_key = "sk-26ac2fd867404de4846819e019ebfd92"
MY_VOCAB_ID = "vocab-zhuomao-c0dcc1be55d7400888dada26fa09c4fe"
MODEL_NAME = "qwen-turbo"

# ▼▼▼ CosyVoice 配置 (严格按照文档) ▼▼▼
TTS_MODEL = "cosyvoice-v3-plus"  # 文档推荐的高速模型
TTS_VOICE = "cosyvoice-v3-plus-myvoice-10c58587c1284f869aee34cc719396cf"  # 龙小淳 (适合猫娘)，若要男声改 "longanyang"

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

def generate_cosyvoice_tts(text_content):
    """
    使用 CosyVoice v2 接口生成语音 (修正参数版)
    """
    print(f"   [TTS] CosyVoice 正在合成: {text_content[:10]}...")
    try:
        # ▼▼▼ 修正点：使用 AudioFormat 枚举指定格式和采样率 ▼▼▼
        synthesizer = SpeechSynthesizer(
            model=TTS_MODEL,
            voice=TTS_VOICE,
            # 直接用这个枚举，既指定了 wav，又指定了 16000Hz
            format=AudioFormat.WAV_44100HZ_MONO_16BIT,
            # instruction='你说话的角色是傲娇公主，你说话的情感是happy。'
        )

        # 调用 call 方法
        audio = synthesizer.call(text_content)

        if audio:
            print(f"   ✅ 合成成功 ({len(audio)} bytes)")
            return audio
        else:
            print("❌ 合成返回空数据")
            return None

    except Exception as e:
        print(f"❌ CosyVoice 报错: {e}")
        # 如果还是报错，打印一下支持的 AudioFormat 列表帮助调试
        # print(dir(AudioFormat))
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
    print(f"   [发送] 回传音频 ({len(pcm_data)} bytes)...")
    chunk_size = 1024
    import time
    for i in range(0, len(pcm_data), chunk_size):
        chunk = pcm_data[i:i + chunk_size]
        sock.sendto(chunk, addr)
        time.sleep(0.012)
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
            model='paraformer-realtime-v2', format='wav', sample_rate=44100,
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
    # 解决 Windows 那个烦人的报错
    if os.name == 'nt':
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind((UDP_IP, UDP_PORT))

    print(f"🚀 茁猫(CosyVoice官方版)已启动! 端口: {UDP_PORT}")
    print(f"🔊 音色: {TTS_VOICE} | 模型: {TTS_MODEL}")

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
                    AudioSegment(data=audio_data, sample_width=2, frame_rate=44100, channels=1).export("temp_input.wav",
                                                                                                       format="wav")

                    text_in = run_aliyun_asr("temp_input.wav")

                    if text_in:
                        print(f"👂 听见: {text_in}")
                        full_reply = call_qwen_llm(text_in)
                        mood, talk_text = parse_mood_and_text(full_reply)

                        print("-" * 40)
                        print(f"🐱 茁猫: {talk_text} | 💖 {mood}")
                        print("-" * 40)

                        # CosyVoice 生成
                        wav_bytes = generate_cosyvoice_tts(talk_text)

                        if wav_bytes:
                            with open("temp_reply.wav", "wb") as f:
                                f.write(wav_bytes)

                            # pydub 处理
                            sound = AudioSegment.from_wav("temp_reply.wav")
                            # sound = sound + 10  # 增加音量
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
