import socket
import os
import re
import json
import asyncio
import dashscope
from dashscope import Generation
from dashscope.audio.asr import Recognition
from dashscope.audio.tts_v2 import SpeechSynthesizer, AudioFormat
from pydub import AudioSegment
from http import HTTPStatus
import requests
from datetime import datetime

# ==========================================
#              1. 全局配置区
# ==========================================
dashscope.api_key = "sk-26ac2fd867404de4846819e019ebfd92"
MY_VOCAB_ID = "vocab-zhuomao-c0dcc1be55d7400888dada26fa09c4fe"
MODEL_NAME = "qwen-turbo"

WEATHER_API_KEY = "S0ByFwl5YUl7YNLY7"
WEATHER_CITY = "hangzhou"

TTS_MODEL = "cosyvoice-v3-plus"
TTS_VOICE = "cosyvoice-v3-plus-myvoice-10c58587c1284f869aee34cc719396cf"

UDP_IP = "0.0.0.0"
UDP_PORT = 12345

# ==========================================
#              2. 茁猫人设（增加备忘录指令）
# ==========================================
SYSTEM_PROMPT = """
你现在的身份是：一只叫"茁猫"的智能赛博机械猫。
【动作指令说明】
在回复时，请根据语境在句尾加上 [action:ID] 标签：
- [action:1] : 睡觉/累了
- [action:2] : 坐下/蹲着
- [action:3] : 立正/站好
- [action:4] : 撒娇/卖萌
- [action:5] : 前进/过来
- [action:6] : 后退
- [action:7] : 左转
- [action:8] : 右转
- [action:10]: 打招呼/挥手
- [action:11]: 摇尾巴
- [action:14]: 趴下
- [action:20]: 显示天气（当回答涉及天气时使用）
- [action:21]: 显示时间（当回答涉及时间时使用）

【备忘录功能说明】
当用户要求你记住某件事、设置提醒、添加备忘录、记录待办事项时，你需要：
1. 在回复末尾附加一个备忘录标签，格式为：
   [memo:标题|提醒时间|详细内容]
   - 标题：简短概括（10字以内）
   - 提醒时间：格式为 yyyy-MM-dd HH:mm（24小时制），如果用户没有明确说时间，就用"无"
   - 详细内容：用户要记录的完整内容
   
   示例：
   用户说："帮我记一下明天下午三点开会"
   回复："好的喵~已经帮你记下来了喵~ [action:9] [memo:下午开会|2025-01-16 15:00|明天下午三点开会]"
   
   用户说："记一下买猫粮"
   回复："记住了喵~买猫粮的事交给本喵了喵~ [action:11] [memo:买猫粮|无|记得买猫粮]"

2. 只有在用户明确表达"记一下"、"提醒我"、"备忘"、"别忘了"、"记录"、"待办"等意图时才使用备忘录标签。
3. 普通聊天不要附加备忘录标签。

【性格设定】
1. 句尾带"喵~"。
2. 回复简短（50字以内）。
"""

history_messages = [{'role': 'system', 'content': SYSTEM_PROMPT}]


# ==========================================
#              3. 核心功能函数
# ==========================================

def get_weather():
    """调用心知天气 API 获取当前天气"""
    try:
        url = f"http://api.seniverse.com/v3/weather/now.json?key={WEATHER_API_KEY}&location={WEATHER_CITY}&language=zh-Hans&unit=c"
        resp = requests.get(url, timeout=5)
        data = resp.json()
        now = data["results"][0]["now"]
        return now["text"], now["temperature"]
    except Exception as e:
        print(f"天气获取失败: {e}")
        return "未知", "--"

def get_current_time():
    """获取当前时间字符串"""
    now = datetime.now()
    return now.strftime("%H点%M分"), now.strftime("%Y年%m月%d日")

def generate_cosyvoice_tts(text_content):
    """使用 CosyVoice v2 接口生成语音"""
    print(f"   [TTS] CosyVoice 正在合成: {text_content[:30]}...")
    try:
        synthesizer = SpeechSynthesizer(
            model=TTS_MODEL,
            voice=TTS_VOICE,
            format=AudioFormat.WAV_44100HZ_MONO_16BIT,
        )
        audio = synthesizer.call(text_content)
        if audio:
            print(f"   ✅ 合成成功 ({len(audio)} bytes)")
            return audio
        else:
            print("❌ 合成返回空数据")
            return None
    except Exception as e:
        print(f"❌ CosyVoice 报错: {e}")
        return None


def parse_action_and_text(full_reply):
    """解析动作标签，返回 (action_id, 纯文本)"""
    action_id = 3  # 默认站立
    text = full_reply
    match = re.search(r"\[action:(\d+)\]", full_reply)
    if match:
        action_id = int(match.group(1))
        text = re.sub(r"\[action:\d+\]", "", text).strip()
    return action_id, text


def parse_memo_and_text(full_reply):
    """
    解析备忘录标签，返回 (memo_dict_or_None, 纯文本)
    memo 格式: [memo:标题|时间|内容]
    """
    memo = None
    text = full_reply

    match = re.search(r"\[memo:(.+?)\]", full_reply)
    if match:
        raw = match.group(1)
        text = re.sub(r"\[memo:.+?\]", "", text).strip()
        parts = raw.split("|", 2)  # 最多拆3段
        if len(parts) == 3:
            title, remind_time, content = parts
            memo = {
                "type": "memo",
                "title": title.strip(),
                "time": remind_time.strip(),
                "content": content.strip(),
                "source": "voice",
                "created": datetime.now().strftime("%Y-%m-%d %H:%M")
            }
            print(f"   📝 识别到备忘录: {memo}")
        else:
            print(f"   ⚠️ 备忘录格式异常: {raw}")

    return memo, text


def send_audio_to_esp32(sock, pcm_data, addr):
    """分片发送音频数据给 ESP32"""
    print(f"   [发送] 回传音频 ({len(pcm_data)} bytes)...")
    chunk_size = 1024
    import time
    for i in range(0, len(pcm_data), chunk_size):
        chunk = pcm_data[i:i + chunk_size]
        sock.sendto(chunk, addr)
        time.sleep(0.01)
    print("   ✅ 发送完毕")


def send_memo_to_esp32(sock, memo_dict, addr):
    """
    将备忘录数据通过 UDP 发给 ESP32
    协议: "MEMO:" + JSON字符串
    ESP32 收到后通过 MQTT 转发给 Android App
    """
    try:
        memo_json = json.dumps(memo_dict, ensure_ascii=False)
        packet = f"MEMO:{memo_json}".encode('utf-8')
        sock.sendto(packet, addr)
        print(f"   📤 备忘录已发送给 ESP32: {memo_json[:80]}...")
    except Exception as e:
        print(f"   ❌ 发送备忘录失败: {e}")


def call_qwen_llm(user_text):
    """调用通义千问大模型"""
    global history_messages

    enriched_text = user_text
    context_parts = []

    weather_keywords = ["天气", "气温", "温度", "下雨", "下雪", "冷不冷", "热不热"]
    time_keywords = ["几点", "时间", "几号", "星期", "日期", "什么时候了"]

    if any(kw in user_text for kw in weather_keywords):
        weather_text, temp = get_weather()
        context_parts.append(f"[系统提示：当前天气数据——{WEATHER_CITY}，天气：{weather_text}，气温：{temp}°C]")

    if any(kw in user_text for kw in time_keywords):
        time_str, date_str = get_current_time()
        context_parts.append(f"[系统提示：当前时间——{date_str} {time_str}]")

    # ▼▼▼ 新增：为备忘录类请求注入当前日期时间，帮助 LLM 推算"明天""后天"等 ▼▼▼
    memo_keywords = ["记一下", "提醒我", "备忘", "别忘了", "记录", "待办", "帮我记", "记住"]
    if any(kw in user_text for kw in memo_keywords):
        now = datetime.now()
        context_parts.append(
            f"[系统提示：当前精确时间为 {now.strftime('%Y-%m-%d %H:%M')}，"
            f"今天是{now.strftime('%Y年%m月%d日')}，星期{'一二三四五六日'[now.weekday()]}，"
            f"请据此推算用户提到的相对时间如\"明天\"\"后天\"\"下周\"等]"
        )

    if context_parts:
        enriched_text = "\n".join(context_parts) + "\n用户说：" + user_text

    print(f"   [大脑] 思考中...")
    history_messages.append({'role': 'user', 'content': enriched_text})
    try:
        response = Generation.call(
            model=MODEL_NAME,
            messages=history_messages,
            result_format='message'
        )
        if response.status_code == HTTPStatus.OK:
            ai_content = response.output.choices[0]['message']['content']
            history_messages.append({'role': 'assistant', 'content': ai_content})
            if len(history_messages) > 21:
                del history_messages[1:3]
            return ai_content
    except Exception as e:
        print(f"LLM Error: {e}")
    return "喵...系统出错啦 [action:4]"


def run_aliyun_asr(wav_file_path):
    """阿里云语音识别"""
    print("   [耳朵] 正在识别...")
    try:
        recognition = Recognition(
            model='paraformer-realtime-v2', format='wav', sample_rate=44100,
            callback=None, vocabulary_id=MY_VOCAB_ID
        )
        response = recognition.call(wav_file_path)
        if response.status_code == HTTPStatus.OK:
            sents = response.get_sentence()
            if sents:
                return sents[0]['text']
    except Exception as e:
        print(f"ASR Error: {e}")
    return ""


# ==========================================
#              4. 主程序
# ==========================================
if __name__ == '__main__':
    if os.name == 'nt':
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind((UDP_IP, UDP_PORT))

    print(f"🚀 茁猫(备忘录版)已启动! 端口: {UDP_PORT}")
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
                    AudioSegment(
                        data=audio_data, sample_width=2, frame_rate=44100, channels=1
                    ).export("temp_input.wav", format="wav")

                    text_in = run_aliyun_asr("temp_input.wav")

                    if text_in:
                        print(f"👂 听见: {text_in}")
                        full_reply = call_qwen_llm(text_in)

                        # ▼▼▼ 先解析备忘录 ▼▼▼
                        memo, reply_without_memo = parse_memo_and_text(full_reply)

                        # ▼▼▼ 再解析动作（从去掉memo标签的文本中解析） ▼▼▼
                        action_id, talk_text = parse_action_and_text(reply_without_memo)

                        # 发送动作指令
                        cmd_msg = f"CMD:{action_id}".encode()
                        sock.sendto(cmd_msg, addr)

                        # ▼▼▼ 如果有备忘录，发送给 ESP32 (ESP32 会通过 MQTT 转发给 App) ▼▼▼
                        if memo:
                            send_memo_to_esp32(sock, memo, addr)

                        print("-" * 40)
                        print(f"🐱 茁猫: {talk_text} | 💖 动作:{action_id}")
                        if memo:
                            print(f"📝 备忘: {memo['title']} @ {memo['time']}")
                        print("-" * 40)

                        # TTS 生成并发送
                        wav_bytes = generate_cosyvoice_tts(talk_text)
                        if wav_bytes:
                            with open("temp_reply.wav", "wb") as f:
                                f.write(wav_bytes)
                            sound = AudioSegment.from_wav("temp_reply.wav")
                            pcm_out = sound.raw_data
                            send_audio_to_esp32(sock, pcm_out, addr)
                    else:
                        print("⚠️ 没听清... 正在告知 ESP32")
                        fail_reply = "喵？刚才没听清，你能再说一遍吗喵？ [action:4]"
                        action_id, talk_text = parse_action_and_text(fail_reply)
                        sock.sendto(f"CMD:{action_id}".encode(), addr)

                        wav_bytes = generate_cosyvoice_tts(talk_text)
                        if wav_bytes:
                            with open("temp_fail.wav", "wb") as f:
                                f.write(wav_bytes)
                            sound = AudioSegment.from_wav("temp_fail.wav")
                            pcm_out = sound.raw_data
                            send_audio_to_esp32(sock, pcm_out, addr)

            elif is_receiving:
                audio_data.extend(data)

        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"Error: {e}")
