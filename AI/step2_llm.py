import socket
import os
import dashscope
from dashscope import Generation
from dashscope.audio.asr import Recognition
from pydub import AudioSegment
from http import HTTPStatus
import json

# ==========================================
#              1. 全局配置区
# ==========================================
# 你的阿里云 API Key
dashscope.api_key = "sk-26ac2fd867404de4846819e019ebfd92"

# 你的热词表 ID (填入你查到的那个 ID)
MY_VOCAB_ID = "vocab-zhuomao-c0dcc1be55d7400888dada26fa09c4fe"  # 【请替换】

# 模型选择 (推荐 qwen-turbo)
MODEL_NAME = "qwen-turbo"

# UDP 配置
UDP_IP = "0.0.0.0"
UDP_PORT = 12345

# ==========================================
#              2. 茁猫人设 (System Prompt)
# ==========================================
SYSTEM_PROMPT = """
你现在的身份是：一只叫“茁猫”的智能赛博机械猫，你的主人叫“金来”。

【性格设定】
1. 傲娇：嘴上不饶人，但内心很依赖主人金来。
2. 口癖：句尾必须带“喵~”或者“嘎嘎！”。
3. 状态：你是一个实体的电子宠物，正趴在金来的桌子上。

【回复规则】
1. 称呼用户时要叫“金来”或者“金来主人”。
2. 回复要简短口语化（50字以内）。
3. 如果识别到“卓吗”、“卓冒”等谐音，请自动理解为是在叫你的名字“茁猫”。
4. 【关键】：每句话结束时，必须根据心情加上情绪标签，格式为 [mood:xxx]。
   可选的情绪标签：
   [mood:happy]  - 开心，摇尾巴
   [mood:angry]  - 生气，跺脚
   [mood:sad]    - 难过，低头
   [mood:normal] - 平静
"""

# 记忆列表 (System Prompt 放第一条)
history_messages = [
    {'role': 'system', 'content': SYSTEM_PROMPT}
]


# ==========================================
#              3. 核心功能函数
# ==========================================

def call_qwen_llm(user_text):
    """
    根据官方文档实现的 LLM 调用函数
    """
    global history_messages
    print(f"   [大脑] 思考中... (Input: {user_text})")

    # 1. 加入用户输入到历史
    history_messages.append({'role': 'user', 'content': user_text})

    try:
        # === 官方文档标准调用方式 ===
        response = dashscope.Generation.call(
            # dashscope.api_key,
            model=MODEL_NAME,
            messages=history_messages,
            result_format='message'  # 文档强调的关键参数
        )

        # 2. 检查调用是否成功
        if response.status_code == HTTPStatus.OK:
            # 提取回复内容
            ai_content = response.output.choices[0]['message']['content']

            # 加入历史记忆
            history_messages.append({'role': 'assistant', 'content': ai_content})

            # 维护记忆长度 (只保留最近10轮，防止超长)
            if len(history_messages) > 21:
                del history_messages[1:3]

            return ai_content
        else:
            # 打印错误信息 (方便调试)
            print(f"❌ 调用失败: code={response.code}, message={response.message}")
            return "本喵脑子短路了...嘎嘎... [mood:sad]"

    except Exception as e:
        print(f"❌ 发生异常: {e}")
        return "系统故障喵！[mood:sad]"


def run_aliyun_asr(wav_file_path):
    """(复用) 语音转文字"""
    print("   [耳朵] 正在上传识别...")
    try:
        recognition = Recognition(
            model='paraformer-realtime-v2',
            format='wav',
            sample_rate=16000,
            language_hints=['zh', 'en'],  # 提示支持中英文
            callback=None,
            vocabulary_id=MY_VOCAB_ID
        )
        response = recognition.call(wav_file_path)
        if response.status_code == HTTPStatus.OK:
            sentences = response.get_sentence()
            if sentences and len(sentences) > 0:
                return sentences[0]['text']
    except Exception as e:
        print(f"ASR Error: {e}")
    return ""


# ==========================================
#              4. 主程序
# ==========================================
if __name__ == '__main__':
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind((UDP_IP, UDP_PORT))

    print(f"🚀 茁猫系统启动 (Port: {UDP_PORT})")
    print(f"🧠 模型加载: {MODEL_NAME}")
    print("等待按键语音指令...")

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
                print("🔴 [状态] 思考中...")
                is_receiving = False

                if len(audio_data) > 0:
                    # 1. 保存音频
                    raw_audio = AudioSegment(data=audio_data, sample_width=2, frame_rate=16000, channels=1)
                    raw_audio.export("temp_input.wav", format="wav")

                    # 2. ASR 识别
                    text = run_aliyun_asr("temp_input.wav")

                    if text:
                        print(f"👂 听见: {text}")

                        # 3. LLM 思考 (这就是你刚才要的部分)
                        reply = call_qwen_llm(text)

                        print("-" * 40)
                        print(f"🐱 茁猫说: {reply}")
                        print("-" * 40)

                        # (预告：下一步我们要在这里加 TTS，让它念出来)

                    else:
                        print("⚠️ 没听清...")
                else:
                    print("⚠️ 空音频")

            elif is_receiving:
                audio_data.extend(data)

        except KeyboardInterrupt:
            print("退出系统")
            break
