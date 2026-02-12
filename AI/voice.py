# -*- coding: utf-8 -*-
import os
import time
import dashscope
from dashscope.audio.tts_v2 import VoiceEnrollmentService

# ================= 配置区域 =================

# 1. 【请替换】你的阿里云 DashScope API Key
# 建议设置在环境变量中，或者直接填在这里（注意不要泄露给他人）
my_api_key = "sk-26ac2fd867404de4846819e019ebfd92"

# 2. 【请替换】你的 GitHub Raw 音频链接
# 必须是以 https://raw.githubusercontent.com/... 开头的链接
# 确保文件是 wav/mp3/m4a 格式，时长 10-20秒
audio_url = "https://raw.githubusercontent.com/izhuomao/Voice/main/录音 (2).m4a"

# 3. 自定义配置
target_model = "cosyvoice-v3-plus"  # 推荐模型，合成时也要用这个
voice_prefix = "myvoice"  # 音色名前缀，仅限小写字母和数字


# ===========================================

def create_and_wait_voice():
    # 设置 API Key
    if my_api_key.startswith("sk-"):
        dashscope.api_key = my_api_key
    else:
        # 尝试从环境变量获取
        dashscope.api_key = os.getenv("DASHSCOPE_API_KEY")

    if not dashscope.api_key:
        print("❌ 错误：未配置 API Key。请在代码中填写或设置环境变量 DASHSCOPE_API_KEY")
        return

    service = VoiceEnrollmentService()

    print(f"--- 步骤 1: 提交声音复刻任务 ---")
    print(f"音频源: {audio_url}")
    print(f"目标模型: {target_model}")

    try:
        # 提交创建请求
        # language_hints=['zh'] 辅助模型识别中文，提高准确度
        voice_id = service.create_voice(
            target_model=target_model,
            prefix=voice_prefix,
            url=audio_url,
            language_hints=['zh']
        )
        print(f"✅ 提交成功！")
        print(f"生成的新 Voice ID: {voice_id}")
    except Exception as e:
        print(f"❌ 提交失败，请检查 URL 是否为 Public 且为 Raw 链接。错误信息:\n{e}")
        return

    print(f"\n--- 步骤 2: 等待训练完成 (轮询状态) ---")
    # 轮询检查状态，直到变为 OK
    max_retries = 30
    for i in range(max_retries):
        try:
            voice_info = service.query_voice(voice_id=voice_id)
            status = voice_info.get("status")
            print(f"[{i + 1}/{max_retries}] 当前状态: {status}")

            if status == "OK":
                print("\n🎉 恭喜！音色复刻完成。")
                print("------------------------------------------------")
                print(f"请保存好你的 Voice ID: 【 {voice_id} 】")
                print(f"后续语音合成时，请使用模型: {target_model}")
                print("------------------------------------------------")
                break
            elif status == "UNDEPLOYED":
                print("❌ 复刻失败 (UNDEPLOYED)。原因可能是音频质量太差、有人声重叠或背景音乐。")
                break

            # 如果是 DEPLOYING，继续等待
            time.sleep(5)
        except Exception as e:
            print(f"查询状态出错: {e}")
            time.sleep(5)
    else:
        print("❌ 超时：长时间未获取到最终状态，请稍后自行查询。")


if __name__ == "__main__":
    create_and_wait_voice()
