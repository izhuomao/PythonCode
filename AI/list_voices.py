# -*- coding: utf-8 -*-
import os
import dashscope
from dashscope.audio.tts_v2 import VoiceEnrollmentService

# ================= 配置区域 =================
# 【请替换】你的阿里云 DashScope API Key
my_api_key = "sk-26ac2fd867404de4846819e019ebfd92"

# 可选：按前缀筛选（如果只想看 myvoice 开头的音色）
# 设为 None 则查询所有
search_prefix = None


# search_prefix = "myvoice"
# ===========================================

def list_all_voices():
    # 1. 配置 API Key
    if my_api_key.startswith("sk-"):
        dashscope.api_key = my_api_key
    else:
        dashscope.api_key = os.getenv("DASHSCOPE_API_KEY")

    if not dashscope.api_key:
        print("❌ 错误：未配置 API Key")
        return

    service = VoiceEnrollmentService()

    print(f"--- 开始查询音色列表 ---")

    # 2. 分页查询逻辑
    page_index = 0
    page_size = 10  # 每次查10条
    total_found = 0

    while True:
        try:
            # 调用接口
            voices = service.list_voices(
                prefix=search_prefix,
                page_index=page_index,
                page_size=page_size
            )

            # 如果当前页没有数据，说明查完了
            if not voices:
                break

            # 打印当前页的音色信息
            for v in voices:
                total_found += 1
                v_id = v.get('voice_id')
                status = v.get('status')
                create_time = v.get('gmt_create')

                print(f"[{total_found}] ID: {v_id}")
                print(f"    状态: {status} | 创建时间: {create_time}")
                print("-" * 40)

            # 准备查下一页
            page_index += 1

        except Exception as e:
            print(f"❌ 查询出错: {e}")
            break

    if total_found == 0:
        print("📭 未找到任何音色。")
    else:
        print(f"✅ 查询结束，共找到 {total_found} 个音色。")


if __name__ == "__main__":
    list_all_voices()
