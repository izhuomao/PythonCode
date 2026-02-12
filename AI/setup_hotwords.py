import dashscope
from dashscope.audio.asr import VocabularyService
import os

# 1. 配置 Key
dashscope.api_key = "sk-26ac2fd867404de4846819e019ebfd92"

# 2. 填入你刚才生成的 ID (必须填对！)
# 比如: "vocab-zhuomao-12345678"
MY_VOCAB_ID = "vocab-zhuomao-c0dcc1be55d7400888dada26fa09c4fe"

# 3. 定义新的完整列表 (记得把旧的也带上，想删谁就不写谁)
new_hotwords_list = [
    {"text": "茁猫", "weight": 5},
    {"text": "金来", "weight": 5},
    {"text": "嘎嘎", "weight": 4},
    {"text": "阿姨洗铁路", "weight": 3},
    # --- 下面是新增的 ---
    {"text": "吃饭了吗", "weight": 4},
    {"text": "启动自爆模式", "weight": 5}
]


def update_hotwords():
    print(f"正在更新热词表: {MY_VOCAB_ID} ...")
    service = VocabularyService()

    try:
        # 调用更新接口
        service.update_vocabulary(MY_VOCAB_ID, new_hotwords_list)
        print("✅ 更新成功！")
        print("新的热词通过 AS R识别时大约需要几秒钟到几分钟生效。")

    except Exception as e:
        print(f"❌ 更新失败: {e}")


# 额外赠送：如果你忘了 ID，可以用这个函数查一下
def find_my_id():
    service = VocabularyService()
    # 列出所有热词表
    vocabs = service.list_vocabularies()
    print("查到的所有热词表:", vocabs)


if __name__ == '__main__':
    # 1. 如果你有ID，直接运行更新
    update_hotwords()

    # 2. 如果你刚才忘了存ID，把上面注释掉，运行下面这行找回ID
    # find_my_id()
