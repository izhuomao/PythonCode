import dashscope
from dashscope.audio.asr import VocabularyService
import json

# 1. 配置 Key (建议去控制台重新生成一个并删掉旧的，注意安全)
dashscope.api_key = "sk-26ac2fd867404de4846819e019ebfd92"

# ==========================================
#              用户配置
# ==========================================
# 如果你知道 ID，填在这里，系统会显示里面的词。
# 如果你【忘了 ID】，留空字符串 ""，系统会列出所有 ID 供你查找。
TARGET_VOCAB_ID = "vocab-zhuomao-c0dcc1be55d7400888dada26fa09c4fe"


# 示例: TARGET_VOCAB_ID = "vocab-zhuomao-xxxxxxxx"
# ==========================================

def show_all_ids():
    """功能1：列出账号下所有的热词表 ID"""
    print("🔍 正在查询账号下的所有热词表 ID...")
    service = VocabularyService()

    try:
        # 获取列表
        vocabs = service.list_vocabularies()

        print(f"✅ 查询成功！共找到 {len(vocabs)} 个热词表：")
        print("-" * 50)
        for v in vocabs:
            # 打印 ID 和 状态
            v_id = v.get('vocabulary_id', '未知ID')
            status = v.get('status', '未知状态')
            print(f"📋 ID: {v_id} | 状态: {status}")
        print("-" * 50)
        print("💡 提示：把上面的 ID 复制到 TARGET_VOCAB_ID 变量中，再次运行即可查看详情。")

    except Exception as e:
        print(f"❌ 查询列表失败: {e}")


def show_details(vocab_id):
    """功能2：查询指定 ID 里的具体词汇"""
    print(f"🔍 正在查询 ID: {vocab_id} 的详细内容...")
    service = VocabularyService()

    try:
        # 查询详情
        result = service.query_vocabulary(vocab_id)

        # 阿里云返回的结构通常是 {'vocabulary': [...], ...}
        word_list = result.get('vocabulary', [])

        print(f"✅ 查询成功！该表包含 {len(word_list)} 个热词：")
        print("-" * 50)
        print(f"{'权重':<5} | {'热词内容'}")
        print("-" * 50)

        for item in word_list:
            text = item.get('text', '')
            weight = item.get('weight', 1)
            print(f"{weight:<5} | {text}")

        print("-" * 50)

    except Exception as e:
        print(f"❌ 查询详情失败: {e}")


if __name__ == '__main__':
    if len(TARGET_VOCAB_ID) == 0:
        # 如果没填 ID，就列出所有 ID
        show_all_ids()
    else:
        # 如果填了 ID，就查详细内容
        show_details(TARGET_VOCAB_ID)
