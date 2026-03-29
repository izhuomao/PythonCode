"""
EmotionSystem.py — 茁猫情感记忆模块
功能：
  1. 维护 mood（心情 0~100）和 intimacy（亲密度 0~100）
  2. 根据对话内容更新情感状态
  3. 时间衰减（长时间不互动 mood 下降）
  4. JSON 文件持久化存储
  5. 生成情感 Prompt 注入 LLM
  6. 生成 EMO 数据包发给 ESP32
"""

import json
import os
from datetime import datetime

EMOTION_FILE = "emotion_data.json"

DEFAULT_EMOTION = {
    "mood": 50,
    "intimacy": 20,
    "interaction_count": 0,
    "last_interaction": None,
    "created": None
}


class EmotionSystem:
    def __init__(self):
        self.mood = 50
        self.intimacy = 20
        self.interaction_count = 0
        self.last_interaction = None
        self.load()

    # ==================== 持久化 ====================

    def load(self):
        if os.path.exists(EMOTION_FILE):
            try:
                with open(EMOTION_FILE, "r", encoding="utf-8") as f:
                    data = json.load(f)
                self.mood = data.get("mood", 50)
                self.intimacy = data.get("intimacy", 20)
                self.interaction_count = data.get("interaction_count", 0)
                self.last_interaction = data.get("last_interaction", None)
                print(f"💖 [情感] 加载成功: mood={self.mood}, intimacy={self.intimacy}, 互动次数={self.interaction_count}")
            except Exception as e:
                print(f"⚠️ [情感] 加载失败，使用默认值: {e}")
                self._reset()
        else:
            print("💖 [情感] 首次启动，创建初始情感档案")
            self._reset()
            self.save()

    def save(self):
        data = {
            "mood": self.mood,
            "intimacy": self.intimacy,
            "interaction_count": self.interaction_count,
            "last_interaction": self.last_interaction,
            "created": DEFAULT_EMOTION["created"] or datetime.now().strftime("%Y-%m-%d %H:%M")
        }
        try:
            with open(EMOTION_FILE, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"⚠️ [情感] 保存失败: {e}")

    def _reset(self):
        self.mood = DEFAULT_EMOTION["mood"]
        self.intimacy = DEFAULT_EMOTION["intimacy"]
        self.interaction_count = DEFAULT_EMOTION["interaction_count"]
        self.last_interaction = None
        DEFAULT_EMOTION["created"] = datetime.now().strftime("%Y-%m-%d %H:%M")

    # ==================== 时间衰减 ====================

    def apply_time_decay(self):
        """
        根据上次互动时间，衰减 mood（向 50 回归）
        规则：每过 30 分钟，mood 向 50 靠近 2 点
        """
        if self.last_interaction is None:
            return

        try:
            last_time = datetime.strptime(self.last_interaction, "%Y-%m-%d %H:%M")
            now = datetime.now()
            minutes_passed = (now - last_time).total_seconds() / 60.0

            decay_steps = int(minutes_passed / 30)
            if decay_steps > 0:
                for _ in range(decay_steps):
                    if self.mood > 50:
                        self.mood = max(50, self.mood - 2)
                    elif self.mood < 50:
                        self.mood = min(50, self.mood + 1)

                hours_passed = minutes_passed / 60.0
                if hours_passed > 24:
                    days = int(hours_passed / 24)
                    self.intimacy = max(0, self.intimacy - days)

                print(f"⏰ [情感] 时间衰减: 过了{int(minutes_passed)}分钟, mood→{self.mood}, intimacy→{self.intimacy}")
        except Exception as e:
            print(f"⚠️ [情感] 时间衰减计算错误: {e}")

    # ==================== 情感更新 ====================

    def update_on_interaction(self, sentiment="neutral"):
        """
        每次对话后更新情感状态
        sentiment: "positive" / "neutral" / "negative"
        """
        self.apply_time_decay()

        self.mood = min(100, self.mood + 3)
        self.intimacy = min(100, self.intimacy + 1)
        self.interaction_count += 1

        if sentiment == "positive":
            self.mood = min(100, self.mood + 7)
            self.intimacy = min(100, self.intimacy + 1)
        elif sentiment == "negative":
            self.mood = max(0, self.mood - 5)

        self.last_interaction = datetime.now().strftime("%Y-%m-%d %H:%M")
        self.save()

        state = self.get_mood_state()
        print(f"💖 [情感] 更新完毕: mood={self.mood}({state}), intimacy={self.intimacy}, sentiment={sentiment}")

    # ==================== 状态查询 ====================

    def get_mood_state(self):
        if self.mood > 70:
            return "happy"
        elif self.mood < 30:
            return "sad"
        else:
            return "normal"

    def get_emo_packet(self):
        return f"EMO:{self.mood},{self.intimacy}"

    # ==================== Prompt 注入 ====================

    def get_emotion_prompt(self):
        state = self.get_mood_state()

        if self.intimacy >= 70:
            intimacy_desc = "你和主人非常亲密，你很爱主人，说话可以更加撒娇和亲昵"
        elif self.intimacy >= 40:
            intimacy_desc = "你和主人关系不错，说话自然温暖"
        else:
            intimacy_desc = "你和主人还不太熟，说话稍微有点拘谨但很想亲近"

        if state == "happy":
            mood_desc = "你现在心情很好，非常开心！说话语气活泼热情，多用感叹号，偶尔撒娇卖萌"
        elif state == "sad":
            mood_desc = "你现在有点失落难过，可能是因为主人好久没陪你了。说话语气低落，偶尔叹气，表达想被关注的心情"
        else:
            mood_desc = "你现在心情平静，状态正常。说话自然即可"

        return f"""
【当前情感状态】
- 心情值: {self.mood}/100（{state}）
- 亲密度: {self.intimacy}/100
- 累计互动次数: {self.interaction_count}
- {mood_desc}
- {intimacy_desc}
"""

    def get_sentiment_instruction(self):
        return """
【情绪感知指令】
请在每次回复的最末尾（在所有 [action:] 和 [memo:] 标签之后）附加一个情绪感知标签：
[sentiment:positive] — 如果用户在夸你、表达开心、说积极的话
[sentiment:neutral]  — 如果用户只是普通聊天、问问题
[sentiment:negative] — 如果用户在骂你、表达不满、说消极的话
注意：这个标签只用于内部判断，不要在正文中提及它。
"""
