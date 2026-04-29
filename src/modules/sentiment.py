# src/modules/sentiment.py 把 TextBlob 替换成 Qwen 做情感分析
"""
情感连续化模块：将文本情感从分类标签转为连续分值 e_τ ∈ [-1, 1]
支持三种后端：
  - 'textblob'     : 轻量，无需 GPU，精度一般
  - 'transformers' : distilbert，精度中等，显存占用小
  - 'qwen'         : Qwen2.5 LLM，精度最高，能理解复杂语境
                     比如 "I wouldn't mind watching it again" 这类隐式正向表达
"""

import torch
import re


class SentimentAnalyzer:
    def __init__(self, backend="textblob", device=None, model_path=None):
        """
        backend    : "textblob" | "transformers" | "qwen"
        device     : torch.device，默认自动检测
        model_path : Qwen 模型路径，backend="qwen" 时必须指定
                     例如 "/root/autodl-tmp/models/Qwen/Qwen2___5-7B-Instruct"
        """
        self.backend    = backend
        self.device     = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model_path = model_path
        self._model     = None
        self._tokenizer = None

    def _load_model(self):
        if self._model is not None:
            return

        if self.backend == "textblob":
            self._model = "textblob"

        elif self.backend == "transformers":
            from transformers import pipeline
            self._model = pipeline(
                "sentiment-analysis",
                model="distilbert-base-uncased-finetuned-sst-2-english",
                device=0 if self.device == "cuda" else -1,
            )

        elif self.backend == "qwen":
            if self.model_path is None:
                raise ValueError("backend='qwen' 时必须指定 model_path 参数")
            from transformers import AutoTokenizer, AutoModelForCausalLM
            self._tokenizer = AutoTokenizer.from_pretrained(
                self.model_path, trust_remote_code=True
            )
            self._model = AutoModelForCausalLM.from_pretrained(
                self.model_path,
                torch_dtype=torch.float16,
                device_map="auto",
                trust_remote_code=True,
            )
            self._model.eval()
        else:
            raise ValueError(f"Unsupported backend: {self.backend}")

    # ── 核心打分函数 ───────────────────────────────────────────────────────────

    def score(self, text: str) -> float:
        """
        输入：用户话语文本
        输出：e_τ ∈ [-1.0, 1.0]
          正值 = 正向情感（喜欢）
          负值 = 负向情感（不喜欢）
          0    = 中性
        """
        if not text or not text.strip():
            return 0.0

        self._load_model()

        if self.backend == "textblob":
            from textblob import TextBlob
            polarity = TextBlob(text).sentiment.polarity
            return round(float(polarity), 4)

        elif self.backend == "transformers":
            result = self._model(text[:512])[0]
            raw_score = float(result["score"])
            if result["label"] == "NEGATIVE":
                raw_score = -raw_score
            return round(raw_score, 4)

        elif self.backend == "qwen":
            return self._qwen_score(text)

    def batch_score(self, texts: list) -> list:
        """批量打分"""
        if self.backend == "qwen" and len(texts) > 1:
            return self._qwen_batch_score(texts)
        return [self.score(t) for t in texts]

    # ── Qwen 情感打分 ──────────────────────────────────────────────────────────

    _QWEN_PROMPT = """You are a sentiment analyzer for a movie recommendation system.
Analyze the user's sentiment toward movies in the conversation below.
Return ONLY a single number between -1.0 and 1.0:
  1.0  = very positive (loves it, highly recommends)
  0.5  = mildly positive (likes it, open to watching)
  0.0  = neutral (no clear preference)
 -0.5  = mildly negative (not interested)
 -1.0  = very negative (dislikes it, refuses to watch)

Examples:
"I absolutely loved The Dark Knight!" → 0.9
"I wouldn't mind watching it again" → 0.4
"It was okay I guess" → 0.1
"Not really my type of movie" → -0.4
"I hated that film, it was terrible" → -0.9

Conversation: {text}

Output ONLY the number, nothing else:"""

    def _qwen_score(self, text: str) -> float:
        """单条文本用 Qwen 打情感分"""
        prompt = self._QWEN_PROMPT.format(text=text[:500])
        messages = [
            {"role": "system", "content": "You are a sentiment analyzer. Output only a number."},
            {"role": "user", "content": prompt}
        ]
        formatted = self._tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self._tokenizer([formatted], return_tensors="pt").to(self._model.device)
        with torch.no_grad():
            outputs = self._model.generate(
                **inputs,
                max_new_tokens=10,
                temperature=0.1,
                do_sample=True,
                pad_token_id=self._tokenizer.eos_token_id,
            )
        response = self._tokenizer.decode(
            outputs[0][inputs["input_ids"].shape[1]:],
            skip_special_tokens=True
        ).strip()
        return self._parse_score(response)

    def _qwen_batch_score(self, texts: list) -> list:
        """
        批量处理：把多条文本拼成一个 prompt，一次推理出所有分数。
        适合训练阶段批量处理，减少推理次数。
        """
        # 每次最多处理 8 条，避免 prompt 过长
        results = []
        for i in range(0, len(texts), 8):
            batch = texts[i:i+8]
            scores = self._qwen_batch_single(batch)
            results.extend(scores)
        return results

    def _qwen_batch_single(self, texts: list) -> list:
        """处理一个小 batch"""
        items = '\n'.join(
            f'{j+1}. "{t[:200]}"' for j, t in enumerate(texts)
        )
        prompt = f"""Analyze the sentiment of each text toward movies. 
Return ONLY a JSON array of numbers between -1.0 and 1.0, one per text.
Example for 3 texts: [0.8, -0.3, 0.0]

Texts:
{items}

Output ONLY the JSON array:"""

        messages = [
            {"role": "system", "content": "You are a sentiment analyzer. Output only a JSON array."},
            {"role": "user", "content": prompt}
        ]
        formatted = self._tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self._tokenizer([formatted], return_tensors="pt").to(self._model.device)
        with torch.no_grad():
            outputs = self._model.generate(
                **inputs,
                max_new_tokens=50,
                temperature=0.1,
                do_sample=True,
                pad_token_id=self._tokenizer.eos_token_id,
            )
        response = self._tokenizer.decode(
            outputs[0][inputs["input_ids"].shape[1]:],
            skip_special_tokens=True
        ).strip()

        # 解析 JSON 数组
        try:
            match = re.search(r'\[.*?\]', response, re.DOTALL)
            if match:
                scores = eval(match.group(0))
                if isinstance(scores, list) and len(scores) == len(texts):
                    return [max(-1.0, min(1.0, float(s))) for s in scores]
        except:
            pass

        # 解析失败时逐条处理
        return [self._qwen_score(t) for t in texts]

    @staticmethod
    def _parse_score(response: str) -> float:
        """从 Qwen 输出中提取数字"""
        response = response.strip()
        # 提取第一个数字
        match = re.search(r'-?\d+\.?\d*', response)
        if match:
            val = float(match.group(0))
            return round(max(-1.0, min(1.0, val)), 4)
        return 0.0


# ── 快速测试 ──────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--backend', default='textblob')
    parser.add_argument('--model_path', default=None)
    args = parser.parse_args()

    analyzer = SentimentAnalyzer(backend=args.backend, model_path=args.model_path)
    tests = [
        "I absolutely loved The Dark Knight, it was amazing!",
        "The film was terrible and boring, I hated it.",
        "It was okay, nothing special.",
        "I wouldn't mind watching it again sometime.",
        "Not really my type of movie to be honest.",
        "Have you seen Inception? I'm looking for something similar.",
    ]
    print(f"Backend: {args.backend}")
    for t in tests:
        print(f"[{analyzer.score(t):+.3f}] {t}")



# # src/modules/sentiment.py
# """
# 情感连续化模块：将文本情感从分类标签转为连续分值 e_τ ∈ [-1, 1]
# 支持两种后端：
#   - 'transformers'（默认，精度高）
#   - 'textblob'（轻量，无需 GPU）
# """

# import torch


# class SentimentAnalyzer:
#     def __init__(self, backend="transformers", device=None):
#         """
#         backend: "transformers" | "textblob"
#         device:  torch.device，默认自动检测
#         """
#         self.backend = backend
#         self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
#         self._model = None  # lazy init

#     def _load_model(self):
#         if self._model is not None:
#             return
#         if self.backend == "transformers":
#             from transformers import pipeline
#             # 使用轻量级 distilbert 模型，避免显存压力
#             self._model = pipeline(
#                 "sentiment-analysis",
#                 model="distilbert-base-uncased-finetuned-sst-2-english",
#                 device=0 if self.device == "cuda" else -1,
#             )
#         elif self.backend == "textblob":
#             from textblob import TextBlob
#             self._model = "textblob"
#         else:
#             raise ValueError(f"Unsupported backend: {self.backend}")

#     def score(self, text: str) -> float:
#         """
#         输入：用户话语文本
#         输出：e_τ ∈ [-1.0, 1.0]
#           正值 = 正向情感（喜欢）
#           负值 = 负向情感（不喜欢）
#           0    = 中性
#         """
#         if not text or not text.strip():
#             return 0.0

#         self._load_model()

#         if self.backend == "transformers":
#             result = self._model(text[:512])[0]  # 截断防止超长
#             raw_score = float(result["score"])   # 置信度 ∈ (0, 1]
#             # POSITIVE → 正值，NEGATIVE → 负值
#             if result["label"] == "NEGATIVE":
#                 raw_score = -raw_score
#             return round(raw_score, 4)

#         elif self.backend == "textblob":
#             from textblob import TextBlob
#             polarity = TextBlob(text).sentiment.polarity  # ∈ [-1, 1]
#             return round(float(polarity), 4)

#     def batch_score(self, texts: list) -> list:
#         """批量打分，适合训练阶段"""
#         return [self.score(t) for t in texts]


# # ── 快速测试 ──────────────────────────────────────────────────────────────────
# if __name__ == "__main__":
#     analyzer = SentimentAnalyzer(backend="textblob")
#     tests = [
#         "I really loved that movie, it was fantastic!",
#         "The film was terrible and boring.",
#         "It was okay, nothing special.",
#     ]
#     for t in tests:
#         print(f"[{analyzer.score(t):+.3f}] {t}")