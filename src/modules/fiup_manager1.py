# src/modules/fiup_manager.py
"""
FIUP (Fine-grained Incremental User Profile) Manager  —— 完整重写版

双库结构：
  显性库 explicit_lib : { "Type:value" -> weight }
      - Type 取值：Genre / Actor / Director / Movie / Keyword
      - weight > 0  表示偏好，weight < 0 表示排斥
      - 同一属性出现在 feedback_log 的 Reject 记录时权重直接置为强负值

  隐性库 implicit_lib : Tensor[emb_dim]
      - 由 RoBERTa [CLS] 向量按情感方向累积
      - 正向情感拉近，负向情感推开

新增结构：
  feedback_log    : List[{item, feedback_type, turn}]
      - feedback_type: "Accept" | "Reject" | "Hesitant"
  mood_history    : List[float]  —— 每轮 e_tau 历史
  uncertain_attrs : 由显性库中 |weight| < uncertain_threshold 的属性自动派生，只读

build_profile_prompt() 输出格式（注入 UniCRS Prompt）：
  [USER_PROFILE]
    Genre: comedy(+), horror(-) |
    Actor: Adam Sandler(+) |
    Reject: The Dark Knight, Avengers |
    Mood: Happy
"""

import json
import os
import re
from typing import Dict, List, Optional, Tuple

import torch


# ── 类型推断规则（与 process.py 中的逻辑保持一致）──────────────────────────────
GENRE_KEYWORDS = {
    "action", "comedy", "comedies", "horror", "thriller", "drama", "dramas",
    "romance", "romantic", "musical", "musicals", "sci-fi", "science fiction",
    "sci fi", "fantasy", "western", "documentary", "animation", "animated",
    "adventure", "mystery", "crime", "sports", "dark comedy", "superhero",
    "suspense", "family", "animation",
}

# DBpedia URI 后缀 → 属性类型映射
URI_SUFFIX_TO_TYPE = {
    "actor":          "Actor",
    "actress":        "Actor",
    "director":       "Director",
    "filmmaker":      "Director",
    "producer":       "Director",
    "film_producer":  "Director",
    "screenwriter":   "Keyword",
    "composer":       "Keyword",
    "film_editor":    "Keyword",
    "writer":         "Keyword",
    "film":           "Movie",
}

# e_tau -> mood label 映射阈值
def _etau_to_mood(e_tau: float) -> str:
    if e_tau >= 0.5:
        return "Happy"
    elif e_tau >= 0.1:
        return "Positive"
    elif e_tau <= -0.5:
        return "Negative"
    elif e_tau <= -0.1:
        return "Slightly Negative"
    else:
        return "Neutral"


def infer_attr_type(name: str, uri: str = "") -> str:
    """
    根据实体名和可选 DBpedia URI 后缀推断属性类型。
    返回: "Genre" | "Actor" | "Director" | "Movie" | "Keyword"
    """
    # 优先用 URI 后缀判断
    if uri:
        r = uri.replace("<http://dbpedia.org/resource/", "").rstrip(">")
        if "(" in r:
            suffix = r[r.rfind("(") + 1: r.rfind(")")]
            # 带年份的 URI 后缀如 "2018_film" 归为 Movie
            if suffix.endswith("film") or re.match(r"\d{4}_film", suffix):
                return "Movie"
            for key, atype in URI_SUFFIX_TO_TYPE.items():
                if key in suffix:
                    return atype

    # 用名称判断
    nl = name.strip().lower()
    if nl in GENRE_KEYWORDS or any(g in nl for g in GENRE_KEYWORDS):
        return "Genre"
    # 含年份括号，如 "Super Troopers (2001)"
    if re.search(r"\(\d{4}\)", name):
        return "Movie"
    return "Keyword"


# ─────────────────────────────────────────────────────────────────────────────

class FIUPManager:
    """
    细粒度增量用户画像管理器。

    Parameters
    ----------
    emb_dim          : 隐性库向量维度，与 RoBERTa hidden_size 一致（768）
    alpha            : 遗忘衰减系数，越大历史越重要（默认 0.8）
    threshold        : 显性库输出阈值，|weight| > threshold 才输出（默认 0.1）
    uncertain_thresh : 置信度低于此值视为"不确定"属性（默认 0.3）
    reject_weight    : Reject 反馈对应的显性库强负权重（默认 -0.9）
    device           : 计算设备
    """

    def __init__(
        self,
        emb_dim: int,
        alpha: float = 0.8,
        threshold: float = 0.1,
        uncertain_thresh: float = 0.3,
        reject_weight: float = -0.9,
        device: Optional[torch.device] = None,
    ):
        self.alpha            = alpha
        self.threshold        = threshold
        self.uncertain_thresh = uncertain_thresh
        self.reject_weight    = reject_weight
        self.emb_dim          = emb_dim
        self.device           = device or torch.device("cpu")

        # 显性库：key = "Type:value"（如 "Genre:comedy"），value = weight ∈ [-1, 1]
        self.explicit_lib: Dict[str, float] = {}

        # 隐性库：语义偏好向量，shape = [emb_dim]
        self.implicit_lib: torch.Tensor = torch.zeros(emb_dim, device=self.device)

        # 推荐反馈日志：[{"item": str, "feedback_type": str, "turn": int}]
        self.feedback_log: List[Dict] = []

        # 每轮 e_tau 历史（用于 mood 趋势和 dialogue_style 推断）
        self.mood_history: List[float] = []

        # 当前轮次计数
        self._turn: int = 0

    # ── 核心更新入口 ──────────────────────────────────────────────────────────

    def update_profile(
        self,
        attributes: List[str],
        e_tau: float,
        context_emb: torch.Tensor,
        attr_uris: Optional[List[str]] = None,
        feedback_items: Optional[List[Tuple[str, str]]] = None,
    ):
        """
        单轮对话后更新双库及反馈日志。

        Parameters
        ----------
        attributes     : 本轮实体名列表（来自 entity_names 字段）
        e_tau          : 情感分值 ∈ [-1, 1]（来自 SentimentAnalyzer）
        context_emb    : RoBERTa [CLS] 向量，shape = [emb_dim]
        attr_uris      : 与 attributes 对齐的 DBpedia URI 列表（可选，提升类型精度）
        feedback_items : [(movie_title, feedback_type), ...]，显式推荐反馈（可选）
                         feedback_type: "Accept" | "Reject" | "Hesitant"
        """
        self._turn += 1
        self.mood_history.append(e_tau)

        self._update_explicit(attributes, e_tau, attr_uris or [])
        self._update_implicit(context_emb, e_tau)

        if feedback_items:
            self._update_feedback(feedback_items)

    # ── 显性库更新 ────────────────────────────────────────────────────────────

    def _update_explicit(
        self,
        attributes: List[str],
        e_tau: float,
        attr_uris: List[str],
    ):
        """
        显性库更新公式：
          key   = "Type:value"（如 "Genre:comedy" / "Actor:Adam Sandler"）
          w_new = α * w_old + (1-α) * |e_τ|
          e_τ < 0 时 w_new 取负（排斥方向）
        """
        strength = abs(e_tau)
        uri_map = dict(zip(attributes, attr_uris)) if attr_uris else {}

        for attr in attributes:
            uri  = uri_map.get(attr, "")
            atype = infer_attr_type(attr, uri)
            key  = f"{atype}:{attr}"

            prev = self.explicit_lib.get(key, 0.0)

            # 若已被 Reject 标记（强负值），不再被正向情感覆盖
            if prev <= self.reject_weight + 0.05 and e_tau > 0:
                continue

            new_weight = self.alpha * prev + (1 - self.alpha) * strength
            if e_tau < 0:
                new_weight = -abs(new_weight)

            self.explicit_lib[key] = round(new_weight, 6)

    # ── 隐性库更新 ────────────────────────────────────────────────────────────

    def _update_implicit(self, context_emb: torch.Tensor, e_tau: float):
        """
        隐性库更新公式：
          I_τ = α * I_τ-1 + (1-α) * (context_emb * e_τ)
        正向情感拉近当前语义方向，负向推开。
        """
        context_emb = context_emb.to(self.device).detach().float()
        delta = (1 - self.alpha) * (context_emb * e_tau)
        self.implicit_lib = self.alpha * self.implicit_lib + delta

    # ── 反馈日志更新 ──────────────────────────────────────────────────────────

    def _update_feedback(self, feedback_items: List[Tuple[str, str]]):
        """
        将推荐反馈写入 feedback_log，Reject 同时写入显性库强负值。
        """
        for item, ftype in feedback_items:
            self.feedback_log.append({
                "item":          item,
                "feedback_type": ftype,
                "turn":          self._turn,
            })
            # Reject → 显性库强负标记，避免重复推荐
            if ftype == "Reject":
                key = f"Movie:{item}"
                self.explicit_lib[key] = self.reject_weight

    # ── 查询接口 ──────────────────────────────────────────────────────────────

    def get_liked_attrs(self, atype: Optional[str] = None) -> List[Tuple[str, float]]:
        """
        返回正向偏好属性列表，按权重降序。
        atype: 过滤特定类型（"Genre" / "Actor" / "Movie" 等），None = 全部
        返回: [(attr_name, weight), ...]
        """
        result = [
            (k.split(":", 1)[1], v)
            for k, v in self.explicit_lib.items()
            if v > self.threshold and (atype is None or k.startswith(atype + ":"))
        ]
        return sorted(result, key=lambda x: -x[1])

    def get_disliked_attrs(self, atype: Optional[str] = None) -> List[Tuple[str, float]]:
        """
        返回负向偏好属性列表，按权重绝对值降序。
        """
        result = [
            (k.split(":", 1)[1], v)
            for k, v in self.explicit_lib.items()
            if v < -self.threshold and (atype is None or k.startswith(atype + ":"))
        ]
        return sorted(result, key=lambda x: x[1])

    @property
    def uncertain_attrs(self) -> List[str]:
        """
        自动派生：显性库中 |weight| ∈ (threshold, uncertain_thresh) 的属性，
        表示已观察到但置信度不足，可用于引导下一轮追问。
        """
        return [
            k.split(":", 1)[1]
            for k, v in self.explicit_lib.items()
            if self.threshold < abs(v) < self.uncertain_thresh
        ]

    @property
    def current_mood(self) -> str:
        """最近一轮的 mood label。"""
        if not self.mood_history:
            return "Neutral"
        return _etau_to_mood(self.mood_history[-1])

    @property
    def avg_mood(self) -> str:
        """整体对话的平均 mood label（用于 dialogue style 推断）。"""
        if not self.mood_history:
            return "Neutral"
        return _etau_to_mood(sum(self.mood_history) / len(self.mood_history))

    def get_rejected_movies(self) -> List[str]:
        """从 feedback_log 中取出所有 Reject 的电影名。"""
        return [
            r["item"] for r in self.feedback_log if r["feedback_type"] == "Reject"
        ]

    def get_accepted_movies(self) -> List[str]:
        """从 feedback_log 中取出所有 Accept 的电影名。"""
        return [
            r["item"] for r in self.feedback_log if r["feedback_type"] == "Accept"
        ]

    def get_implicit_vector(self) -> torch.Tensor:
        """返回隐性偏好向量（clone，外部不会影响内部状态）。"""
        return self.implicit_lib.clone()

    def get_explicit_vector(self, attr2id: Dict[str, int], total_attrs: int) -> torch.Tensor:
        """
        将显性库转为稠密向量（供模型 embedding 输入）。
        attr2id: "Type:value" -> index 映射
        """
        vec = torch.zeros(total_attrs, device=self.device)
        for key, weight in self.explicit_lib.items():
            if key in attr2id:
                vec[attr2id[key]] = weight
        return vec

    # ── Prompt 构造 ───────────────────────────────────────────────────────────

    def build_profile_prompt(
        self,
        max_liked: int = 5,
        max_disliked: int = 3,
        max_rejected: int = 3,
    ) -> str:
        """
        将双库信息转为自然语言 Prompt 片段，注入 UniCRS 的 Prompt。

        输出示例：
          [USER_PROFILE] Genre: comedy(+), horror(-) | Actor: Adam Sandler(+) |
                         Reject: The Dark Knight | Mood: Happy

        Parameters
        ----------
        max_liked    : 输出的正向属性数量上限（避免 Prompt 过长）
        max_disliked : 输出的负向属性数量上限
        max_rejected : 输出的 Reject 电影数量上限
        """
        parts = []

        # ① 按类型聚合显性偏好
        type_order = ["Genre", "Actor", "Director", "Movie", "Keyword"]
        liked_by_type: Dict[str, List[str]] = {t: [] for t in type_order}
        disliked_by_type: Dict[str, List[str]] = {t: [] for t in type_order}

        for key, weight in sorted(
            self.explicit_lib.items(), key=lambda x: -abs(x[1])
        ):
            atype, val = key.split(":", 1)
            if atype not in liked_by_type:
                continue
            if weight > self.threshold:
                liked_by_type[atype].append(f"{val}(+)")
            elif weight < -self.threshold:
                # Reject 强负值单独在 ③ 里输出，这里只输出普通负向
                if weight > self.reject_weight + 0.05:
                    disliked_by_type[atype].append(f"{val}(-)")

        # 合并 liked + disliked，按 type_order 顺序，总量不超限
        liked_total = 0
        for atype in type_order:
            items = liked_by_type[atype] + disliked_by_type[atype]
            if items and liked_total < max_liked + max_disliked:
                budget = max_liked + max_disliked - liked_total
                chunk  = items[:budget]
                parts.append(f"{atype}: {', '.join(chunk)}")
                liked_total += len(chunk)

        # ② Reject 黑名单（来自 feedback_log）
        rejected = self.get_rejected_movies()[:max_rejected]
        if rejected:
            parts.append(f"Reject: {', '.join(rejected)}")

        # ③ Mood
        parts.append(f"Mood: {self.current_mood}")

        if not any(p for p in parts if not p.startswith("Mood:")):
            # 显性库和 reject 都为空，只剩 Mood 没有价值
            return ""

        return "[USER_PROFILE] " + " | ".join(parts)

    # ── 序列化 / 反序列化 ─────────────────────────────────────────────────────

    def save(self, path: str):
        """保存完整状态到 JSON 文件（用于跨 session 持久化）。"""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        state = {
            "alpha":            self.alpha,
            "threshold":        self.threshold,
            "uncertain_thresh": self.uncertain_thresh,
            "reject_weight":    self.reject_weight,
            "emb_dim":          self.emb_dim,
            "explicit_lib":     self.explicit_lib,
            "implicit_lib":     self.implicit_lib.cpu().tolist(),
            "feedback_log":     self.feedback_log,
            "mood_history":     self.mood_history,
            "_turn":            self._turn,
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(state, f, ensure_ascii=False, indent=2)

    @classmethod
    def load(cls, path: str, device=None) -> "FIUPManager":
        """从 JSON 文件恢复完整状态。"""
        with open(path, "r", encoding="utf-8") as f:
            state = json.load(f)
        mgr = cls(
            emb_dim          = state["emb_dim"],
            alpha            = state["alpha"],
            threshold        = state["threshold"],
            uncertain_thresh = state.get("uncertain_thresh", 0.3),
            reject_weight    = state.get("reject_weight", -0.9),
            device           = device,
        )
        mgr.explicit_lib  = state["explicit_lib"]
        mgr.implicit_lib  = torch.tensor(state["implicit_lib"], device=mgr.device)
        mgr.feedback_log  = state.get("feedback_log", [])
        mgr.mood_history  = state.get("mood_history", [])
        mgr._turn         = state.get("_turn", 0)
        return mgr

    def reset(self):
        """重置为空状态（新用户 / 新会话）。"""
        self.explicit_lib = {}
        self.implicit_lib = torch.zeros(self.emb_dim, device=self.device)
        self.feedback_log = []
        self.mood_history = []
        self._turn        = 0

    def __repr__(self):
        return (
            f"FIUPManager(turn={self._turn}, α={self.alpha}, "
            f"explicit={len(self.explicit_lib)} attrs, "
            f"feedback={len(self.feedback_log)}, "
            f"mood={self.current_mood}, "
            f"implicit_norm={self.implicit_lib.norm():.3f})"
        )


# ── 快速测试 ──────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import torch

    mgr = FIUPManager(emb_dim=768, alpha=0.8)

    # 模拟五轮对话
    turns = [
        # (attributes,         uris,  e_tau, feedback_items)
        (["comedy", "Adam Sandler (2001)"],
         ["", "<http://dbpedia.org/resource/Adam_Sandler_(actor)>"],
         0.85, None),

        (["horror", "The Dark Knight"],
         ["", ""],
         -0.6, [("The Dark Knight", "Reject")]),

        (["comedy", "Jim Carrey"],
         ["", "<http://dbpedia.org/resource/Jim_Carrey_(actor)>"],
         0.7, [("The Mask (1994)", "Accept")]),

        (["romance"],
         [""],
         -0.3, None),

        (["action", "thriller"],
         ["", ""],
         0.5, None),
    ]

    for attrs, uris, e, fb in turns:
        emb = torch.randn(768)
        mgr.update_profile(attrs, e, emb, attr_uris=uris, feedback_items=fb)
        print(mgr)
        print("Prompt:", mgr.build_profile_prompt())
        print("Uncertain:", mgr.uncertain_attrs)
        print()