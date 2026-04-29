# src/modules/fiup_manager.py
"""
FIUP (Fine-grained Incremental User Profile) Manager
实现论文中"双库"动态更新逻辑：
  - 显性库 explicit_lib：属性 ID → 权重（可正可负）
  - 隐性库 implicit_lib：对话语义偏好向量
"""

import torch
import json
import os
from typing import Dict, List, Optional


class FIUPManager:
    def __init__(
        self,
        emb_dim: int,
        alpha: float = 0.8,
        threshold: float = 0.1,
        device: Optional[torch.device] = None,
    ):
        """
        emb_dim:   隐性库向量维度，与 RoBERTa hidden size 一致（通常 768）
        alpha:     遗忘衰减系数，越大历史越重要
        threshold: 显性库过滤阈值，仅保留 |weight| > threshold 的属性
        device:    计算设备
        """
        self.alpha = alpha
        self.threshold = threshold
        self.emb_dim = emb_dim
        self.device = device or torch.device("cpu")

        # 显性库：字典，key = 属性名(str) 或属性ID(int)，value = 权重(float)
        self.explicit_lib: Dict[str, float] = {}

        # 隐性库：向量，shape = [emb_dim]
        self.implicit_lib: torch.Tensor = torch.zeros(emb_dim, device=self.device)

    # ── 核心更新 ──────────────────────────────────────────────────────────────

    def update_profile(
        self,
        attributes: List[str],
        e_tau: float,
        context_emb: torch.Tensor,
    ):
        """
        单轮对话后更新双库。

        attributes:  本轮识别出的实体/属性名列表
        e_tau:       情感分值 ∈ [-1, 1]
        context_emb: 当前 [CLS] 向量，shape = [emb_dim]
        """
        self._update_explicit(attributes, e_tau)
        self._update_implicit(context_emb, e_tau)

    def _update_explicit(self, attributes: List[str], e_tau: float):
        """
        显性库更新公式：
          strength = |e_τ|
          w_i,τ = α * w_i,τ-1 + (1-α) * strength
          若 e_τ < 0，则取负（表达排斥）
        """
        strength = abs(e_tau)
        for attr in attributes:
            prev = self.explicit_lib.get(attr, 0.0)
            new_weight = self.alpha * prev + (1 - self.alpha) * strength
            # 情感方向控制：负向情感使权重为负
            if e_tau < 0:
                new_weight = -abs(new_weight)
            self.explicit_lib[attr] = round(new_weight, 6)

    def _update_implicit(self, context_emb: torch.Tensor, e_tau: float):
        """
        隐性库更新公式：
          I_τ = α * I_τ-1 + (1-α) * (context_emb * e_τ)
        正向情感拉近，负向情感推开。
        """
        context_emb = context_emb.to(self.device).detach().float()
        delta = (1 - self.alpha) * (context_emb * e_tau)
        self.implicit_lib = self.alpha * self.implicit_lib + delta

    # ── 查询接口 ──────────────────────────────────────────────────────────────

    def get_liked_attrs(self) -> List[str]:
        """返回用户喜欢的属性（权重 > threshold）"""
        return [k for k, v in self.explicit_lib.items() if v > self.threshold]

    def get_disliked_attrs(self) -> List[str]:
        """返回用户不喜欢的属性（权重 < -threshold）"""
        return [k for k, v in self.explicit_lib.items() if v < -self.threshold]

    def get_implicit_vector(self) -> torch.Tensor:
        """返回隐性偏好向量"""
        return self.implicit_lib.clone()

    def get_explicit_vector(self, attr2id: Dict[str, int], total_attrs: int) -> torch.Tensor:
        """
        将显性库转为稠密向量（用于模型输入）
        attr2id: 属性名 → 索引的映射
        total_attrs: 属性总数
        """
        vec = torch.zeros(total_attrs, device=self.device)
        for attr, weight in self.explicit_lib.items():
            if attr in attr2id:
                vec[attr2id[attr]] = weight
        return vec

    # ── Prompt 构造 ───────────────────────────────────────────────────────────
    def build_profile_prompt(self) -> str:
        # 只取前3个喜欢、前2个不喜欢，避免信息过载
        likes = ", ".join(self.likes[:3]) if self.likes else ""
        dislikes = ", ".join(self.dislikes[:2]) if self.dislikes else ""

        if likes and dislikes:
            return f"User likes {likes}; user dislikes {dislikes}"
        elif likes:
            return f"User likes {likes}"
        elif dislikes:
            return f"User dislikes {dislikes}"
        else:
            return ""
    # def build_profile_prompt(self) -> str:
    #     """
    #     将双库信息转为自然语言 Prompt 片段，插入上下文中。
    #     示例输出：
    #       "[User Profile] Likes: action, thriller | Dislikes: romance"
    #     """
    #     liked = self.get_liked_attrs()
    #     disliked = self.get_disliked_attrs()

    #     parts = []
    #     if liked:
    #         parts.append(f"Likes: {', '.join(liked)}")
    #     if disliked:
    #         parts.append(f"Dislikes: {', '.join(disliked)}")

    #     if not parts:
    #         return ""  # 尚无足够信息时返回空串

    #     return "[User Profile] " + " | ".join(parts)

    # ── 持久化 ────────────────────────────────────────────────────────────────

    def save(self, path: str):
        """保存状态到文件（用于多轮对话跨 session 保持）"""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        state = {
            "alpha": self.alpha,
            "threshold": self.threshold,
            "emb_dim": self.emb_dim,
            "explicit_lib": self.explicit_lib,
            "implicit_lib": self.implicit_lib.cpu().tolist(),
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(state, f, ensure_ascii=False, indent=2)

    @classmethod
    def load(cls, path: str, device=None) -> "FIUPManager":
        """从文件恢复状态"""
        with open(path, "r", encoding="utf-8") as f:
            state = json.load(f)
        manager = cls(
            emb_dim=state["emb_dim"],
            alpha=state["alpha"],
            threshold=state["threshold"],
            device=device,
        )
        manager.explicit_lib = state["explicit_lib"]
        manager.implicit_lib = torch.tensor(state["implicit_lib"], device=manager.device)
        return manager

    def reset(self):
        """重置（新用户 / 新会话）"""
        self.explicit_lib = {}
        self.implicit_lib = torch.zeros(self.emb_dim, device=self.device)

    def __repr__(self):
        return (
            f"FIUPManager(α={self.alpha}, "
            f"explicit={len(self.explicit_lib)} attrs, "
            f"implicit_norm={self.implicit_lib.norm():.3f})"
        )


# ── 快速测试 ──────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import torch

    mgr = FIUPManager(emb_dim=768, alpha=0.8)

    # 模拟三轮对话
    turns = [
        (["action", "thriller"], 0.9,  torch.randn(768)),
        (["romance"],            -0.7, torch.randn(768)),
        (["action", "comedy"],   0.5,  torch.randn(768)),
    ]
    for attrs, e, emb in turns:
        mgr.update_profile(attrs, e, emb)
        print(mgr)
        print("Prompt:", mgr.build_profile_prompt())
        print()