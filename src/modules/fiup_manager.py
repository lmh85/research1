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
        self.alpha = alpha
        self.threshold = threshold
        self.emb_dim = emb_dim
        self.device = device or torch.device("cpu")
        self.explicit_lib: Dict[str, float] = {}
        self.implicit_lib: torch.Tensor = torch.zeros(emb_dim, device=self.device)

    def update_profile(self, attributes: List[str], e_tau: float, context_emb: torch.Tensor):
        self._update_explicit(attributes, e_tau)
        self._update_implicit(context_emb, e_tau)

    def _update_explicit(self, attributes: List[str], e_tau: float):
        strength = abs(e_tau)
        for attr in attributes:
            prev = self.explicit_lib.get(attr, 0.0)
            new_weight = self.alpha * prev + (1 - self.alpha) * strength
            if e_tau < 0:
                new_weight = -abs(new_weight)
            self.explicit_lib[attr] = round(new_weight, 6)

    def _update_implicit(self, context_emb: torch.Tensor, e_tau: float):
        context_emb = context_emb.to(self.device).detach().float()
        delta = (1 - self.alpha) * (context_emb * e_tau)
        self.implicit_lib = self.alpha * self.implicit_lib + delta

    def get_liked_attrs(self) -> List[str]:
        return [k for k, v in self.explicit_lib.items() if v > self.threshold]

    def get_disliked_attrs(self) -> List[str]:
        return [k for k, v in self.explicit_lib.items() if v < -self.threshold]

    def get_implicit_vector(self) -> torch.Tensor:
        return self.implicit_lib.clone()

    def get_explicit_vector(self, attr2id: Dict[str, int], total_attrs: int) -> torch.Tensor:
        vec = torch.zeros(total_attrs, device=self.device)
        for attr, weight in self.explicit_lib.items():
            if attr in attr2id:
                vec[attr2id[attr]] = weight
        return vec

    def build_profile_prompt(self) -> str:
        """
        将双库信息转为自然语言 Prompt 片段，注入对话 Prompt。

        Movie 级别偏好优先，Genre 次之，按权重绝对值降序排列。
        喜欢最多3个，不喜欢最多2个，避免信息过载。

        示例输出：
          "User likes The Dark Knight, comedy; dislikes romance"
        """
        liked    = self.get_liked_attrs()
        disliked = self.get_disliked_attrs()

        # 按权重绝对值降序
        liked_sorted    = sorted(liked,    key=lambda x: abs(self.explicit_lib.get(x, 0)), reverse=True)
        disliked_sorted = sorted(disliked, key=lambda x: abs(self.explicit_lib.get(x, 0)), reverse=True)

        # Movie 和 Genre 分开，让输出更自然
        like_movies = [k.replace('Movie:', '') for k in liked_sorted    if k.startswith('Movie:')][:2]
        like_genres = [k.replace('Genre:', '') for k in liked_sorted    if k.startswith('Genre:')][:2]
        dis_movies  = [k.replace('Movie:', '') for k in disliked_sorted if k.startswith('Movie:')][:1]
        dis_genres  = [k.replace('Genre:', '') for k in disliked_sorted if k.startswith('Genre:')][:1]
        # 兜底：无前缀的属性也保留
        like_others = [k for k in liked_sorted    if not k.startswith('Movie:') and not k.startswith('Genre:')][:1]
        dis_others  = [k for k in disliked_sorted if not k.startswith('Movie:') and not k.startswith('Genre:')][:1]

        likes_list    = like_movies + like_genres + like_others
        dislikes_list = dis_movies  + dis_genres  + dis_others

        if not likes_list and not dislikes_list:
            return ""

        parts = []
        if likes_list:
            parts.append(f"likes {', '.join(likes_list)}")
        if dislikes_list:
            parts.append(f"dislikes {', '.join(dislikes_list)}")

        return "User " + "; ".join(parts)

    def save(self, path: str):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        state = {
            "alpha":        self.alpha,
            "threshold":    self.threshold,
            "emb_dim":      self.emb_dim,
            "explicit_lib": self.explicit_lib,
            "implicit_lib": self.implicit_lib.cpu().tolist(),
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(state, f, ensure_ascii=False, indent=2)

    @classmethod
    def load(cls, path: str, device=None) -> "FIUPManager":
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
        self.explicit_lib = {}
        self.implicit_lib = torch.zeros(self.emb_dim, device=self.device)

    def __repr__(self):
        return (
            f"FIUPManager(α={self.alpha}, "
            f"explicit={len(self.explicit_lib)} attrs, "
            f"implicit_norm={self.implicit_lib.norm():.3f})"
        )


if __name__ == "__main__":
    mgr = FIUPManager(emb_dim=768, alpha=0.8)
    turns = [
        (["Movie:The Dark Knight (2008)", "Genre:action", "Genre:thriller"], 0.9,  torch.randn(768)),
        (["Genre:romance"],                                                   -0.7, torch.randn(768)),
        (["Movie:Inception (2010)", "Genre:comedy"],                          0.5,  torch.randn(768)),
    ]
    for attrs, e, emb in turns:
        mgr.update_profile(attrs, e, emb)
        print(mgr)
        print("Prompt:", mgr.build_profile_prompt())
        print()
        