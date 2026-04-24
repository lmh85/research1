# src/modules/kg_expander.py
"""
KG 邻居扩展模块（RAG 位置二）

思路：
  当前对话上下文提到的实体（entity 字段）只有对话中直接出现的电影/实体。
  通过 2 跳 KG 检索，找到"语义相关"的候选电影加入 entity 列表，
  让 KGPrompt 的 cross-attention 能覆盖更丰富的图结构，提升 Recall@K。

2 跳路径：
  电影 A --[高质量关系]--> 中间节点（演员/导演/类型）
                         --> 中间节点 --[被哪些电影指向]--> 电影 B

只使用高质量语义关系（导演/演员/类型/编剧/作曲/解说），
过滤掉低质量关系（发行公司/国家/语言/地点），避免引入噪声。

使用方式：
  在 dataset_rec.py 的 prepare_data 里调用 expander.expand()，
  替换原始 entity 字段，其余流程不变。
"""

import json
import os
from collections import defaultdict
from typing import List, Set, Optional


# 高质量关系关键词（DBpedia ontology URI 中包含这些词的关系才使用）
_HIGH_QUALITY_KEYWORDS = (
    'genre', 'director', 'starring', 'composer',
    'writer', 'narrator', 'musicComposer',
)


class KGExpander:
    """
    基于 DBpedia 子图的 2 跳邻居扩展器。

    Parameters
    ----------
    kg_path        : dbpedia_subkg.json 路径
    relation2id_path : relation2id.json 路径
    item_ids_path  : item_ids.json 路径（推荐候选电影 ID 集合）
    max_expand     : 每个样本最多扩展多少个新实体（默认 16）
    """

    def __init__(
        self,
        kg_path: str,
        relation2id_path: str,
        item_ids_path: str,
        max_expand: int = 16,
    ):
        self.max_expand = max_expand

        # ── 加载数据 ──────────────────────────────────────────────────────────
        with open(kg_path, 'r', encoding='utf-8') as f:
            self._kg: dict = json.load(f)          # {str(entity_id): [[rel_id, nb_id], ...]}

        with open(relation2id_path, 'r', encoding='utf-8') as f:
            r2id: dict = json.load(f)

        with open(item_ids_path, 'r', encoding='utf-8') as f:
            self._item_ids: Set[int] = set(json.load(f))

        # ── 筛选高质量关系 ID ─────────────────────────────────────────────────
        self._hq_rels: frozenset = frozenset(
            v for k, v in r2id.items()
            if any(kw in k for kw in _HIGH_QUALITY_KEYWORDS)
        )

        # ── 预构建反向索引：中间节点 → 指向它的电影集合 ──────────────────────
        # nb_to_movies[nb_id] = {movie_id, ...}
        self._nb_to_movies: dict = defaultdict(set)
        for eid_str, triples in self._kg.items():
            eid = int(eid_str)
            if eid not in self._item_ids:
                continue
            for triple in triples:
                rel_id, nb_id = triple[0], triple[1]
                if rel_id in self._hq_rels and nb_id not in self._item_ids:
                    self._nb_to_movies[nb_id].add(eid)

    # ── 核心接口 ──────────────────────────────────────────────────────────────

    def expand(self, entity_ids: List[int]) -> List[int]:
        """
        对输入的实体 ID 列表做 2 跳扩展，返回扩展后的列表。

        扩展策略：
          1. 遍历 entity_ids 中的每个实体
          2. 通过高质量关系找到其中间节点（演员/导演/类型等）
          3. 通过反向索引找到同样连接该中间节点的其他电影
          4. 将这些电影加入扩展集合（去重、排除自身、限制上限）

        原始 entity_ids 中的非电影实体（演员/类型等）也保留，
        新增的实体追加在后面。

        Parameters
        ----------
        entity_ids : 原始实体 ID 列表（来自数据集的 entity 字段）

        Returns
        -------
        扩展后的实体 ID 列表，长度 = len(entity_ids) + min(新增数, max_expand)
        """
        if not entity_ids:
            return entity_ids

        original_set = set(entity_ids)
        candidates: dict = {}  # movie_id -> score（共享中间节点越多分越高）

        for eid in entity_ids:
            eid_str = str(eid)
            if eid_str not in self._kg:
                continue
            for triple in self._kg[eid_str]:
                rel_id, nb_id = triple[0], triple[1]
                if rel_id not in self._hq_rels:
                    continue
                if nb_id in self._item_ids:
                    continue
                # 通过中间节点找相关电影
                for movie_id in self._nb_to_movies.get(nb_id, set()):
                    if movie_id not in original_set:
                        candidates[movie_id] = candidates.get(movie_id, 0) + 1

        if not candidates:
            return entity_ids

        # 按共享中间节点数量降序排列，取 top-max_expand
        top_movies = sorted(candidates.items(), key=lambda x: -x[1])
        extra = [mid for mid, _ in top_movies[:self.max_expand]]

        return list(entity_ids) + extra

    # ── 统计信息 ──────────────────────────────────────────────────────────────

    def stats(self) -> dict:
        return {
            'kg_entities':    len(self._kg),
            'item_ids':       len(self._item_ids),
            'hq_relations':   len(self._hq_rels),
            'bridge_nodes':   len(self._nb_to_movies),
            'max_expand':     self.max_expand,
        }