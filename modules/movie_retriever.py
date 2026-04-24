# src/modules/movie_retriever.py
"""
电影描述 RAG 检索模块（RAG 位置一）

功能：
  1. 离线从 DBpedia 子图构建电影结构化描述索引
     格式："{电影名}. Genre: {类型}. Director: {导演}. Stars: {主演}. Writer: {编剧}"
  2. 在线检索：给定当前对话中出现的实体 ID 列表，
     通过共享属性（导演/演员/类型/编剧）找到语义最相关的电影描述，
     返回 top-K 个描述文本，拼入 Prompt 供模型参考。

设计原则：
  - 纯结构化检索，不依赖任何额外模型或向量数据库
  - 与 KGExpander 共享反向索引逻辑，内存中只建一次
  - 对话中直接提到的电影不重复检索（避免冗余）

注入 Prompt 示例：
  [RETRIEVED] Brüno: comedy. Director: Larry Charles. Stars: Sacha Baron Cohen.
              Ali G Indahouse: comedy. Director: Mark Mylod. Stars: Sacha Baron Cohen.
"""

import json
import os
from collections import defaultdict, Counter
from typing import List, Dict, Optional


# 用于生成描述的关系（有序，决定描述文本的拼接顺序）
_DESC_RELS = ('genre', 'director', 'starring', 'writer')
_MAX_VALS  = {'genre': 2, 'director': 1, 'starring': 3, 'writer': 1}

# 用于相似度检索的关系（高质量语义关系，和 KGExpander 保持一致）
_RETRIEVAL_RELS = frozenset(_DESC_RELS)


def _clean_uri(uri: str) -> str:
    """DBpedia URI → 可读名称"""
    return uri.replace('<http://dbpedia.org/resource/', '').rstrip('>').replace('_', ' ')


class MovieRetriever:
    """
    基于 DBpedia 子图的电影描述检索器。

    Parameters
    ----------
    kg_path          : dbpedia_subkg.json 路径
    entity2id_path   : entity2id.json 路径
    relation2id_path : relation2id.json 路径
    item_ids_path    : item_ids.json 路径
    top_k            : 每次检索返回的描述数量（默认 2）
    """

    def __init__(
        self,
        kg_path: str,
        entity2id_path: str,
        relation2id_path: str,
        item_ids_path: str,
        top_k: int = 2,
    ):
        self.top_k = top_k

        # ── 加载原始数据 ──────────────────────────────────────────────────────
        with open(kg_path, 'r', encoding='utf-8') as f:
            self._kg: Dict = json.load(f)

        with open(entity2id_path, 'r', encoding='utf-8') as f:
            e2id = json.load(f)
        self._id2e: Dict[int, str] = {v: k for k, v in e2id.items()}

        with open(relation2id_path, 'r', encoding='utf-8') as f:
            r2id = json.load(f)

        with open(item_ids_path, 'r', encoding='utf-8') as f:
            self._item_ids = set(json.load(f))

        # ── 筛选有效关系 ──────────────────────────────────────────────────────
        self._rel_id_to_name: Dict[int, str] = {}
        for uri, rid in r2id.items():
            rel_name = uri.replace('<http://dbpedia.org/ontology/', '').rstrip('>')
            if rel_name in _RETRIEVAL_RELS:
                self._rel_id_to_name[rid] = rel_name

        self._retrieval_rel_ids = frozenset(self._rel_id_to_name.keys())

        # ── 离线构建描述索引 ──────────────────────────────────────────────────
        self._descriptions: Dict[int, str] = self._build_descriptions()

        # ── 离线构建反向索引：中间节点 → 电影集合 ───────────────────────────
        # nb_to_movies[nb_id] = {movie_id, ...}
        self._nb_to_movies: Dict[int, set] = defaultdict(set)
        for eid_str, triples in self._kg.items():
            eid = int(eid_str)
            if eid not in self._item_ids:
                continue
            for triple in triples:
                rel_id, nb_id = triple[0], triple[1]
                if rel_id in self._retrieval_rel_ids and nb_id not in self._item_ids:
                    self._nb_to_movies[nb_id].add(eid)

    # ── 核心接口 ──────────────────────────────────────────────────────────────

    def retrieve(self, entity_ids: List[int]) -> str:
        """
        给定对话中出现的实体 ID，检索最相关的 top-K 电影描述。

        Parameters
        ----------
        entity_ids : 当前对话的 entity 字段（实体 ID 列表）

        Returns
        -------
        注入 Prompt 的字符串，格式：
          "[RETRIEVED] {desc1} | {desc2}"
        若无相关描述则返回空字符串。
        """
        if not entity_ids:
            return ''

        query_set = set(entity_ids)
        scores: Counter = Counter()

        for eid in entity_ids:
            eid_str = str(eid)
            if eid_str not in self._kg:
                continue
            for triple in self._kg[eid_str]:
                rel_id, nb_id = triple[0], triple[1]
                if rel_id not in self._retrieval_rel_ids:
                    continue
                if nb_id in self._item_ids:
                    continue
                # 通过中间节点找相关电影，按共享属性数量计分
                for movie_id in self._nb_to_movies.get(nb_id, set()):
                    if movie_id not in query_set and movie_id in self._descriptions:
                        scores[movie_id] += 1

        if not scores:
            return ''

        top_movies = [mid for mid, _ in scores.most_common(self.top_k)]
        descs = [self._descriptions[mid] for mid in top_movies if mid in self._descriptions]

        if not descs:
            return ''

        return '[RETRIEVED] ' + ' | '.join(descs)

    def get_description(self, entity_id: int) -> str:
        """返回单个电影的描述文本，不存在则返回空字符串。"""
        return self._descriptions.get(entity_id, '')

    # ── 内部方法 ──────────────────────────────────────────────────────────────

    def _build_descriptions(self) -> Dict[int, str]:
        """
        为所有候选电影构建结构化描述文本。

        格式："{电影名}. Genre: X, Y. Director: Z. Stars: A, B, C. Writer: W"
        """
        descriptions = {}

        for eid in self._item_ids:
            eid_str = str(eid)
            name = _clean_uri(self._id2e.get(eid, str(eid)))

            # 去掉 "(film)" / "(2018 film)" 等后缀，让名字更自然
            import re
            name = re.sub(r'\s*\(.*?film.*?\)\s*$', '', name).strip()

            parts: Dict[str, List[str]] = defaultdict(list)
            if eid_str in self._kg:
                for triple in self._kg[eid_str]:
                    rel_name = self._rel_id_to_name.get(triple[0])
                    if rel_name:
                        nb_name = _clean_uri(self._id2e.get(triple[1], ''))
                        # 去掉括号说明，让名字更简洁
                        nb_name = re.sub(r'\s*\(.*?\)\s*$', '', nb_name).strip()
                        if nb_name:
                            parts[rel_name].append(nb_name)

            if not parts:
                descriptions[eid] = name
                continue

            desc = name
            for rel in _DESC_RELS:
                if rel in parts:
                    vals = parts[rel][:_MAX_VALS[rel]]
                    label = rel.capitalize() if rel != 'starring' else 'Stars'
                    desc += f'. {label}: {", ".join(vals)}'

            descriptions[eid] = desc

        return descriptions

    def stats(self) -> dict:
        """返回索引统计信息，用于日志输出。"""
        covered = sum(1 for d in self._descriptions.values() if '.' in d)
        return {
            'total_movies':    len(self._descriptions),
            'with_attributes': covered,
            'coverage':        f'{covered / max(len(self._descriptions), 1) * 100:.1f}%',
            'bridge_nodes':    len(self._nb_to_movies),
            'top_k':           self.top_k,
        }