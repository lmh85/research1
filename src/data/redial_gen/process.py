# src/data/redial/process.py  —— FIUP 完整版
"""
在原版 process.py 基础上新增三个字段写入 *_data_processed.jsonl：
  entity_names : List[str]
      格式 "Type:value"，如 "Genre:comedy" / "Actor:Adam Sandler" / "Movie:Super Troopers (2001)"
      由 movie_name（优先）+ 过滤后的 entity_name（补充类型/演员名）组成
      与 FIUPManager.infer_attr_type() 的分类逻辑完全对应

  user_id  : str  — 对话发起者 ID（initiatorWorkerId）
  conv_id  : str  — 对话编号（conversationId）

原版字段（context / resp / rec / entity）逻辑不变。
"""

import json
import re
import html
from tqdm.auto import tqdm


# ── 噪声过滤 ──────────────────────────────────────────────────────────────────
NOISE_WORDS = {
    "you", "not", "there", "it", "i", "we", "in", "movies", "movie",
    "remember", "who", "loved", "ill", "rock", "is", "a", "the",
    "film", "films", "show", "series", "omg", "check", "sounds",
    "enjoy", "time", "plays", "play", "loves", "theaters", "theater",
    "newer", "classic", "dark", "adults", "house", "america", "netflix",
    "something", "anything", "some", "one", "good", "great", "nice",
    "what", "ok", "okay", "yes", "no", "well",
}

# entity_name 里的风格/题材关键词，归为 Genre 类
GENRE_KEYWORDS = {
    "action", "comedy", "comedies", "horror", "thriller", "drama", "dramas",
    "romance", "romantic", "musical", "musicals", "sci-fi", "science fiction",
    "sci fi", "fantasy", "western", "documentary", "animation", "animated",
    "adventure", "mystery", "crime", "sports", "dark comedy", "superhero",
    "suspense", "family",
}

# DBpedia URI 后缀 → 属性类型
URI_SUFFIX_TO_TYPE = {
    "actor":         "Actor",
    "actress":       "Actor",
    "director":      "Director",
    "filmmaker":     "Director",
    "producer":      "Director",
    "film_producer": "Director",
    "screenwriter":  "Keyword",
    "composer":      "Keyword",
    "film_editor":   "Keyword",
    "writer":        "Keyword",
}

movie_pattern = re.compile(r"@\d+")


def _uri_to_type(uri: str) -> str:
    """从 DBpedia URI 后缀推断属性类型。"""
    r = uri.replace("<http://dbpedia.org/resource/", "").rstrip(">")
    if "(" in r:
        suffix = r[r.rfind("(") + 1: r.rfind(")")]
        if suffix == "film" or re.match(r"\d{4}_film", suffix):
            return "Movie"
        for key, atype in URI_SUFFIX_TO_TYPE.items():
            if key in suffix:
                return atype
    return ""  # 无法判断


def _name_to_type(name: str) -> str:
    """从实体名本身推断类型。"""
    nl = name.strip().lower()
    if nl in GENRE_KEYWORDS or any(g in nl for g in GENRE_KEYWORDS):
        return "Genre"
    if re.search(r"\(\d{4}\)", name):
        return "Movie"
    # 大写开头且多词，认为是人名 → Keyword（演员名通过 URI 更准确）
    return "Keyword"


def _make_entity_name(name: str, uri: str = "") -> str:
    """
    返回带类型前缀的实体名，如 "Genre:comedy" / "Actor:Adam Sandler"。
    """
    if uri:
        atype = _uri_to_type(uri)
        if atype:
            clean = name.strip()
            return f"{atype}:{clean}"
    atype = _name_to_type(name)
    return f"{atype}:{name.strip()}"


def is_clean(name: str) -> bool:
    return name.strip().lower() not in NOISE_WORDS and len(name.strip()) > 1


def process_utt(utt, movieid2name, replace_movieId):
    def convert(match):
        movieid = match.group(0)[1:]
        if movieid in movieid2name:
            return " ".join(movieid2name[movieid].split())
        return match.group(0)

    if replace_movieId:
        utt = re.sub(movie_pattern, convert, utt)
    utt = " ".join(utt.split())
    utt = html.unescape(utt)
    return utt


def process(data_file, out_file, movie_set):
    with open(data_file, "r", encoding="utf-8") as fin, \
         open(out_file, "w", encoding="utf-8") as fout:

        for line in tqdm(fin):
            dialog = json.loads(line)
            if not dialog["messages"]:
                continue

            movieid2name = dialog["movieMentions"]
            user_id = str(dialog.get("initiatorWorkerId", "unknown"))
            conv_id = str(dialog.get("conversationId",   "unknown"))

            context, resp = [], ""
            entity_list   = []

            # 每轮新增的 "Type:value" 实体名，随轮次累积
            context_entity_names_list: list = []

            messages = dialog["messages"]
            turn_i = 0

            while turn_i < len(messages):
                worker_id  = messages[turn_i]["senderWorkerId"]
                utt_turn   = []
                entity_turn = []
                movie_turn  = []
                entity_names_turn: list = []  # 本轮新增

                turn_j = turn_i
                while turn_j < len(messages) and \
                      messages[turn_j]["senderWorkerId"] == worker_id:

                    msg = messages[turn_j]
                    utt = process_utt(msg["text"], movieid2name, replace_movieId=True)
                    utt_turn.append(utt)

                    # ── 原版：收集实体/电影 ID ──────────────────────────────
                    entity_ids = [
                        entity2id[e] for e in msg["entity"] if e in entity2id
                    ]
                    entity_turn.extend(entity_ids)

                    movie_ids = [
                        entity2id[m] for m in msg["movie"] if m in entity2id
                    ]
                    movie_turn.extend(movie_ids)

                    # ── 新增：收集带类型前缀的实体名 ────────────────────────
                    # ① movie_name 优先（带年份，格式干净）
                    for mn in msg.get("movie_name", []):
                        entity_names_turn.append(f"Movie:{mn.strip()}")

                    # ② entity URI（类型准确）
                    uri_name_pairs = list(zip(
                        msg.get("entity_name", []),
                        msg.get("entity", []),
                    ))
                    for en, uri in uri_name_pairs:
                        if not is_clean(en):
                            continue
                        typed = _make_entity_name(en, uri)
                        entity_names_turn.append(typed)

                    # ③ entity_name 中没有对应 URI 的剩余部分（entity_name 比 entity 多时）
                    extra_names = msg.get("entity_name", [])[len(uri_name_pairs):]
                    for en in extra_names:
                        if is_clean(en):
                            entity_names_turn.append(_make_entity_name(en))

                    turn_j += 1

                utt  = " ".join(utt_turn)
                resp = utt

                # ── 原版：累积上下文实体 ID ──────────────────────────────────
                context_entity_list = [e for el in entity_list for e in el]
                context_entity_list_extend = list(set(context_entity_list))

                # ── 新增：累积上下文实体名（去重、保序）────────────────────
                seen = set()
                context_entity_names = []
                for name in [n for nl in context_entity_names_list for n in nl]:
                    if name not in seen:
                        seen.add(name)
                        context_entity_names.append(name)

                if not context:
                    context.append("")

                turn = {
                    # ── 原版字段（完全不变）──────────────────────────────────
                    "context": context,
                    "resp":    resp,
                    "rec":     list(set(movie_turn + entity_turn)),
                    "entity":  context_entity_list_extend,
                    # ── 新增字段 ─────────────────────────────────────────────
                    "entity_names": context_entity_names,
                    "user_id":      user_id,
                    "conv_id":      conv_id,
                }
                fout.write(json.dumps(turn, ensure_ascii=False) + "\n")

                context.append(resp)
                entity_list.append(movie_turn + entity_turn)
                context_entity_names_list.append(entity_names_turn)
                movie_set |= set(movie_turn)

                turn_i = turn_j


if __name__ == "__main__":
    with open("entity2id.json", "r", encoding="utf-8") as f:
        entity2id = json.load(f)
    item_set = set()

    process("valid_data_dbpedia.jsonl", "valid_data_processed.jsonl", item_set)
    process("test_data_dbpedia.jsonl",  "test_data_processed.jsonl",  item_set)
    process("train_data_dbpedia.jsonl", "train_data_processed.jsonl", item_set)

    with open("item_ids.json", "w", encoding="utf-8") as f:
        json.dump(list(item_set), f, ensure_ascii=False)
    print(f"#item: {len(item_set)}")


# import json
# import re

# import html
# from tqdm.auto import tqdm

# movie_pattern = re.compile(r'@\d+')


# def process_utt(utt, movieid2name, replace_movieId):
#     def convert(match):
#         movieid = match.group(0)[1:]
#         if movieid in movieid2name:
#             movie_name = movieid2name[movieid]
#             movie_name = ' '.join(movie_name.split())
#             return movie_name
#         else:
#             return match.group(0)

#     if replace_movieId:
#         utt = re.sub(movie_pattern, convert, utt)
#     utt = ' '.join(utt.split())
#     utt = html.unescape(utt)

#     return utt


# def process(data_file, out_file, movie_set):
#     with open(data_file, 'r', encoding='utf-8') as fin, open(out_file, 'w', encoding='utf-8') as fout:
#         for line in tqdm(fin):
#             dialog = json.loads(line)
#             if len(dialog['messages']) == 0:
#                 continue

#             movieid2name = dialog['movieMentions']
#             user_id, resp_id = dialog['initiatorWorkerId'], dialog['respondentWorkerId']
#             context, resp = [], ''
#             entity_list = []
#             messages = dialog['messages']
#             turn_i = 0
#             while turn_i < len(messages):
#                 worker_id = messages[turn_i]['senderWorkerId']
#                 utt_turn = []
#                 entity_turn = []
#                 movie_turn = []

#                 turn_j = turn_i
#                 while turn_j < len(messages) and messages[turn_j]['senderWorkerId'] == worker_id:
#                     utt = process_utt(messages[turn_j]['text'], movieid2name, replace_movieId=True)
#                     utt_turn.append(utt)

#                     entity_ids = [entity2id[entity] for entity in messages[turn_j]['entity'] if entity in entity2id]
#                     entity_turn.extend(entity_ids)

#                     movie_ids = [entity2id[movie] for movie in messages[turn_j]['movie'] if movie in entity2id]
#                     movie_turn.extend(movie_ids)

#                     turn_j += 1

#                 utt = ' '.join(utt_turn)

#                 # if worker_id == user_id:
#                 #     context.append(utt)
#                 #     entity_list.append(entity_turn + movie_turn)
#                 # else:
#                 resp = utt

#                 context_entity_list = [entity for entity_l in entity_list for entity in entity_l]
#                 context_entity_list_extend = []
#                 # entity_links = [id2entity[id] for id in context_entity_list if id in id2entity]
#                 # for entity in entity_links:
#                 #     if entity in node2entity:
#                 #         for e in node2entity[entity]['entity']:
#                 #             if e in entity2id:
#                 #                 context_entity_list_extend.append(entity2id[e])
#                 context_entity_list_extend += context_entity_list
#                 context_entity_list_extend = list(set(context_entity_list_extend))

#                 if len(context) == 0:
#                     context.append('')
#                 turn = {
#                     'context': context,
#                     'resp': resp,
#                     'rec': list(set(movie_turn + entity_turn)),
#                     'entity': context_entity_list_extend,
#                 }
#                 fout.write(json.dumps(turn, ensure_ascii=False) + '\n')

#                 context.append(resp)
#                 entity_list.append(movie_turn + entity_turn)
#                 movie_set |= set(movie_turn)

#                 turn_i = turn_j


# if __name__ == '__main__':
#     with open('entity2id.json', 'r', encoding='utf-8') as f:
#         entity2id = json.load(f)
#     item_set = set()
#     # with open('node2text_link_clean.json', 'r', encoding='utf-8') as f:
#     #     node2entity = json.load(f)

#     process('valid_data_dbpedia.jsonl', 'valid_data_processed.jsonl', item_set)
#     process('test_data_dbpedia.jsonl', 'test_data_processed.jsonl', item_set)
#     process('train_data_dbpedia.jsonl', 'train_data_processed.jsonl', item_set)

#     with open('item_ids.json', 'w', encoding='utf-8') as f:
#         json.dump(list(item_set), f, ensure_ascii=False)
#     print(f'#item: {len(item_set)}')
