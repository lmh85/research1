# src/data/redial/process_mask.py  —— FIUP 修改版
"""
在原版 process_mask.py 基础上新增三个字段写入 *_data_processed.jsonl：
  entity_names : List[str]
      格式 "Type:value"，如 "Genre:comedy" / "Actor:Adam Sandler" / "Movie:Super Troopers (2001)"
      只从 user（initiator）侧的对话轮次中提取，和原版 context 的来源保持一致
      由 movie_name（优先）+ 过滤后的 entity_name（补充类型/演员名）组成

  user_id  : str  — 对话发起者 ID（initiatorWorkerId）
  conv_id  : str  — 对话编号（conversationId）

原版字段（context / resp / rec / entity）逻辑完全不变。
"""

import json
import re
import html
from tqdm.auto import tqdm


# ── 噪声过滤（与 process.py 保持一致）────────────────────────────────────────
NOISE_WORDS = {
    "you", "not", "there", "it", "i", "we", "in", "movies", "movie",
    "remember", "who", "loved", "ill", "rock", "is", "a", "the",
    "film", "films", "show", "series", "omg", "check", "sounds",
    "enjoy", "time", "plays", "play", "loves", "theaters", "theater",
    "newer", "classic", "dark", "adults", "house", "america", "netflix",
    "something", "anything", "some", "one", "good", "great", "nice",
    "what", "ok", "okay", "yes", "no", "well",
}

GENRE_KEYWORDS = {
    "action", "comedy", "comedies", "horror", "thriller", "drama", "dramas",
    "romance", "romantic", "musical", "musicals", "sci-fi", "science fiction",
    "sci fi", "fantasy", "western", "documentary", "animation", "animated",
    "adventure", "mystery", "crime", "sports", "dark comedy", "superhero",
    "suspense", "family",
}

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
    r = uri.replace("<http://dbpedia.org/resource/", "").rstrip(">")
    if "(" in r:
        suffix = r[r.rfind("(") + 1: r.rfind(")")]
        if suffix == "film" or re.match(r"\d{4}_film", suffix):
            return "Movie"
        for key, atype in URI_SUFFIX_TO_TYPE.items():
            if key in suffix:
                return atype
    return ""


def _name_to_type(name: str) -> str:
    nl = name.strip().lower()
    if nl in GENRE_KEYWORDS or any(g in nl for g in GENRE_KEYWORDS):
        return "Genre"
    if re.search(r"\(\d{4}\)", name):
        return "Movie"
    return "Keyword"


def _make_entity_name(name: str, uri: str = "") -> str:
    if uri:
        atype = _uri_to_type(uri)
        if atype:
            return f"{atype}:{name.strip()}"
    return f"{_name_to_type(name)}:{name.strip()}"


def is_clean(name: str) -> bool:
    return name.strip().lower() not in NOISE_WORDS and len(name.strip()) > 1


def process_utt(utt, movieid2name, replace_movieId, remove_movie=False):
    def convert(match):
        movieid = match.group(0)[1:]
        if movieid in movieid2name:
            if remove_movie:
                return "<movie>"
            movie_name = movieid2name[movieid]
            return movie_name
        else:
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
            if len(dialog["messages"]) == 0:
                continue

            movieid2name = dialog["movieMentions"]

            # ── [FIUP] 对话级 ID ──────────────────────────────────────────────
            user_id = str(dialog.get("initiatorWorkerId", "unknown"))
            conv_id = str(dialog.get("conversationId",   "unknown"))
            # ────────────────────────────────────────────────────────────────

            # 原版变量
            initiator_id = dialog["initiatorWorkerId"]
            context, resp = [], ""
            entity_list = []

            # ── [FIUP] 用户侧实体名累积（只取 user/initiator 侧，与 context 来源一致）
            context_entity_names_list = []
            # ────────────────────────────────────────────────────────────────

            messages = dialog["messages"]
            turn_i = 0

            while turn_i < len(messages):
                worker_id = messages[turn_i]["senderWorkerId"]
                utt_turn      = []
                mask_utt_turn = []
                entity_turn   = []
                movie_turn    = []

                # ── [FIUP] 本轮实体名 ─────────────────────────────────────────
                entity_names_turn = []
                # ────────────────────────────────────────────────────────────

                turn_j = turn_i
                while turn_j < len(messages) and \
                      messages[turn_j]["senderWorkerId"] == worker_id:

                    msg = messages[turn_j]

                    # 原版：生成普通文本和 mask 文本
                    utt = process_utt(
                        msg["text"], movieid2name,
                        replace_movieId=True, remove_movie=False
                    )
                    utt_turn.append(utt)

                    mask_utt = process_utt(
                        msg["text"], movieid2name,
                        replace_movieId=True, remove_movie=True
                    )
                    mask_utt_turn.append(mask_utt)

                    # 原版：收集实体/电影 ID
                    entity_ids = [
                        entity2id[e] for e in msg["entity"] if e in entity2id
                    ]
                    entity_turn.extend(entity_ids)

                    movie_ids = [
                        entity2id[m] for m in msg["movie"] if m in entity2id
                    ]
                    movie_turn.extend(movie_ids)

                    # ── [FIUP] 收集实体名（只在 user/initiator 侧收集，与 context 一致）
                    if worker_id == initiator_id:
                        for mn in msg.get("movie_name", []):
                            entity_names_turn.append(f"Movie:{mn.strip()}")

                        uri_name_pairs = list(zip(
                            msg.get("entity_name", []),
                            msg.get("entity", []),
                        ))
                        for en, uri in uri_name_pairs:
                            if is_clean(en):
                                entity_names_turn.append(_make_entity_name(en, uri))

                        extra_names = msg.get("entity_name", [])[len(uri_name_pairs):]
                        for en in extra_names:
                            if is_clean(en):
                                entity_names_turn.append(_make_entity_name(en))
                    # ────────────────────────────────────────────────────────

                    turn_j += 1

                utt      = " ".join(utt_turn)
                mask_utt = " ".join(mask_utt_turn)

                if worker_id == initiator_id:
                    # ── 原版：user 侧只更新 context，不写 turn ─────────────
                    context.append(utt)
                    entity_list.append(entity_turn + movie_turn)

                    # ── [FIUP] 累积 user 侧实体名 ────────────────────────────
                    context_entity_names_list.append(entity_names_turn)
                    # ────────────────────────────────────────────────────────

                else:
                    # ── 原版：system 侧写 turn ────────────────────────────────
                    resp = utt

                    context_entity_list = [
                        e for el in entity_list for e in el
                    ]
                    context_entity_list_extend = []
                    context_entity_list_extend += context_entity_list
                    context_entity_list_extend = list(set(context_entity_list_extend))

                    # ── [FIUP] 累积上下文实体名（去重、保序）────────────────
                    seen = set()
                    context_entity_names = []
                    for name in [n for nl in context_entity_names_list for n in nl]:
                        if name not in seen:
                            seen.add(name)
                            context_entity_names.append(name)
                    # ────────────────────────────────────────────────────────

                    if len(context) == 0:
                        context.append("")

                    turn = {
                        # ── 原版字段（完全不变）──────────────────────────────
                        "context": context,
                        "resp":    mask_utt,
                        "rec":     movie_turn,
                        "entity":  context_entity_list_extend,
                        # ── [FIUP] 新增字段 ───────────────────────────────────
                        "entity_names": context_entity_names,
                        "user_id":      user_id,
                        "conv_id":      conv_id,
                    }
                    fout.write(json.dumps(turn, ensure_ascii=False) + "\n")

                    context.append(resp)
                    entity_list.append(movie_turn + entity_turn)
                    movie_set |= set(movie_turn)

                turn_i = turn_j


if __name__ == "__main__":
    with open("entity2id.json", "r", encoding="utf-8") as f:
        entity2id = json.load(f)
    id2entity = {v: k for k, v in entity2id.items()}
    movie_set = set()

    process("valid_data_dbpedia.jsonl", "valid_data_processed.jsonl", movie_set)
    process("test_data_dbpedia.jsonl",  "test_data_processed.jsonl",  movie_set)
    process("train_data_dbpedia.jsonl", "train_data_processed.jsonl", movie_set)

    with open("movie_ids.json", "w", encoding="utf-8") as f:
        json.dump(list(movie_set), f, ensure_ascii=False)
    print(f"#movie: {len(movie_set)}")




# import json
# import re

# import html
# from tqdm.auto import tqdm

# movie_pattern = re.compile(r'@\d+')


# def process_utt(utt, movieid2name, replace_movieId, remove_movie=False):
#     def convert(match):
#         movieid = match.group(0)[1:]
#         if movieid in movieid2name:
#             if remove_movie:
#                 return '<movie>'
#             movie_name = movieid2name[movieid]
#             # movie_name = f'<soi>{movie_name}<eoi>'
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
#                 mask_utt_turn = []

#                 turn_j = turn_i
#                 while turn_j < len(messages) and messages[turn_j]['senderWorkerId'] == worker_id:
#                     utt = process_utt(messages[turn_j]['text'], movieid2name, replace_movieId=True, remove_movie=False)
#                     utt_turn.append(utt)

#                     mask_utt = process_utt(messages[turn_j]['text'], movieid2name, replace_movieId=True, remove_movie=True)
#                     mask_utt_turn.append(mask_utt)

#                     entity_ids = [entity2id[entity] for entity in messages[turn_j]['entity'] if entity in entity2id]
#                     entity_turn.extend(entity_ids)

#                     movie_ids = [entity2id[movie] for movie in messages[turn_j]['movie'] if movie in entity2id]
#                     movie_turn.extend(movie_ids)

#                     turn_j += 1

#                 utt = ' '.join(utt_turn)
#                 mask_utt = ' '.join(mask_utt_turn)

#                 if worker_id == user_id:
#                     context.append(utt)
#                     entity_list.append(entity_turn + movie_turn)
#                 else:
#                     resp = utt

#                     context_entity_list = [entity for entity_l in entity_list for entity in entity_l]
#                     context_entity_list_extend = []
#                     # entity_links = [id2entity[id] for id in context_entity_list if id in id2entity]
#                     # for entity in entity_links:
#                     #     if entity in node2entity:
#                     #         for e in node2entity[entity]['entity']:
#                     #             if e in entity2id:
#                     #                 context_entity_list_extend.append(entity2id[e])
#                     context_entity_list_extend += context_entity_list
#                     context_entity_list_extend = list(set(context_entity_list_extend))

#                     if len(context) == 0:
#                         context.append('')
#                     turn = {
#                         'context': context,
#                         'resp': mask_utt,
#                         'rec': movie_turn,
#                         'entity': context_entity_list_extend,
#                     }
#                     fout.write(json.dumps(turn, ensure_ascii=False) + '\n')

#                     context.append(resp)
#                     entity_list.append(movie_turn + entity_turn)
#                     movie_set |= set(movie_turn)

#                 turn_i = turn_j


# if __name__ == '__main__':
#     with open('entity2id.json', 'r', encoding='utf-8') as f:
#         entity2id = json.load(f)
#     id2entity = {v: k for k, v in entity2id.items()}
#     movie_set = set()
#     # with open('node2abs_link_clean.json', 'r', encoding='utf-8') as f:
#     #     node2entity = json.load(f)

#     process('valid_data_dbpedia.jsonl', 'valid_data_processed.jsonl', movie_set)
#     process('test_data_dbpedia.jsonl', 'test_data_processed.jsonl', movie_set)
#     process('train_data_dbpedia.jsonl', 'train_data_processed.jsonl', movie_set)

#     with open('movie_ids.json', 'w', encoding='utf-8') as f:
#         json.dump(list(movie_set), f, ensure_ascii=False)
#     print(f'#movie: {len(movie_set)}')
