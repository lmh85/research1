import json
import re
from tqdm.auto import tqdm


def process(data_file, out_file, movie_set, split_name):
    with open(data_file, 'r', encoding='utf-8') as fin, \
         open(out_file, 'w', encoding='utf-8') as fout:

        for conv_idx, line in enumerate(tqdm(fin, desc=split_name)):
            dialog = json.loads(line)

            context, resp = [], ''
            entity_list = []
            movie_names_seen = []   # 累积出现的电影名（用于 entity_names）

            for turn in dialog:
                resp = turn['text']
                entity_link = [entity2id[e] for e in turn['entity_link'] if e in entity2id]
                movie_link  = [entity2id[m] for m in turn['movie_link']  if m in entity2id]

                if len(context) == 0:
                    context.append('')

                # ── [FIUP] entity_names：直接用原始数据里的 movie_name 字段 ──
                # 格式：["Movie:电影名"] 和 ReDial 保持一致
                turn_movie_names = []
                for mname in turn.get('movie_name', []):
                    if mname:
                        turn_movie_names.append(f'Movie:{mname}')
                movie_names_seen.extend(turn_movie_names)

                turn_data = {
                    'context':      context.copy(),
                    'resp':         resp,
                    'rec':          list(set(movie_link + entity_link)),
                    'entity':       list(set(entity_list)),
                    'entity_names': list(set(movie_names_seen)),  # [FIUP]
                    'conv_id':      str(conv_idx),                # [FIUP] 对话 ID
                    'user_id':      str(conv_idx),                # [FIUP] 用 conv_id 作为用户 ID
                }
                fout.write(json.dumps(turn_data, ensure_ascii=False) + '\n')

                context.append(resp)
                entity_list.extend(entity_link + movie_link)
                movie_set |= set(movie_link)


if __name__ == '__main__':
    with open('entity2id.json', 'r', encoding='utf-8') as f:
        entity2id = json.load(f)

    item_set = set()
    process('test_data_dbpedia.jsonl',  'test_data_processed.jsonl',  item_set, 'test')
    process('valid_data_dbpedia.jsonl', 'valid_data_processed.jsonl', item_set, 'valid')
    process('train_data_dbpedia.jsonl', 'train_data_processed.jsonl', item_set, 'train')

    with open('item_ids.json', 'w', encoding='utf-8') as f:
        json.dump(list(item_set), f, ensure_ascii=False)
    print(f'#item: {len(item_set)}')


# import json
# from tqdm.auto import tqdm


# def process(data_file, out_file, movie_set):
#     with open(data_file, 'r', encoding='utf-8') as fin, open(out_file, 'w', encoding='utf-8') as fout:
#         for line in tqdm(fin):
#             dialog = json.loads(line)

#             context, resp = [], ''
#             entity_list = []

#             for turn in dialog:
#                 resp = turn['text']
#                 entity_link = [entity2id[entity] for entity in turn['entity_link'] if entity in entity2id]
#                 movie_link = [entity2id[movie] for movie in turn['movie_link'] if movie in entity2id]

#                 # if turn['role'] == 'SEEKER':
#                 #     context.append(resp)
#                 #     entity_list.extend(entity_link + movie_link)
#                 # else:
#                 if len(context) == 0:
#                     context.append('')
#                 turn = {
#                     'context': context,
#                     'resp': resp,
#                     'rec': list(set(movie_link + entity_link)),
#                     'entity': list(set(entity_list))
#                 }
#                 fout.write(json.dumps(turn, ensure_ascii=False) + '\n')

#                 context.append(resp)
#                 entity_list.extend(entity_link + movie_link)
#                 movie_set |= set(movie_link)


# if __name__ == '__main__':
#     with open('entity2id.json', 'r', encoding='utf-8') as f:
#         entity2id = json.load(f)
#     item_set = set()

#     process('test_data_dbpedia.jsonl', 'test_data_processed.jsonl', item_set)
#     process('valid_data_dbpedia.jsonl', 'valid_data_processed.jsonl', item_set)
#     process('train_data_dbpedia.jsonl', 'train_data_processed.jsonl', item_set)

#     with open('item_ids.json', 'w', encoding='utf-8') as f:
#         json.dump(list(item_set), f, ensure_ascii=False)
#     print(f'#item: {len(item_set)}')
