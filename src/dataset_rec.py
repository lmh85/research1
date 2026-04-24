import json
import os
from collections import defaultdict

import torch
from torch.utils.data import Dataset
from loguru import logger
from utils import padded_tensor


def build_prompt_with_fiup(
    context_text: str,
    kg_text: str,
    fiup_prompt: str,
    tokenizer,
    max_length: int = 200,
) -> dict:
    if fiup_prompt:
        full_text = f"{context_text} {fiup_prompt} {kg_text}"
    else:
        full_text = f"{context_text} {kg_text}"

    encoded = tokenizer(
        full_text,
        max_length=max_length,
        truncation=True,
        padding="max_length",
        return_tensors="pt",
    )
    return encoded


class CRSRecDataset(Dataset):
    """
    UniCRS 推荐任务数据集。
    支持 KG 邻居扩展（RAG）：通过 KGExpander 将 entity 字段扩展为 2 跳邻居，
    让 KGPrompt 的 cross-attention 覆盖更丰富的图结构，提升 Recall@K。

    使用方式：传入 kg_expander 实例即可，不传则与原版完全一致。
    """

    def __init__(
        self,
        dataset,
        split,
        tokenizer,
        context_max_length,
        prompt_tokenizer,
        prompt_max_length,
        entity_max_length,
        debug=False,
        use_resp=False,
        kg_expander=None,       # [RAG] KGExpander 实例，None = 不扩展
    ):
        super().__init__()
        self.debug = debug
        self.tokenizer = tokenizer
        self.prompt_tokenizer = prompt_tokenizer
        self.use_resp = use_resp
        self.split = split
        self.kg_expander = kg_expander      # [RAG]

        self.context_max_length = context_max_length
        self.prompt_max_length = prompt_max_length - 1
        self.entity_max_length = entity_max_length

        self.data = []
        data_file = os.path.join('data', dataset, f'{split}_data_processed.jsonl')
        self.prepare_data(data_file)

    def prepare_data(self, data_file):
        with open(data_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            if self.debug:
                lines = lines[:1024]

            for line in lines:
                dialog = json.loads(line)
                if len(dialog['rec']) == 0:
                    continue

                # ── 原版 context 拼接逻辑（不变）──────────────────────────────
                context = ''
                prompt_context = ''

                for i, utt in enumerate(dialog['context']):
                    if utt == '':
                        continue
                    if i % 2 == 0:
                        context += 'User: '
                        prompt_context += 'User: '
                    else:
                        context += 'System: '
                        prompt_context += 'System: '
                    context += utt
                    context += self.tokenizer.eos_token
                    prompt_context += utt
                    prompt_context += self.prompt_tokenizer.sep_token

                if self.use_resp and 'resp' in dialog:
                    if i % 2 == 0:
                        resp = 'System: '
                    else:
                        resp = 'User: '
                    resp += dialog['resp']
                    context += resp + self.tokenizer.eos_token
                    prompt_context += resp + self.prompt_tokenizer.sep_token

                # ── 原版 tokenize 逻辑（不变）─────────────────────────────────
                context_ids = self.tokenizer.convert_tokens_to_ids(
                    self.tokenizer.tokenize(context)
                )
                context_ids = context_ids[-self.context_max_length:]

                prompt_ids = self.prompt_tokenizer.convert_tokens_to_ids(
                    self.prompt_tokenizer.tokenize(prompt_context)
                )
                prompt_ids = prompt_ids[-self.prompt_max_length:]
                prompt_ids.insert(0, self.prompt_tokenizer.cls_token_id)

                # ── [RAG] KG 邻居扩展 ─────────────────────────────────────────
                entity_ids = dialog.get('entity', [])
                if self.kg_expander is not None:
                    entity_ids = self.kg_expander.expand(entity_ids)
                # ─────────────────────────────────────────────────────────────

                # ── FIUP 字段 ─────────────────────────────────────────────────
                entity_names = dialog.get('entity_names', [])
                user_id = str(dialog.get('user_id', dialog.get('conv_id', 'unknown')))
                context_str = prompt_context

                for item in dialog['rec']:
                    self.data.append({
                        'context':      context_ids,
                        'prompt':       prompt_ids,
                        'entity':       entity_ids[-self.entity_max_length:],
                        'rec':          item,
                        # FIUP 字段
                        'context_str':  context_str,
                        'user_id':      user_id,
                        'entity_names': entity_names,
                    })

        logger.info(f'[{self.split}] dataset size: {len(self.data)}')

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


class CRSRecDataCollator:
    """
    与原版逻辑完全一致，支持 FIUP 字段透传。
    """

    def __init__(
        self,
        tokenizer,
        prompt_tokenizer,
        device,
        context_max_length,
        prompt_max_length,
        entity_max_length,
        pad_entity_id,
        debug=False,
    ):
        self.tokenizer = tokenizer
        self.prompt_tokenizer = prompt_tokenizer
        self.device = device
        self.context_max_length = context_max_length
        self.prompt_max_length = prompt_max_length
        self.entity_max_length = entity_max_length
        self.pad_entity_id = pad_entity_id
        self.debug = debug
        self.padding = 'max_length' if debug else True

    def __call__(self, batch):
        context_batch = defaultdict(list)
        prompt_batch = defaultdict(list)
        entity_batch = []
        label_batch = []

        user_id_list = []
        context_str_list = []
        entity_names_list = []

        for sample in batch:
            context_batch['input_ids'].append(sample['context'])
            prompt_batch['input_ids'].append(sample['prompt'])
            entity_batch.append(sample['entity'])
            label_batch.append(sample['rec'])

            user_id_list.append(sample.get('user_id', 'unknown'))
            context_str_list.append(sample.get('context_str', ''))
            entity_names_list.append(sample.get('entity_names', []))

        input_batch = {}

        context_batch = self.tokenizer.pad(
            context_batch, padding=self.padding, max_length=self.context_max_length
        )
        context_batch['rec_labels'] = label_batch
        for k, v in context_batch.items():
            if not isinstance(v, torch.Tensor):
                context_batch[k] = torch.as_tensor(v, device=self.device)
        input_batch['context'] = context_batch

        prompt_batch = self.prompt_tokenizer.pad(
            prompt_batch, padding=self.padding, max_length=self.prompt_max_length
        )
        for k, v in prompt_batch.items():
            if not isinstance(v, torch.Tensor):
                prompt_batch[k] = torch.as_tensor(v, device=self.device)
        input_batch['prompt'] = prompt_batch

        entity_batch = padded_tensor(
            entity_batch, pad_idx=self.pad_entity_id, pad_tail=True, device=self.device
        )
        input_batch['entity'] = entity_batch

        input_batch['user_id']      = user_id_list
        input_batch['context_str']  = context_str_list
        input_batch['entity_names'] = entity_names_list

        return input_batch



# import json
# import os
# from collections import defaultdict

# import torch
# from torch.utils.data import Dataset, DataLoader
# from tqdm.auto import tqdm
# from transformers import AutoTokenizer

# from utils import padded_tensor


# class CRSRecDataset(Dataset):
#     def __init__(
#         self, dataset, split, tokenizer, debug=False,
#         context_max_length=None, entity_max_length=None,
#         prompt_tokenizer=None, prompt_max_length=None,
#         use_resp=False
#     ):
#         super(CRSRecDataset, self).__init__()
#         self.debug = debug
#         self.tokenizer = tokenizer
#         self.prompt_tokenizer = prompt_tokenizer
#         self.use_resp = use_resp

#         self.context_max_length = context_max_length
#         if self.context_max_length is None:
#             self.context_max_length = self.tokenizer.model_max_length

#         self.prompt_max_length = prompt_max_length
#         if self.prompt_max_length is None:
#             self.prompt_max_length = self.prompt_tokenizer.model_max_length
#         self.prompt_max_length -= 1

#         self.entity_max_length = entity_max_length
#         if self.entity_max_length is None:
#             self.entity_max_length = self.tokenizer.model_max_length

#         dataset_dir = os.path.join('data', dataset)
#         data_file = os.path.join(dataset_dir, f'{split}_data_processed.jsonl')
#         self.data = []
#         self.prepare_data(data_file)

#     def prepare_data(self, data_file):
#         with open(data_file, 'r', encoding='utf-8') as f:
#             lines = f.readlines()
#             if self.debug:
#                 lines = lines[:1024]

#             for line in tqdm(lines):
#                 dialog = json.loads(line)
#                 if len(dialog['rec']) == 0:
#                     continue
#                 if len(dialog['context']) == 1 and dialog['context'][0] == '':
#                     continue

#                 context = ''
#                 prompt_context = ''

#                 for i, utt in enumerate(dialog['context']):
#                     if utt == '':
#                         continue
#                     if i % 2 == 0:
#                         context += 'User: '
#                         prompt_context += 'User: '
#                     else:
#                         context += 'System: '
#                         prompt_context += 'System: '
#                     context += utt
#                     context += self.tokenizer.eos_token
#                     prompt_context += utt
#                     prompt_context += self.prompt_tokenizer.sep_token

#                 if context == '':
#                     continue
#                 if self.use_resp:
#                     if i % 2 == 0:
#                         resp = 'System: '
#                     else:
#                         resp = 'User: '
#                     resp += dialog['resp']
#                     context += resp + self.tokenizer.eos_token
#                     prompt_context += resp + self.prompt_tokenizer.sep_token

#                 context_ids = self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(context))
#                 context_ids = context_ids[-self.context_max_length:]

#                 prompt_ids = self.prompt_tokenizer.convert_tokens_to_ids(self.prompt_tokenizer.tokenize(prompt_context))
#                 prompt_ids = prompt_ids[-self.prompt_max_length:]
#                 prompt_ids.insert(0, self.prompt_tokenizer.cls_token_id)

#                 for item in dialog['rec']:
#                     data = {
#                         'context': context_ids,
#                         'entity': dialog['entity'][-self.entity_max_length:],
#                         'rec': item,
#                         'prompt': prompt_ids
#                     }
#                     self.data.append(data)

#     def __getitem__(self, ind):
#         return self.data[ind]

#     def __len__(self):
#         return len(self.data)


# class CRSRecDataCollator:
#     def __init__(
#         self, tokenizer, device, pad_entity_id, use_amp=False, debug=False,
#         context_max_length=None, entity_max_length=None,
#         prompt_tokenizer=None, prompt_max_length=None
#     ):
#         self.debug = debug
#         self.device = device
#         self.tokenizer = tokenizer
#         self.prompt_tokenizer = prompt_tokenizer

#         self.padding = 'max_length' if self.debug else True
#         self.pad_to_multiple_of = 8 if use_amp else None

#         self.context_max_length = context_max_length
#         if self.context_max_length is None:
#             self.context_max_length = self.tokenizer.model_max_length

#         self.prompt_max_length = prompt_max_length
#         if self.prompt_max_length is None:
#             self.prompt_max_length = self.prompt_tokenizer.model_max_length

#         self.pad_entity_id = pad_entity_id
#         self.entity_max_length = entity_max_length
#         if self.entity_max_length is None:
#             self.entity_max_length = self.tokenizer.model_max_length

#         # self.rec_prompt_ids = self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize('Recommend:'))

#     def __call__(self, data_batch):
#         context_batch = defaultdict(list)
#         prompt_batch = defaultdict(list)
#         entity_batch = []
#         label_batch = []

#         for data in data_batch:
#             # input_ids = data['context'][-(self.context_max_length - len(self.rec_prompt_ids)):] + self.rec_prompt_ids
#             input_ids = data['context']
#             context_batch['input_ids'].append(input_ids)
#             entity_batch.append(data['entity'])
#             label_batch.append(data['rec'])
#             prompt_batch['input_ids'].append(data['prompt'])

#         input_batch = {}

#         context_batch = self.tokenizer.pad(
#             context_batch, padding=self.padding, pad_to_multiple_of=self.pad_to_multiple_of,
#             max_length=self.context_max_length
#         )
#         context_batch['rec_labels'] = label_batch
#         for k, v in context_batch.items():
#             if not isinstance(v, torch.Tensor):
#                 context_batch[k] = torch.as_tensor(v, device=self.device)
#         input_batch['context'] = context_batch

#         prompt_batch = self.prompt_tokenizer.pad(
#             prompt_batch, padding=self.padding, max_length=self.prompt_max_length,
#             pad_to_multiple_of=self.pad_to_multiple_of
#         )
#         for k, v in prompt_batch.items():
#             if not isinstance(v, torch.Tensor):
#                 prompt_batch[k] = torch.as_tensor(v, device=self.device)
#         input_batch['prompt'] = prompt_batch

#         entity_batch = padded_tensor(entity_batch, pad_idx=self.pad_entity_id, pad_tail=True, device=self.device)
#         input_batch['entity'] = entity_batch

#         return input_batch


# if __name__ == '__main__':
#     from dataset_dbpedia import DBpedia
#     from config import gpt2_special_tokens_dict, prompt_special_tokens_dict
#     from pprint import pprint

#     debug = True
#     device = torch.device('cpu')
#     dataset = 'inspired'

#     kg = DBpedia(dataset, debug=debug).get_entity_kg_info()

#     model_name_or_path = "../utils/tokenizer/dialogpt-small"
#     tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
#     tokenizer.add_special_tokens(gpt2_special_tokens_dict)
#     prompt_tokenizer = AutoTokenizer.from_pretrained('../utils/tokenizer/roberta-base')
#     prompt_tokenizer.add_special_tokens(prompt_special_tokens_dict)

#     dataset = CRSRecDataset(
#         dataset=dataset, split='test', tokenizer=tokenizer, debug=debug,
#         prompt_tokenizer=prompt_tokenizer
#     )
#     for i in range(len(dataset)):
#         if i == 3:
#             break
#         data = dataset[i]
#         print(data)
#         print(tokenizer.decode(data['context']))
#         print(prompt_tokenizer.decode(data['prompt']))
#         print()

#     data_collator = CRSRecDataCollator(
#         tokenizer=tokenizer, device=device, pad_entity_id=kg['pad_entity_id'],
#         prompt_tokenizer=prompt_tokenizer
#     )
#     dataloader = DataLoader(
#         dataset,
#         batch_size=2,
#         collate_fn=data_collator,
#     )

#     input_max_len = 0
#     entity_max_len = 0
#     for batch in tqdm(dataloader):
#         if debug:
#             pprint(batch)
#             exit()

#         input_max_len = max(input_max_len, batch['context']['input_ids'].shape[1])
#         entity_max_len = max(entity_max_len, batch['entity'].shape[1])

#     print(input_max_len)
#     print(entity_max_len)
#     # (767, 26), (645, 29), (528, 16) -> (767, 29)
#     # inspired: (993, 25), (749, 20), (749, 31) -> (993, 31)
