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

    支持两个 RAG 组件：
      - kg_expander     : KG 2跳邻居扩展，扩充 entity 字段
                          原始实体优先保留，扩展实体填充剩余空间
      - movie_retriever : 电影描述检索（暂时禁用，等KG扩展稳定后再开）

    两个参数都是可选的，不传则与原版完全一致。
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
        kg_expander=None,
        movie_retriever=None,    # 保留参数接口，暂不使用
        train_file=None,         # [AUG] 两阶段训练用，指定增强数据文件路径
    ):
        super().__init__()
        self.debug            = debug
        self.tokenizer        = tokenizer
        self.prompt_tokenizer = prompt_tokenizer
        self.use_resp         = use_resp
        self.split            = split
        self.kg_expander      = kg_expander
        self.movie_retriever  = None   # 暂时强制禁用，避免干扰KG扩展效果验证

        self.context_max_length = context_max_length
        self.prompt_max_length  = prompt_max_length - 1
        self.entity_max_length  = entity_max_length

        self.data = []
        # train_file 只对 train split 生效，valid/test 始终用原始 processed 文件
        if split == 'train' and train_file is not None and os.path.exists(train_file):
            data_file = train_file
            print(f'[AUG] 使用增强训练数据: {train_file}')
        else:
            data_file = os.path.join('data', dataset, f'{split}_data_processed.jsonl')
        self.prepare_data(data_file)

    def prepare_data(self, data_file):
        n_kg_expanded = 0

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
                        context       += 'User: '
                        prompt_context += 'User: '
                    else:
                        context       += 'System: '
                        prompt_context += 'System: '
                    context        += utt + self.tokenizer.eos_token
                    prompt_context += utt + self.prompt_tokenizer.sep_token

                if self.use_resp and 'resp' in dialog:
                    resp = ('System: ' if i % 2 == 0 else 'User: ') + dialog['resp']
                    context        += resp + self.tokenizer.eos_token
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

                # ── [RAG-KG] KG 邻居扩展 + 离线预计算扩展得分 ───────────────
                entity_ids = dialog.get('entity', [])
                kg_expand_scores = {}   # {entity_id -> 归一化相关性得分}，推理时直接查表

                if self.kg_expander is not None:
                    original_ids = entity_ids
                    orig_set     = set(original_ids)

                    # 计算候选电影的共享节点数（用于重排序）
                    candidates = {}
                    for eid in original_ids:
                        eid_str = str(eid)
                        if eid_str not in self.kg_expander._kg:
                            continue
                        for triple in self.kg_expander._kg[eid_str]:
                            rel_id, nb_id = triple[0], triple[1]
                            if rel_id not in self.kg_expander._hq_rels:
                                continue
                            if nb_id in self.kg_expander._item_ids:
                                continue
                            for mid in self.kg_expander._nb_to_movies.get(nb_id, set()):
                                if mid not in orig_set:
                                    candidates[mid] = candidates.get(mid, 0) + 1

                    # 只保留共享节点数 >= 3 的高质量扩展，归一化得分
                    filtered = {mid: cnt for mid, cnt in candidates.items() if cnt >= 3}
                    if filtered:
                        max_cnt = max(filtered.values())
                        kg_expand_scores = {mid: cnt / max_cnt
                                            for mid, cnt in filtered.items()}
                        n_kg_expanded += 1

                    # entity 字段仍用原始实体（不把扩展实体加进训练）
                    entity_ids = original_ids[-self.entity_max_length:]
                else:
                    entity_ids = entity_ids[-self.entity_max_length:]

                # ── FIUP 字段 ──────────────────────────────────────────────────
                entity_names = dialog.get('entity_names', [])
                user_id      = str(dialog.get('user_id', dialog.get('conv_id', 'unknown')))
                context_str  = prompt_context

                for item in dialog['rec']:
                    self.data.append({
                        'context':          context_ids,
                        'prompt':           prompt_ids,
                        'entity':           entity_ids,
                        'rec':              item,
                        'context_str':      context_str,
                        'user_id':          user_id,
                        'entity_names':     entity_names,
                        'kg_expand_scores': kg_expand_scores,  # {entity_id -> 得分}，空则{}
                    })

        logger.info(f'[{self.split}] dataset size: {len(self.data)}')
        if self.kg_expander is not None:
            logger.info(f'[{self.split}][RAG-KG] KG expansion hit: '
                        f'{n_kg_expanded} / {len(self.data)} samples')

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


class CRSRecDataCollator:

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
        self.tokenizer        = tokenizer
        self.prompt_tokenizer = prompt_tokenizer
        self.device           = device
        self.context_max_length = context_max_length
        self.prompt_max_length  = prompt_max_length
        self.entity_max_length  = entity_max_length
        self.pad_entity_id    = pad_entity_id
        self.debug            = debug
        self.padding          = 'max_length' if debug else True

    def __call__(self, batch):
        context_batch     = defaultdict(list)
        prompt_batch      = defaultdict(list)
        entity_batch      = []
        label_batch       = []
        user_id_list      = []
        context_str_list  = []
        entity_names_list = []
        kg_expand_scores_list = []

        for sample in batch:
            context_batch['input_ids'].append(sample['context'])
            prompt_batch['input_ids'].append(sample['prompt'])
            entity_batch.append(sample['entity'])
            label_batch.append(sample['rec'])
            user_id_list.append(sample.get('user_id', 'unknown'))
            context_str_list.append(sample.get('context_str', ''))
            entity_names_list.append(sample.get('entity_names', []))
            kg_expand_scores_list.append(sample.get('kg_expand_scores', {}))

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
        input_batch['entity']       = entity_batch
        input_batch['user_id']           = user_id_list
        input_batch['context_str']       = context_str_list
        input_batch['entity_names']      = entity_names_list
        input_batch['kg_expand_scores']  = kg_expand_scores_list  # List[Dict[int,float]]

        return input_batch