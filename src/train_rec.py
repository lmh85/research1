import argparse
import math
import os
import sys
import time

import numpy as np
import torch
import transformers
import wandb
from accelerate import Accelerator
from accelerate.utils import set_seed
from loguru import logger
from torch.utils.data import DataLoader, random_split
from tqdm.auto import tqdm
from transformers import AdamW, get_linear_schedule_with_warmup, AutoTokenizer, AutoModel

from config import gpt2_special_tokens_dict, prompt_special_tokens_dict
from dataset_dbpedia import DBpedia
from dataset_rec import CRSRecDataset, CRSRecDataCollator, build_prompt_with_fiup
from evaluate_rec import RecEvaluator
from model_gpt2 import PromptGPT2forCRS
from model_prompt import KGPrompt
from modules.sentiment import SentimentAnalyzer
from modules.fiup_manager import FIUPManager
import re as _re
import json as _json

# ── [RAG] 条件导入，文件不存在时优雅降级 ─────────────────────────────────────
try:
    from modules.kg_expander import KGExpander
    _HAS_KG_EXPANDER = True
except ImportError:
    _HAS_KG_EXPANDER = False
    logger.warning("[RAG] modules/kg_expander.py not found, --use_kg_expand will be ignored.")

try:
    from modules.movie_retriever import MovieRetriever
    _HAS_MOVIE_RETRIEVER = True
except ImportError:
    _HAS_MOVIE_RETRIEVER = False
    logger.warning("[RAG-DESC] modules/movie_retriever.py not found, --use_movie_retrieval will be ignored.")
# ─────────────────────────────────────────────────────────────────────────────


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_dir", type=str, default='save')
    parser.add_argument("--debug", action='store_true')
    # data
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--shot", type=float, default=1)
    parser.add_argument("--use_resp", action="store_true")
    parser.add_argument("--context_max_length", type=int)
    parser.add_argument("--prompt_max_length", type=int)
    parser.add_argument("--entity_max_length", type=int)
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument("--tokenizer", type=str)
    parser.add_argument("--text_tokenizer", type=str)
    # model
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--text_encoder", type=str)
    parser.add_argument("--num_bases", type=int, default=8)
    parser.add_argument("--n_prefix_rec", type=int)
    parser.add_argument("--prompt_encoder", type=str)
    # FIUP
    parser.add_argument("--use_fiup", action="store_true", default=False)
    parser.add_argument("--fiup_alpha", type=float, default=0.8)
    parser.add_argument("--fiup_lambda", type=float, default=0.1,
                        help="显性库重排序融合权重")
    parser.add_argument("--fiup_implicit_alpha", type=float, default=0.2,
                        help="隐性库 entity_embeds 加权强度")
    parser.add_argument("--sentiment_backend", type=str, default="textblob",
                        choices=["textblob", "transformers"])
    parser.add_argument("--fiup_states_dir", type=str, default=None,
                        help="对话阶段保存的 FIUP 画像目录")
    # [AUG] 两阶段训练
    parser.add_argument("--train_file", type=str, default=None,
                        help="第一阶段训练用的增强数据文件路径，"
                             "如 data/redial_gen/train_data_augmented.jsonl；"
                             "不指定则使用原始 train_data_processed.jsonl")
    # RAG-KG
    parser.add_argument("--use_kg_expand", action="store_true", default=False)
    parser.add_argument("--kg_expand_max", type=int, default=16)
    # RAG-DESC
    parser.add_argument("--use_movie_retrieval", action="store_true", default=False)
    parser.add_argument("--retrieval_top_k", type=int, default=2)
    # optim
    parser.add_argument("--num_train_epochs", type=int, default=10)
    parser.add_argument("--max_train_steps", type=int, default=None)
    parser.add_argument("--per_device_train_batch_size", type=int, default=4)
    parser.add_argument("--per_device_eval_batch_size", type=int, default=4)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--learning_rate", type=float, default=1e-5)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument('--max_grad_norm', type=float)
    parser.add_argument('--num_warmup_steps', type=int)
    parser.add_argument('--fp16', action='store_true')
    # wandb
    parser.add_argument("--use_wandb", action="store_true")
    parser.add_argument("--entity", type=str)
    parser.add_argument("--project", type=str)
    parser.add_argument("--name", type=str)
    parser.add_argument("--log_all", action="store_true")

    args = parser.parse_args()
    return args


# ── 辅助：用 text_encoder 提取 [CLS] 语义向量 ────────────────────────────────
def encode_context_emb(ctx_str, text_tokenizer, text_encoder, device, max_length=128):
    if not ctx_str:
        return torch.zeros(text_encoder.config.hidden_size)
    tokenized = text_tokenizer(
        ctx_str, return_tensors="pt",
        max_length=max_length, truncation=True, padding=True,
    ).to(device)
    with torch.no_grad():
        outputs = text_encoder(**tokenized)
        context_emb = outputs.last_hidden_state[:, 0, :].squeeze(0).cpu()
    return context_emb


# ── 隐性库：entity_embeds 语义加权 ───────────────────────────────────────────
def apply_implicit_bias(entity_embeds, user_ids, fiup_managers, alpha=0.2):
    """
    用 FIUP 隐性库向量对 entity_embeds 做余弦相似度加权。
    公式：entity_embeds_new[i] = entity_embeds[i] * (1 + α * cos_sim(implicit_vec, entity_embeds[i]))
    batch 内多用户取平均隐性库向量。
    """
    import torch.nn.functional as F

    if not fiup_managers or alpha == 0.0:
        return entity_embeds

    device = entity_embeds.device
    implicit_vecs = []
    for uid in user_ids:
        mgr = fiup_managers.get(uid)
        if mgr is not None:
            vec = mgr.get_implicit_vector()
            if vec.norm() > 1e-6:
                implicit_vecs.append(vec)

    if not implicit_vecs:
        return entity_embeds

    implicit_mean = torch.stack(implicit_vecs, dim=0).mean(dim=0).to(device)
    entity_norm   = F.normalize(entity_embeds, dim=-1)
    implicit_norm = F.normalize(implicit_mean.unsqueeze(0), dim=-1)
    cos_sim       = (entity_norm * implicit_norm).sum(dim=-1)
    weights       = (1.0 + alpha * cos_sim).unsqueeze(-1)
    return entity_embeds * weights


# ── 显性库：logits 重排序 ──────────────────────────────────────────────────────
def _build_explicit_prefix_index(explicit_lib: dict) -> tuple:
    """
    将显性库建立两个索引：
    1. movie_index: {Movie前缀 -> 权重}，去掉年份后匹配
    2. genre_index: {genre关键词(小写) -> 权重}，用于和KG genre名字做子串匹配

    返回 (movie_index, genre_index)
    """
    movie_index = {}
    genre_index = {}

    for key, weight in explicit_lib.items():
        if key.startswith('Movie:'):
            clean = _re.sub(r'\s*\(\d{4}\)\s*$', '', key).strip()
            clean = _re.sub(r'\s*\(.*?film.*?\)\s*$', '', clean).strip()
            if clean not in movie_index:
                movie_index[clean] = weight
            else:
                movie_index[clean] = max(movie_index[clean], weight)

        elif key.startswith('Genre:'):
            genre_kw = key.replace('Genre:', '').lower().strip()
            if genre_kw not in genre_index:
                genre_index[genre_kw] = weight
            else:
                genre_index[genre_kw] = max(genre_index[genre_kw], weight)

    return movie_index, genre_index


def _match_genre_score(kg_genres: list, genre_index: dict) -> float:
    """
    给定一部电影的 KG genre 列表和显性库 genre 索引，
    计算该电影的 genre 偏好得分（取所有匹配权重的最大值）。
    双向子串匹配：'comedy' in 'comedy film' ✓ / 'comedy film' in 'comedy films' ✓
    """
    best_score = 0.0
    for kg_genre in kg_genres:
        for genre_kw, weight in genre_index.items():
            if genre_kw in kg_genre or kg_genre in genre_kw:
                if abs(weight) > abs(best_score):
                    best_score = weight
    return best_score


def fiup_rerank(logits, item_ids, user_ids, fiup_managers, id2name,
                id2genres, fiup_lambda=0.1, batch_kg_scores=None,
                kg_expand_weight=0.3):
    """
    用 FIUP 显性库 + KG 扩展对推荐 logits 加权重排序。

    得分组成：
      pref_scores  = FIUP显性库偏好（Movie级别 + Genre级别）
      kg_scores    = KG 2跳扩展相关性（dataset阶段离线预计算，共享节点>=3）
      final_logits = logits + λ * pref_scores + λ * kg_expand_weight * kg_scores

    batch_kg_scores: List[Dict[int, float]]，每个样本的KG扩展得分字典
    kg_expand_weight: KG扩展信号相对于FIUP信号的比例（默认0.3）
    """
    device = logits.device
    pref_scores = torch.zeros_like(logits)
    kg_scores   = torch.zeros_like(logits)

    item_id_to_idx = {eid: j for j, eid in enumerate(item_ids)}

    for b, uid in enumerate(user_ids):
        # ── FIUP 显性库偏好分 ─────────────────────────────────────────────
        mgr = fiup_managers.get(uid)
        if mgr is not None:
            explicit_lib = mgr.explicit_lib
            if explicit_lib:
                movie_index, genre_index = _build_explicit_prefix_index(explicit_lib)
                for j, eid in enumerate(item_ids):
                    movie_name = id2name.get(eid, '')
                    score = 0.0
                    if movie_name:
                        lookup_key = f'Movie:{movie_name}'
                        movie_score = explicit_lib.get(lookup_key, None)
                        if movie_score is None:
                            movie_score = movie_index.get(lookup_key, 0.0)
                        score = movie_score
                    if genre_index:
                        kg_genres = id2genres.get(eid, [])
                        if kg_genres:
                            genre_score = _match_genre_score(kg_genres, genre_index)
                            if abs(genre_score) > abs(score):
                                score = genre_score
                    pref_scores[b, j] = score

        # ── KG 扩展相关性分（离线预计算，直接查表）───────────────────────
        if batch_kg_scores is not None and b < len(batch_kg_scores):
            expand_scores = batch_kg_scores[b]   # Dict[int, float]
            if expand_scores:
                for mid, s in expand_scores.items():
                    j = item_id_to_idx.get(mid)
                    if j is not None:
                        kg_scores[b, j] = s

    # 归一化 FIUP 偏好分到 [-1, 1]
    max_abs = pref_scores.abs().max()
    if max_abs > 1e-8:
        pref_scores = pref_scores / max_abs

    # kg_scores 已在 [0, 1]，不需要额外归一化
    return logits + fiup_lambda * pref_scores + fiup_lambda * kg_expand_weight * kg_scores


# ── eval 阶段通用推理函数（避免 valid/test 代码重复）─────────────────────────
def run_eval(dataloader, prompt_encoder, text_encoder, model, kg, evaluator,
             accelerator, args, fiup_managers, id2name_for_rerank,
             id2genres_for_rerank, split_name):
    """
    返回 (report_dict, mean_loss)
    split_name: 'valid' 或 'test'，仅用于日志前缀
    """
    loss_list = []
    prompt_encoder.eval()

    for batch in tqdm(dataloader, desc=split_name, disable=not accelerator.is_local_main_process):
        # 提取 user_ids（重排序和隐性库加权都需要）
        if args.use_fiup:
            batch_size = len(batch["context"]["input_ids"])
            user_ids   = batch.get("user_id", [f"{split_name}_{i}" for i in range(batch_size)])

        with torch.no_grad():
            token_embeds  = text_encoder(**batch['prompt']).last_hidden_state
            prompt_embeds = prompt_encoder(
                entity_ids=batch['entity'],
                token_embeds=token_embeds,
                output_entity=True,
                use_rec_prefix=True,
            )
            batch['context']['prompt_embeds'] = prompt_embeds

            entity_embeds = prompt_encoder.get_entity_embeds()

            # ── [FIUP] 隐性库加权 ─────────────────────────────────────────
            if args.use_fiup and fiup_managers:
                entity_embeds = apply_implicit_bias(
                    entity_embeds=entity_embeds,
                    user_ids=user_ids,
                    fiup_managers=fiup_managers,
                    alpha=args.fiup_implicit_alpha,
                )
            # ──────────────────────────────────────────────────────────────
            batch['context']['entity_embeds'] = entity_embeds

            outputs = model(**batch['context'], rec=True)
            loss_list.append(float(outputs.rec_loss))
            logits  = outputs.rec_logits[:, kg['item_ids']]

            # ── [FIUP] 显性库重排序 + KG扩展（离线预计算得分）────────────
            if args.use_fiup and fiup_managers:
                # 从 batch 读离线预计算的 KG 扩展得分（dataset 阶段已算好）
                batch_kg_scores = batch.get('kg_expand_scores', None)
                logits = fiup_rerank(
                    logits=logits,
                    item_ids=kg['item_ids'],
                    user_ids=user_ids,
                    fiup_managers=fiup_managers,
                    id2name=id2name_for_rerank,
                    id2genres=id2genres_for_rerank,
                    fiup_lambda=args.fiup_lambda,
                    batch_kg_scores=batch_kg_scores,
                    kg_expand_weight=0.3,
                )
            # ──────────────────────────────────────────────────────────────

            ranks  = torch.topk(logits, k=50, dim=-1).indices.tolist()
            ranks  = [[kg['item_ids'][r] for r in batch_rank] for batch_rank in ranks]
            labels = batch['context']['rec_labels']
            evaluator.evaluate(ranks, labels)

    report = accelerator.gather(evaluator.report())
    for k, v in report.items():
        report[k] = v.sum().item()

    out = {f'{split_name}/{k}': v / report['count']
           for k, v in report.items() if k != 'count'}
    out[f'{split_name}/loss'] = np.mean(loss_list)
    evaluator.reset_metric()
    return out


# ─────────────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    args = parse_args()
    config = vars(args)

    mixed_precision = "fp16" if args.fp16 else "no"
    accelerator = Accelerator(device_placement=False, mixed_precision=mixed_precision)
    device = accelerator.device

    local_time = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
    logger.remove()
    logger.add(sys.stderr, level='DEBUG' if accelerator.is_local_main_process else 'ERROR')
    logger.add(f'log/{local_time}.log', level='DEBUG' if accelerator.is_local_main_process else 'ERROR')
    logger.info(accelerator.state)
    logger.info(config)

    if accelerator.is_local_main_process:
        transformers.utils.logging.set_verbosity_info()
    else:
        transformers.utils.logging.set_verbosity_error()

    if args.use_wandb:
        name = args.name if args.name else local_time
        name += '_' + str(accelerator.process_index)
        if args.log_all:
            group = args.name if args.name else 'DDP_' + local_time
            run = wandb.init(entity=args.entity, project=args.project,
                             group=group, config=config, name=name)
        else:
            run = wandb.init(entity=args.entity, project=args.project,
                             config=config, name=name) if accelerator.is_local_main_process else None
    else:
        run = None

    if args.seed is not None:
        set_seed(args.seed)

    if args.output_dir is not None:
        os.makedirs(args.output_dir, exist_ok=True)

    kg = DBpedia(dataset=args.dataset, debug=args.debug).get_entity_kg_info()

    # ── 构建 entity_id -> 可读名称 映射（显性库重排序用）─────────────────────
    _e2id_path = os.path.join('data', args.dataset, 'entity2id.json')
    with open(_e2id_path) as _f:
        _e2id = _json.load(_f)
    id2name_for_rerank = {}
    for _uri, _eid in _e2id.items():
        _name = _uri.replace('<http://dbpedia.org/resource/', '').rstrip('>')
        _name = _re.sub(r'\s*\(.*?film.*?\)\s*$', '', _name.replace('_', ' ')).strip()
        id2name_for_rerank[_eid] = _name

    # ── 构建 entity_id -> genre列表 映射（Genre 重排序用）────────────────────
    _kg_path   = os.path.join('data', args.dataset, 'dbpedia_subkg.json')
    _r2id_path = os.path.join('data', args.dataset, 'relation2id.json')
    _item_path = os.path.join('data', args.dataset, 'item_ids.json')
    id2genres_for_rerank = {}
    if all(os.path.exists(p) for p in [_kg_path, _r2id_path, _item_path]):
        with open(_kg_path) as _f:
            _kg = _json.load(_f)
        with open(_r2id_path) as _f:
            _r2id = _json.load(_f)
        with open(_item_path) as _f:
            _item_ids_set = set(_json.load(_f))
        _id2e = {v: k for k, v in _e2id.items()}
        _genre_rel_ids = {v for k, v in _r2id.items() if 'genre' in k.lower()}
        for _eid_str, _triples in _kg.items():
            _eid = int(_eid_str)
            if _eid not in _item_ids_set:
                continue
            _genres = []
            for _triple in _triples:
                if _triple[0] in _genre_rel_ids:
                    _nb_uri = _id2e.get(_triple[1], '')
                    _nb_name = _nb_uri.replace(
                        '<http://dbpedia.org/resource/', '').rstrip('>').replace('_', ' ').lower()
                    if _nb_name:
                        _genres.append(_nb_name)
            if _genres:
                id2genres_for_rerank[_eid] = _genres
        logger.info(f'[Genre] 构建电影->genre映射完成，覆盖 {len(id2genres_for_rerank)} 部电影')
    # ─────────────────────────────────────────────────────────────────────────

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    tokenizer.add_special_tokens(gpt2_special_tokens_dict)
    model = PromptGPT2forCRS.from_pretrained(args.model)
    model.resize_token_embeddings(len(tokenizer))
    model.config.pad_token_id = tokenizer.pad_token_id
    model = model.to(device)

    text_tokenizer = AutoTokenizer.from_pretrained(args.text_tokenizer)
    text_tokenizer.add_special_tokens(prompt_special_tokens_dict)
    text_encoder = AutoModel.from_pretrained(args.text_encoder)
    text_encoder.resize_token_embeddings(len(text_tokenizer))
    text_encoder = text_encoder.to(device)

    prompt_encoder = KGPrompt(
        model.config.n_embd, text_encoder.config.hidden_size,
        model.config.n_head, model.config.n_layer, 2,
        n_entity=kg['num_entities'], num_relations=kg['num_relations'],
        num_bases=args.num_bases,
        edge_index=kg['edge_index'], edge_type=kg['edge_type'],
        n_prefix_rec=args.n_prefix_rec,
    )
    if args.prompt_encoder is not None:
        prompt_encoder.load(args.prompt_encoder)
    prompt_encoder = prompt_encoder.to(device)

    for module in [model, text_encoder]:
        module.requires_grad_(False)

    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {"params": [p for n, p in prompt_encoder.named_parameters()
                    if not any(nd in n for nd in no_decay) and p.requires_grad],
         "weight_decay": args.weight_decay},
        {"params": [p for n, p in prompt_encoder.named_parameters()
                    if any(nd in n for nd in no_decay) and p.requires_grad],
         "weight_decay": 0.0},
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate)

    # ── [RAG-KG] 初始化 KG 邻居扩展器 ───────────────────────────────────────
    kg_expander = None
    if args.use_kg_expand:
        if not _HAS_KG_EXPANDER:
            logger.warning("[RAG] KGExpander module not available, skipping.")
        else:
            dataset_dir  = os.path.join('data', args.dataset)
            kg_path      = os.path.join(dataset_dir, 'dbpedia_subkg.json')
            r2id_path    = os.path.join(dataset_dir, 'relation2id.json')
            item_id_path = os.path.join(dataset_dir, 'item_ids.json')
            if all(os.path.exists(p) for p in [kg_path, r2id_path, item_id_path]):
                kg_expander = KGExpander(
                    kg_path=kg_path,
                    relation2id_path=r2id_path,
                    item_ids_path=item_id_path,
                    max_expand=args.kg_expand_max,
                )
                logger.info(f"[RAG] KGExpander initialized: {kg_expander.stats()}")
            else:
                logger.warning("[RAG] KG files not found, skipping KG expansion.")
    # ─────────────────────────────────────────────────────────────────────────

    # ── [RAG-DESC] 初始化电影描述检索器 ──────────────────────────────────────
    movie_retriever = None
    if args.use_movie_retrieval:
        if not _HAS_MOVIE_RETRIEVER:
            logger.warning("[RAG-DESC] MovieRetriever module not available, skipping.")
        else:
            dataset_dir  = os.path.join('data', args.dataset)
            kg_path      = os.path.join(dataset_dir, 'dbpedia_subkg.json')
            e2id_path    = os.path.join(dataset_dir, 'entity2id.json')
            r2id_path    = os.path.join(dataset_dir, 'relation2id.json')
            item_id_path = os.path.join(dataset_dir, 'item_ids.json')
            if all(os.path.exists(p) for p in [kg_path, e2id_path, r2id_path, item_id_path]):
                movie_retriever = MovieRetriever(
                    kg_path=kg_path,
                    entity2id_path=e2id_path,
                    relation2id_path=r2id_path,
                    item_ids_path=item_id_path,
                    top_k=args.retrieval_top_k,
                )
                logger.info(f"[RAG-DESC] MovieRetriever initialized: {movie_retriever.stats()}")
            else:
                logger.warning("[RAG-DESC] KG files not found, skipping movie retrieval.")
    # ─────────────────────────────────────────────────────────────────────────

    # ── dataset & dataloader ──────────────────────────────────────────────────
    # 注意：CRSRecDataset 当前版本不接受 kg_expander 参数
    # 如果 dataset_rec.py 已更新支持 kg_expander，可将下面三处的注释打开
    def make_dataset(split):
        return CRSRecDataset(
            dataset=args.dataset, split=split, debug=args.debug,
            tokenizer=tokenizer,
            context_max_length=args.context_max_length,
            use_resp=args.use_resp,
            prompt_tokenizer=text_tokenizer,
            prompt_max_length=args.prompt_max_length,
            entity_max_length=args.entity_max_length,
            kg_expander=kg_expander,
            movie_retriever=movie_retriever,
            train_file=args.train_file,
        )

    train_dataset = make_dataset('train')
    shot_len      = int(len(train_dataset) * args.shot)
    train_dataset = random_split(train_dataset, [shot_len, len(train_dataset) - shot_len])[0]
    valid_dataset = make_dataset('valid')
    test_dataset  = make_dataset('test')

    data_collator = CRSRecDataCollator(
        tokenizer=tokenizer, device=device, debug=args.debug,
        context_max_length=args.context_max_length,
        entity_max_length=args.entity_max_length,
        pad_entity_id=kg['pad_entity_id'],
        prompt_tokenizer=text_tokenizer,
        prompt_max_length=args.prompt_max_length,
    )
    train_dataloader = DataLoader(
        train_dataset, batch_size=args.per_device_train_batch_size,
        collate_fn=data_collator, shuffle=True,
    )
    valid_dataloader = DataLoader(
        valid_dataset, batch_size=args.per_device_eval_batch_size,
        collate_fn=data_collator,
    )
    test_dataloader = DataLoader(
        test_dataset, batch_size=args.per_device_eval_batch_size,
        collate_fn=data_collator,
    )
    evaluator = RecEvaluator()

    prompt_encoder, optimizer, train_dataloader, valid_dataloader, test_dataloader = accelerator.prepare(
        prompt_encoder, optimizer, train_dataloader, valid_dataloader, test_dataloader
    )

    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    else:
        args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)
    total_batch_size = (args.per_device_train_batch_size
                        * accelerator.num_processes
                        * args.gradient_accumulation_steps)
    completed_steps = 0

    lr_scheduler = get_linear_schedule_with_warmup(
        optimizer, args.num_warmup_steps, args.max_train_steps
    )
    lr_scheduler = accelerator.prepare(lr_scheduler)

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Batch size per device = {args.per_device_train_batch_size}")
    logger.info(f"  Total train batch size = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")

    progress_bar = tqdm(range(args.max_train_steps),
                        disable=not accelerator.is_local_main_process)

    metric, mode = 'loss', -1
    best_metric     = float('inf')
    best_metric_dir = os.path.join(args.output_dir, 'best')
    os.makedirs(best_metric_dir, exist_ok=True)

    # ── [FIUP] 初始化：加载对话阶段积累的画像（若指定目录）──────────────────
    fiup_managers = {}
    if args.use_fiup:
        sentiment_analyzer = SentimentAnalyzer(backend=args.sentiment_backend)
        EMB_DIM = text_encoder.config.hidden_size

        if args.fiup_states_dir and os.path.isdir(args.fiup_states_dir):
            loaded = 0
            for fname in os.listdir(args.fiup_states_dir):
                if not fname.endswith('.json'):
                    continue
                uid  = fname[:-5]
                path = os.path.join(args.fiup_states_dir, fname)
                try:
                    fiup_managers[uid] = FIUPManager.load(path, device=device)
                    loaded += 1
                except Exception as e:
                    logger.warning(f'[FIUP] Failed to load {path}: {e}')
            logger.info(
                f'[FIUP] Loaded {loaded} user profiles from {args.fiup_states_dir}; '
                f'new users will be initialized on first encounter.'
            )
        else:
            logger.info('[FIUP] No fiup_states_dir provided; all profiles initialized from scratch.')

        logger.info(
            f'[FIUP] use_fiup=True  fiup_lambda={args.fiup_lambda}  '
            f'fiup_implicit_alpha={args.fiup_implicit_alpha}  '
            f'fiup_alpha={args.fiup_alpha}  backend={args.sentiment_backend}'
        )
    # ─────────────────────────────────────────────────────────────────────────

    # ════════════════════════ train loop ═════════════════════════════════════
    for epoch in range(args.num_train_epochs):
        train_loss = []
        prompt_encoder.train()

        for step, batch in enumerate(train_dataloader):

            # ── [FIUP] 训练阶段：更新画像 + 隐性库加权 ───────────────────────
            if args.use_fiup:
                batch_size         = len(batch["context"]["input_ids"])
                user_ids           = batch.get("user_id",      [f"u{step}_{i}" for i in range(batch_size)])
                context_strs       = batch.get("context_str",  [""] * batch_size)
                entity_names_batch = batch.get("entity_names", [[] for _ in range(batch_size)])

                for uid, ctx_str, attrs in zip(user_ids, context_strs, entity_names_batch):
                    if uid not in fiup_managers:
                        fiup_managers[uid] = FIUPManager(emb_dim=EMB_DIM, alpha=args.fiup_alpha)
                    mgr         = fiup_managers[uid]
                    e_tau       = sentiment_analyzer.score(ctx_str)
                    context_emb = encode_context_emb(
                        ctx_str, text_tokenizer, text_encoder, device, max_length=128
                    )
                    mgr.update_profile(attrs, e_tau, context_emb)
            # ────────────────────────────────────────────────────────────────

            with torch.no_grad():
                token_embeds = text_encoder(**batch['prompt']).last_hidden_state  # 只算一次

            prompt_embeds = prompt_encoder(
                entity_ids=batch['entity'],
                token_embeds=token_embeds,
                output_entity=True,
                use_rec_prefix=True,
            )
            batch['context']['prompt_embeds'] = prompt_embeds

            entity_embeds = prompt_encoder.get_entity_embeds()

            # ── [FIUP] 隐性库加权（训练阶段）────────────────────────────────
            if args.use_fiup and fiup_managers:
                entity_embeds = apply_implicit_bias(
                    entity_embeds=entity_embeds,
                    user_ids=user_ids,
                    fiup_managers=fiup_managers,
                    alpha=args.fiup_implicit_alpha,
                )
            # ────────────────────────────────────────────────────────────────
            batch['context']['entity_embeds'] = entity_embeds

            loss = model(**batch['context'], rec=True).rec_loss / args.gradient_accumulation_steps
            accelerator.backward(loss)
            train_loss.append(float(loss))

            if step % args.gradient_accumulation_steps == 0 or step == len(train_dataloader) - 1:
                if args.max_grad_norm is not None:
                    accelerator.clip_grad_norm_(prompt_encoder.parameters(), args.max_grad_norm)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
                progress_bar.update(1)
                completed_steps += 1
                if run:
                    run.log({'loss': np.mean(train_loss) * args.gradient_accumulation_steps})

            if completed_steps >= args.max_train_steps:
                break

        train_loss_mean = np.mean(train_loss) * args.gradient_accumulation_steps
        logger.info(f'epoch {epoch} train loss {train_loss_mean}')
        del train_loss, batch

        # ── valid ─────────────────────────────────────────────────────────────
        valid_report = run_eval(
            valid_dataloader, prompt_encoder, text_encoder, model, kg,
            evaluator, accelerator, args, fiup_managers, id2name_for_rerank, id2genres_for_rerank, 'valid',
        )
        valid_report['epoch'] = epoch
        logger.info(f'{valid_report}')
        if run:
            run.log(valid_report)

        if valid_report['valid/loss'] < best_metric:
            prompt_encoder.save(best_metric_dir)
            best_metric = valid_report['valid/loss']
            logger.info(f'new best model with loss')

        # ── test ──────────────────────────────────────────────────────────────
        test_report = run_eval(
            test_dataloader, prompt_encoder, text_encoder, model, kg,
            evaluator, accelerator, args, fiup_managers, id2name_for_rerank, id2genres_for_rerank, 'test',
        )
        test_report['epoch'] = epoch
        logger.info(f'{test_report}')
        if run:
            run.log(test_report)

    final_dir = os.path.join(args.output_dir, 'final')
    prompt_encoder.save(final_dir)
    logger.info('save final model')