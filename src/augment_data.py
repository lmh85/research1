"""
LLM 数据增强脚本
基于 ACL 2025 "Beyond Single Labels" 的思路
用 Qwen2.5-7B-Instruct 为每条对话扩充正样本标签

流程：
1. 读取 train_data_processed.jsonl
2. 对每条有 rec 标签的样本，用 Qwen 找语义相关的额外电影
3. 用 Qwen 对候选电影打相关性分，过滤低分
4. 输出 train_data_augmented.jsonl（rec 字段扩充后的版本）
"""

import json
import os
import re
import argparse
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# ── 参数 ─────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='redial_gen',
                    help='数据集名称（redial_gen 或 inspired_gen）')
parser.add_argument('--data_dir', type=str,
                    default='/root/autodl-tmp/UniCRS-main/src/data',
                    help='数据根目录')
parser.add_argument('--model_path', type=str,
                    default='/root/autodl-tmp/models/Qwen/Qwen2.5-7B-Instruct',
                    help='Qwen 模型路径')
parser.add_argument('--top_k_candidates', type=int, default=10,
                    help='每条样本最多扩充多少个候选')
parser.add_argument('--score_threshold', type=float, default=0.6,
                    help='相关性分数阈值，低于此分数的候选被过滤')
parser.add_argument('--max_samples', type=int, default=-1,
                    help='调试用，限制处理样本数，-1 表示全部')
parser.add_argument('--batch_size', type=int, default=8,
                    help='每批处理的样本数')
parser.add_argument('--skip_lines', type=int, default=0,
                    help='跳过前N行，用于断点续跑')
parser.add_argument('--output_file', type=str, default=None,
                    help='指定输出文件路径，不指定则使用默认路径')
args = parser.parse_args()

DATASET_DIR = os.path.join(args.data_dir, args.dataset)
TRAIN_FILE  = os.path.join(DATASET_DIR, 'train_data_processed.jsonl')
OUT_FILE    = args.output_file if args.output_file else os.path.join(DATASET_DIR, 'train_data_augmented.jsonl')
E2ID_FILE   = os.path.join(DATASET_DIR, 'entity2id.json')
ITEM_FILE   = os.path.join(DATASET_DIR, 'item_ids.json')

print(f'[INFO] 数据集: {args.dataset}')
print(f'[INFO] 训练文件: {TRAIN_FILE}')
print(f'[INFO] 输出文件: {OUT_FILE}')

# ── 加载数据索引 ──────────────────────────────────────────────────────────────
with open(E2ID_FILE) as f:
    e2id = json.load(f)
with open(ITEM_FILE) as f:
    item_ids_list = json.load(f)
item_ids_set = set(item_ids_list)

# 构建 entity_id -> 可读电影名 映射
id2name = {}
for uri, eid in e2id.items():
    if eid not in item_ids_set:
        continue
    name = uri.replace('<http://dbpedia.org/resource/', '').rstrip('>')
    name = re.sub(r'\s*\(.*?film.*?\)\s*$', '', name.replace('_', ' ')).strip()
    name = re.sub(r'\s*\(\d{4}\)\s*$', '', name).strip()
    if name:
        id2name[eid] = name

# 构建电影名 -> entity_id 的反向映射（用于把 LLM 输出转回 id）
name2id = {}
for eid, name in id2name.items():
    name_lower = name.lower()
    if name_lower not in name2id:
        name2id[name_lower] = eid

# 候选电影名列表（供 LLM 选择）
all_movie_names = list(id2name.values())
print(f'[INFO] 候选电影总数: {len(all_movie_names)}')

# ── 加载 Qwen 模型 ────────────────────────────────────────────────────────────
print(f'[INFO] 加载模型: {args.model_path}')
tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    args.model_path,
    torch_dtype=torch.float16,
    device_map='auto',
    trust_remote_code=True,
)
model.eval()
print('[INFO] 模型加载完成')


# ── 辅助函数 ──────────────────────────────────────────────────────────────────
def build_retrieval_prompt(context_str: str, original_movie: str,
                           candidate_pool: list) -> str:
    """
    构建语义检索 prompt：给定对话上下文和原始推荐电影，
    从候选池里找出最相关的几部电影。
    """
    # 从候选池里随机抽样 50 部作为选项（避免 prompt 太长）
    import random
    sample_pool = random.sample(candidate_pool, min(50, len(candidate_pool)))
    # 确保原始电影不在候选里
    sample_pool = [m for m in sample_pool if m.lower() != original_movie.lower()][:49]
    candidates_str = '\n'.join(f'{i+1}. {m}' for i, m in enumerate(sample_pool))

    prompt = f"""You are a movie recommendation assistant. Given a conversation and a recommended movie, find other movies that are semantically similar and would satisfy the same user request.

Conversation:
{context_str}

Original recommended movie: {original_movie}

From the following candidate movies, select up to {args.top_k_candidates} movies that are most relevant to the user's request (similar genre, theme, or style):
{candidates_str}

Output ONLY a JSON array of movie names, no explanation. Example: ["Movie A", "Movie B", "Movie C"]
If no movies are relevant, output: []"""
    return prompt, sample_pool


def build_scoring_prompt(context_str: str, candidate_movies: list) -> str:
    """
    构建相关性打分 prompt：对候选电影列表打分 0-1。
    """
    candidates_str = '\n'.join(f'{i+1}. {m}' for i, m in enumerate(candidate_movies))
    prompt = f"""You are a movie recommendation assistant. Score how relevant each movie is to the user's request in the conversation below.

Conversation:
{context_str}

Rate each movie's relevance from 0.0 to 1.0:
{candidates_str}

Output ONLY a JSON object mapping movie name to score. Example: {{"Movie A": 0.9, "Movie B": 0.3}}
Only include movies with score > {args.score_threshold}."""
    return prompt


def call_qwen(prompt: str, max_new_tokens: int = 512) -> str:
    """调用 Qwen 生成回复"""
    messages = [
        {"role": "system", "content": "You are a helpful movie recommendation assistant. Always respond with valid JSON only."},
        {"role": "user", "content": prompt}
    ]
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer([text], return_tensors='pt').to(model.device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=0.1,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
        )
    response = tokenizer.decode(
        outputs[0][inputs['input_ids'].shape[1]:],
        skip_special_tokens=True
    ).strip()
    return response


def parse_json_response(response: str):
    """从 LLM 输出中提取 JSON"""
    # 去掉 markdown 代码块
    response = re.sub(r'```json\s*', '', response)
    response = re.sub(r'```\s*', '', response)
    response = response.strip()
    try:
        return json.loads(response)
    except:
        # 尝试提取第一个 JSON 对象或数组
        match = re.search(r'[\[{].*[\]}]', response, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(0))
            except:
                pass
    return None


def augment_sample(context_str: str, original_rec_ids: list) -> list:
    """
    对单条样本做增强，返回扩充后的 rec id 列表。
    """
    if not original_rec_ids:
        return original_rec_ids

    # 获取原始推荐电影名
    original_names = [id2name.get(rid, '') for rid in original_rec_ids]
    original_names = [n for n in original_names if n]
    if not original_names:
        return original_rec_ids

    # 以第一部原始推荐电影为锚点做检索
    anchor_movie = original_names[0]

    # Step 1: 语义检索
    retrieval_prompt, sample_pool = build_retrieval_prompt(
        context_str, anchor_movie, all_movie_names
    )
    retrieval_response = call_qwen(retrieval_prompt, max_new_tokens=256)
    retrieved_names = parse_json_response(retrieval_response)

    if not retrieved_names or not isinstance(retrieved_names, list):
        return original_rec_ids

    # 过滤掉原始推荐中已有的电影
    original_names_lower = {n.lower() for n in original_names}
    retrieved_names = [n for n in retrieved_names
                       if isinstance(n, str) and n.lower() not in original_names_lower][:args.top_k_candidates]

    if not retrieved_names:
        return original_rec_ids

    # Step 2: 相关性打分
    scoring_prompt = build_scoring_prompt(context_str, retrieved_names)
    scoring_response = call_qwen(scoring_prompt, max_new_tokens=256)
    scores = parse_json_response(scoring_response)

    if not scores or not isinstance(scores, dict):
        return original_rec_ids

    # 过滤低分候选
    high_score_names = [name for name, score in scores.items()
                        if isinstance(score, (int, float)) and score >= args.score_threshold]

    # 转换为 entity_id
    augmented_ids = list(original_rec_ids)
    for name in high_score_names:
        eid = name2id.get(name.lower())
        if eid is not None and eid not in set(augmented_ids):
            augmented_ids.append(eid)

    return augmented_ids


# ── 主处理循环 ────────────────────────────────────────────────────────────────
print('[INFO] 开始处理训练数据...')

with open(TRAIN_FILE, 'r') as fin, open(OUT_FILE, 'w') as fout:
    total, augmented_count, error_count = 0, 0, 0

    for i, line in enumerate(tqdm(fin)):
        # 断点续跑：跳过已处理的行
        if i < args.skip_lines:
            continue
        if args.max_samples > 0 and i >= args.skip_lines + args.max_samples:
            break

        try:
            sample = json.loads(line)
            total += 1

            # 只对有 rec 标签的样本做增强
            if not sample.get('rec'):
                fout.write(json.dumps(sample, ensure_ascii=False) + '\n')
                continue

            # 构建对话上下文字符串
            context = sample.get('context', [])
            context_str = ''
            for j, utt in enumerate(context):
                if utt:
                    prefix = 'User: ' if j % 2 == 0 else 'System: '
                    context_str += prefix + utt + '\n'
            context_str = context_str.strip()[-1000:]  # 截断到最后1000字符

            # 增强
            original_rec = sample['rec'] if isinstance(sample['rec'], list) else [sample['rec']]
            augmented_rec = augment_sample(context_str, original_rec)

            if len(augmented_rec) > len(original_rec):
                augmented_count += 1

            sample['rec'] = augmented_rec
            fout.write(json.dumps(sample, ensure_ascii=False) + '\n')

        except Exception as e:
            error_count += 1
            # 出错时写入原始样本，不丢数据
            fout.write(line)
            if error_count <= 5:
                print(f'[WARN] 第{i}行处理出错: {e}')

        # 每100条打印一次进度
        if (i + 1) % 100 == 0:
            print(f'[INFO] 已处理 {i+1} 条，增强了 {augmented_count} 条，错误 {error_count} 条')

print(f'\n[DONE] 总计: {total} 条，增强: {augmented_count} 条，错误: {error_count} 条')
print(f'[DONE] 输出文件: {OUT_FILE}')
# 注意：以下参数已在文件顶部的argparse里添加，这里只是说明
# --skip_lines N   : 跳过前N行（用于断点续跑）
# --output_file PATH: 指定输出文件路径（不指定则覆盖默认路径）