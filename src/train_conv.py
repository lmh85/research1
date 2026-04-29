##train_conv.py 的 Qwen 情感分析参数加上
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
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import AdamW, get_linear_schedule_with_warmup, AutoTokenizer, AutoModel

from config import gpt2_special_tokens_dict, prompt_special_tokens_dict
from dataset_conv import CRSConvDataCollator, CRSConvDataset
from dataset_dbpedia import DBpedia
from evaluate_conv import ConvEvaluator
from model_gpt2 import PromptGPT2forCRS
from model_prompt import KGPrompt

from modules.sentiment import SentimentAnalyzer
from modules.fiup_manager import FIUPManager
from dataset_rec import build_prompt_with_fiup


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_dir", type=str)
    parser.add_argument("--debug", action='store_true')
    # data
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--context_max_length', type=int)
    parser.add_argument('--resp_max_length', type=int)
    parser.add_argument("--entity_max_length", type=int)
    parser.add_argument("--prompt_max_length", type=int)
    parser.add_argument("--tokenizer", type=str)
    parser.add_argument("--ignore_pad_token_for_loss", action='store_true')
    parser.add_argument("--text_tokenizer", type=str)
    # model
    parser.add_argument("--model", type=str)
    parser.add_argument("--max_gen_len", type=int, default=50)
    parser.add_argument("--text_encoder", type=str)
    parser.add_argument("--prompt_encoder", type=str)
    parser.add_argument("--n_prefix_conv", type=int)
    parser.add_argument("--num_bases", type=int, default=8)
    # optim
    parser.add_argument("--num_train_epochs", type=int, default=10)
    parser.add_argument("--max_train_steps", type=int, default=None)
    parser.add_argument("--per_device_train_batch_size", type=int, default=4)
    parser.add_argument("--per_device_eval_batch_size", type=int, default=4)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--learning_rate", type=float, default=1e-5)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument('--max_grad_norm', type=float)
    parser.add_argument('--num_warmup_steps', type=int, default=10000)
    parser.add_argument('--fp16', action='store_true')
    # wandb
    parser.add_argument("--use_wandb", action="store_true")
    parser.add_argument("--entity", type=str)
    parser.add_argument("--project", type=str)
    parser.add_argument("--name", type=str)
    parser.add_argument("--log_all", action="store_true")
    # FIUP
    parser.add_argument("--use_fiup", action="store_true", default=False)
    parser.add_argument("--fiup_alpha", type=float, default=0.8)
    parser.add_argument("--sentiment_backend", type=str, default="textblob",
                        choices=["textblob", "transformers", "qwen"])
    parser.add_argument("--qwen_sentiment_model", type=str, default=None,
                        help="Qwen 情感分析模型路径，backend=qwen 时必须指定")
    parser.add_argument("--fiup_states_dir", type=str, default=None,
                        help="FIUP 画像保存目录，训练结束后保存，推荐训练阶段加载")

    args = parser.parse_args()
    return args


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
            run = wandb.init(entity=args.entity, project=args.project, group=group, config=config, name=name)
        else:
            run = wandb.init(entity=args.entity, project=args.project, config=config, name=name) \
                if accelerator.is_local_main_process else None
    else:
        run = None

    if args.seed is not None:
        set_seed(args.seed)

    if args.output_dir is not None:
        os.makedirs(args.output_dir, exist_ok=True)

    kg = DBpedia(dataset=args.dataset, debug=args.debug).get_entity_kg_info()

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

    # ── [FIUP] 初始化 ─────────────────────────────────────────────────────────
    if args.use_fiup:
        sentiment_analyzer_conv = SentimentAnalyzer(
            backend=args.sentiment_backend,
            model_path=args.qwen_sentiment_model,
        )
        fiup_managers_conv: dict = {}
        EMB_DIM_CONV = text_encoder.config.hidden_size
        logger.info(f"[FIUP-Conv] Initialized. EMB_DIM={EMB_DIM_CONV}, alpha={args.fiup_alpha}")
    # ─────────────────────────────────────────────────────────────────────────

    prompt_encoder = KGPrompt(
        model.config.n_embd, text_encoder.config.hidden_size, model.config.n_head, model.config.n_layer, 2,
        n_entity=kg['num_entities'], num_relations=kg['num_relations'], num_bases=args.num_bases,
        edge_index=kg['edge_index'], edge_type=kg['edge_type'],
        n_prefix_conv=args.n_prefix_conv
    )
    if args.prompt_encoder is not None:
        prompt_encoder.load(args.prompt_encoder)
    prompt_encoder = prompt_encoder.to(device)

    for module in [model, text_encoder]:
        module.requires_grad_(False)

    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {"params": [p for m in [prompt_encoder] for n, p in m.named_parameters()
                    if not any(nd in n for nd in no_decay) and p.requires_grad],
         "weight_decay": args.weight_decay},
        {"params": [p for m in [prompt_encoder] for n, p in m.named_parameters()
                    if any(nd in n for nd in no_decay) and p.requires_grad],
         "weight_decay": 0.0},
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate)

    train_dataset = CRSConvDataset(
        args.dataset, 'train', tokenizer, debug=args.debug,
        context_max_length=args.context_max_length, resp_max_length=args.resp_max_length,
        entity_max_length=args.entity_max_length,
        prompt_tokenizer=text_tokenizer, prompt_max_length=args.prompt_max_length
    )
    valid_dataset = CRSConvDataset(
        args.dataset, 'valid', tokenizer, debug=args.debug,
        context_max_length=args.context_max_length, resp_max_length=args.resp_max_length,
        entity_max_length=args.entity_max_length,
        prompt_tokenizer=text_tokenizer, prompt_max_length=args.prompt_max_length
    )
    test_dataset = CRSConvDataset(
        args.dataset, 'test', tokenizer, debug=args.debug,
        context_max_length=args.context_max_length, resp_max_length=args.resp_max_length,
        entity_max_length=args.entity_max_length,
        prompt_tokenizer=text_tokenizer, prompt_max_length=args.prompt_max_length
    )

    data_collator_teacher = CRSConvDataCollator(
        tokenizer=tokenizer, device=device, use_amp=args.fp16, debug=args.debug, gen=False,
        ignore_pad_token_for_loss=args.ignore_pad_token_for_loss,
        context_max_length=args.context_max_length + args.resp_max_length,
        entity_max_length=args.entity_max_length, pad_entity_id=kg['pad_entity_id'],
        prompt_tokenizer=text_tokenizer
    )
    data_collator_generator = CRSConvDataCollator(
        tokenizer=tokenizer, device=device, gen=True, use_amp=args.fp16, debug=args.debug,
        ignore_pad_token_for_loss=args.ignore_pad_token_for_loss,
        context_max_length=args.context_max_length, resp_max_length=args.resp_max_length,
        entity_max_length=args.entity_max_length, pad_entity_id=kg['pad_entity_id'],
        prompt_tokenizer=text_tokenizer
    )

    train_dataloader = DataLoader(train_dataset, batch_size=args.per_device_train_batch_size,
                                  shuffle=True, num_workers=args.num_workers, collate_fn=data_collator_teacher)
    valid_dataloader = DataLoader(valid_dataset, batch_size=args.per_device_eval_batch_size,
                                  num_workers=args.num_workers, collate_fn=data_collator_teacher)
    test_dataloader  = DataLoader(test_dataset,  batch_size=args.per_device_eval_batch_size,
                                  num_workers=args.num_workers, collate_fn=data_collator_teacher)
    valid_gen_dataloader = DataLoader(valid_dataset, batch_size=args.per_device_eval_batch_size,
                                      num_workers=args.num_workers, collate_fn=data_collator_generator)
    test_gen_dataloader  = DataLoader(test_dataset,  batch_size=args.per_device_eval_batch_size,
                                      num_workers=args.num_workers, collate_fn=data_collator_generator)

    gen_file_path = os.path.join('log', f'gen_{local_time}.jsonl')
    evaluator = ConvEvaluator(tokenizer=tokenizer, log_file_path=gen_file_path)
    prompt_encoder, optimizer, train_dataloader = accelerator.prepare(prompt_encoder, optimizer, train_dataloader)

    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    else:
        args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)
    total_batch_size = args.per_device_train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps
    completed_steps = 0

    lr_scheduler = get_linear_schedule_with_warmup(optimizer, args.num_warmup_steps, args.max_train_steps)
    lr_scheduler = accelerator.prepare(lr_scheduler)

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Batch size per device = {args.per_device_train_batch_size}")
    logger.info(f"  Total train batch size = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")

    progress_bar = tqdm(range(args.max_train_steps), disable=not accelerator.is_local_main_process)

    metric, mode = 'loss', -1
    best_metric = float('inf')
    best_metric_dir = os.path.join(args.output_dir, 'best')
    os.makedirs(best_metric_dir, exist_ok=True)

    # ════════════════════════ train loop ═════════════════════════════════════
    for epoch in range(args.num_train_epochs):
        train_loss = []
        prompt_encoder.train()

        for step, batch in enumerate(train_dataloader):

            # ── [FIUP] 画像更新 + Prompt 注入 ─────────────────────────────────
            if args.use_fiup:
                batch_size = len(batch["context"]["input_ids"])
                user_ids   = batch.get("user_id", None)
                if user_ids is None or all(uid == 'unknown' for uid in user_ids):
                    user_ids = [f"u{step}_{i}" for i in range(batch_size)]
                context_strs       = batch.get("context_str",  [""] * batch_size)
                entity_names_batch = batch.get("entity_names", [[] for _ in range(batch_size)])

                orig_prompt_ids   = batch['prompt']['input_ids']
                orig_prompt_texts = text_tokenizer.batch_decode(orig_prompt_ids, skip_special_tokens=True)

                fiup_prompts = []
                for uid, ctx_str, attrs, prompt_txt in zip(user_ids, context_strs,
                                                            entity_names_batch, orig_prompt_texts):
                    if uid not in fiup_managers_conv:
                        fiup_managers_conv[uid] = FIUPManager(emb_dim=EMB_DIM_CONV, alpha=args.fiup_alpha)
                    mgr   = fiup_managers_conv[uid]
                    e_tau = sentiment_analyzer_conv.score(ctx_str)

                    tokenized = text_tokenizer(
                        ctx_str, return_tensors="pt",
                        max_length=128, truncation=True, padding=True
                    ).to(device)
                    with torch.no_grad():
                        context_emb = text_encoder(**tokenized).last_hidden_state[:, 0, :].squeeze(0).cpu()

                    mgr.update_profile(attrs, e_tau, context_emb)
                    fiup_prompts.append(mgr.build_profile_prompt())

                new_prompt_texts = [
                    f"[USER_PROFILE] {fp} [SEP] {orig}" if fp else orig
                    for orig, fp in zip(orig_prompt_texts, fiup_prompts)
                ]
                new_prompt_batch = text_tokenizer(
                    new_prompt_texts, max_length=args.prompt_max_length,
                    padding="max_length", truncation=True, return_tensors="pt"
                ).to(device)
                batch['prompt']['input_ids']      = new_prompt_batch['input_ids']
                batch['prompt']['attention_mask'] = new_prompt_batch['attention_mask']
            # ── [FIUP] 结束 ───────────────────────────────────────────────────

            with torch.no_grad():
                token_embeds = text_encoder(**batch['prompt']).last_hidden_state
            prompt_embeds = prompt_encoder(
                entity_ids=batch['entity'], token_embeds=token_embeds,
                output_entity=False, use_conv_prefix=True
            )
            batch['context']['prompt_embeds'] = prompt_embeds

            loss = model(**batch['context'], conv=True,
                         conv_labels=batch['resp']).conv_loss / args.gradient_accumulation_steps
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

        train_loss = np.mean(train_loss) * args.gradient_accumulation_steps
        logger.info(f'epoch {epoch} train loss {train_loss}')
        del train_loss, batch

        # ── valid ─────────────────────────────────────────────────────────────
        valid_loss = []
        prompt_encoder.eval()
        for batch in tqdm(valid_dataloader, disable=not accelerator.is_local_main_process):
            with torch.no_grad():
                token_embeds  = text_encoder(**batch['prompt']).last_hidden_state
                prompt_embeds = prompt_encoder(entity_ids=batch['entity'], token_embeds=token_embeds,
                                               output_entity=False, use_conv_prefix=True)
                batch['context']['prompt_embeds'] = prompt_embeds
                loss = model(**batch['context'], conv=True, conv_labels=batch['resp']).conv_loss
                valid_loss.append(float(loss))

        evaluator.log_file.write(f'\n\n*** valid-{evaluator.log_cnt} ***\n\n')
        for batch in tqdm(valid_gen_dataloader, disable=not accelerator.is_local_main_process):
            with torch.no_grad():
                token_embeds  = text_encoder(**batch['prompt']).last_hidden_state
                prompt_embeds = prompt_encoder(entity_ids=batch['entity'], token_embeds=token_embeds,
                                               output_entity=False, use_conv_prefix=True)
                batch['context']['prompt_embeds'] = prompt_embeds
                gen_seqs = accelerator.unwrap_model(model).generate(
                    **batch['context'], max_new_tokens=args.max_gen_len, no_repeat_ngram_size=3)
                gen_resp_ids = []
                for gen_seq, length in zip(gen_seqs, batch['context_len']):
                    gen_seq = [t for t in gen_seq if t != tokenizer.pad_token_id]
                    gen_resp_ids.append(gen_seq[length:])
                evaluator.evaluate(gen_resp_ids, batch['resp'], log=accelerator.is_local_main_process)

        accelerator.wait_for_everyone()
        report = evaluator.report()
        valid_report = {f'valid/{k}': v for k, v in report.items()}
        valid_report['valid/loss'] = np.mean(valid_loss)
        valid_report['epoch'] = epoch
        logger.info(valid_report)
        if run:
            run.log(valid_report)
        evaluator.reset_metric()

        if valid_report['valid/loss'] < best_metric:
            best_metric = valid_report['valid/loss']
            prompt_encoder.save(best_metric_dir)
            logger.info('new best model with loss')

        # ── test ──────────────────────────────────────────────────────────────
        test_loss = []
        for batch in tqdm(test_dataloader, disable=not accelerator.is_local_main_process):
            with torch.no_grad():
                token_embeds  = text_encoder(**batch['prompt']).last_hidden_state
                prompt_embeds = prompt_encoder(entity_ids=batch['entity'], token_embeds=token_embeds,
                                               output_entity=False, use_conv_prefix=True)
                batch['context']['prompt_embeds'] = prompt_embeds
                loss = model(**batch['context'], conv=True, conv_labels=batch['resp']).conv_loss
                test_loss.append(float(loss))

        evaluator.log_file.write(f'\n*** test-{evaluator.log_cnt} ***\n\n')
        for batch in tqdm(test_gen_dataloader, disable=not accelerator.is_local_main_process):
            with torch.no_grad():
                token_embeds  = text_encoder(**batch['prompt']).last_hidden_state
                prompt_embeds = prompt_encoder(entity_ids=batch['entity'], token_embeds=token_embeds,
                                               output_entity=False, use_conv_prefix=True)
                batch['context']['prompt_embeds'] = prompt_embeds
                gen_seqs = accelerator.unwrap_model(model).generate(
                    **batch['context'], max_new_tokens=args.max_gen_len, no_repeat_ngram_size=3)
                gen_resp_ids = []
                for gen_seq, length in zip(gen_seqs, batch['context_len']):
                    gen_seq = [t for t in gen_seq if t != tokenizer.pad_token_id]
                    gen_resp_ids.append(gen_seq[length:])
                evaluator.evaluate(gen_resp_ids, batch['resp'], log=accelerator.is_local_main_process)

        accelerator.wait_for_everyone()
        report = evaluator.report()
        test_report = {f'test/{k}': v for k, v in report.items()}
        test_report['test/loss'] = np.mean(test_loss)
        test_report['epoch'] = epoch
        logger.info(test_report)
        if run:
            run.log(test_report)
        evaluator.reset_metric()
        evaluator.log_cnt += 1

    # ── [FIUP] 训练结束后保存画像状态 ─────────────────────────────────────────
    if args.use_fiup and fiup_managers_conv:
        states_dir = args.fiup_states_dir or os.path.join(args.output_dir, 'fiup_states')
        os.makedirs(states_dir, exist_ok=True)
        saved = 0
        for uid, mgr in fiup_managers_conv.items():
            try:
                mgr.save(os.path.join(states_dir, f'{uid}.json'))
                saved += 1
            except Exception as e:
                logger.warning(f'[FIUP] Failed to save {uid}: {e}')
        logger.info(f'[FIUP] Saved {saved} user profiles to {states_dir}')
    # ─────────────────────────────────────────────────────────────────────────

    final_dir = os.path.join(args.output_dir, 'final')
    prompt_encoder.save(final_dir)
    logger.info('save final model')




# import argparse
# import math
# import os
# import sys
# import time

# import numpy as np
# import torch
# import transformers
# import wandb
# from accelerate import Accelerator
# from accelerate.utils import set_seed
# from loguru import logger
# from torch.utils.data import DataLoader
# from tqdm.auto import tqdm
# from transformers import AdamW, get_linear_schedule_with_warmup, AutoTokenizer, AutoModel

# from config import gpt2_special_tokens_dict, prompt_special_tokens_dict
# from dataset_conv import CRSConvDataCollator, CRSConvDataset
# from dataset_dbpedia import DBpedia
# from evaluate_conv import ConvEvaluator
# from model_gpt2 import PromptGPT2forCRS
# from model_prompt import KGPrompt

# from modules.sentiment import SentimentAnalyzer    # [FIUP-NEW]
# from modules.fiup_manager import FIUPManager        # [FIUP-NEW]
# # build_prompt_with_fiup 仅推荐任务（train_rec.py）使用，对话任务不注入 Prompt


# def parse_args():
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--seed", type=int, default=42, help="A seed for reproducible training.")
#     parser.add_argument("--output_dir", type=str, help="Where to store the final model.")
#     parser.add_argument("--debug", action='store_true', help="Debug mode.")
#     # data
#     parser.add_argument("--dataset", type=str, required=True, help="A file containing all data.")
#     parser.add_argument('--num_workers', type=int, default=0)
#     parser.add_argument('--context_max_length', type=int, help="max length of both encoder and decoder input.")
#     parser.add_argument('--resp_max_length', type=int, help="max length of decoder input.")
#     parser.add_argument("--entity_max_length", type=int, help="max entity length in dataset.")
#     parser.add_argument("--prompt_max_length", type=int)
#     parser.add_argument("--tokenizer", type=str)
#     parser.add_argument("--ignore_pad_token_for_loss", action='store_true')
#     parser.add_argument("--text_tokenizer", type=str)
#     # model
#     parser.add_argument("--model", type=str)
#     parser.add_argument("--max_gen_len", type=int, default=50)
#     parser.add_argument("--text_encoder", type=str)
#     parser.add_argument("--prompt_encoder", type=str)
#     parser.add_argument("--n_prefix_conv", type=int)
#     parser.add_argument("--num_bases", type=int, default=8, help="num_bases in RGCN")
#     # optim
#     parser.add_argument("--num_train_epochs", type=int, default=10, help="Total number of training epochs to perform.")
#     parser.add_argument("--max_train_steps", type=int, default=None,
#                         help="Total number of training steps to perform. If provided, overrides num_train_epochs.")
#     parser.add_argument("--per_device_train_batch_size", type=int, default=4,
#                         help="Batch size (per device) for the training dataloader.")
#     parser.add_argument("--per_device_eval_batch_size", type=int, default=4,
#                         help="Batch size (per device) for the evaluation dataloader.")
#     parser.add_argument("--gradient_accumulation_steps", type=int, default=1,
#                         help="Number of updates steps to accumulate before performing a backward/update pass.")
#     parser.add_argument("--learning_rate", type=float, default=1e-5,
#                         help="Initial learning rate (after the potential warmup period) to use.")
#     parser.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay to use.")
#     parser.add_argument('--max_grad_norm', type=float)
#     parser.add_argument('--num_warmup_steps', type=int, default=10000)
#     parser.add_argument('--fp16', action='store_true', help='use automatic mixed precision to speed up.')
#     # wandb
#     parser.add_argument("--use_wandb", action="store_true", help="whether to use wandb")
#     parser.add_argument("--entity", type=str, help="wandb username")
#     parser.add_argument("--project", type=str, help="wandb exp project")
#     parser.add_argument("--name", type=str, help="wandb exp name")
#     parser.add_argument("--log_all", action="store_true", help="log in all processes, otherwise only in rank0")

#     # FIUP 
#     parser.add_argument("--use_fiup", action="store_true", default=False,
#                         help="启用 FIUP 双库用户画像模块")
#     parser.add_argument("--fiup_alpha", type=float, default=0.8,
#                         help="FIUP 遗忘衰减系数")
#     parser.add_argument("--sentiment_backend", type=str, default="textblob",
#                         choices=["textblob", "transformers"],
#                         help="情感分析后端")
#     parser.add_argument("--fiup_states_dir", type=str, default=None,
#                         help="FIUP 画像状态保存目录；训练结束后序列化所有用户画像，"
#                              "供 train_rec.py 加载复用。默认为 output_dir/fiup_states")

#     args = parser.parse_args()
#     return args


# if __name__ == '__main__':
#     args = parse_args()
#     config = vars(args)

#     # Initialize the accelerator. We will let the accelerator handle device placement for us.
#     # accelerator = Accelerator(device_placement=False, fp16=args.fp16)
#     mixed_precision = "fp16" if args.fp16 else "no"
#     accelerator = Accelerator(device_placement=False, mixed_precision=mixed_precision)  
#     device = accelerator.device


#     # Make one log on every process with the configuration for debugging.
#     local_time = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
#     logger.remove()
#     logger.add(sys.stderr, level='DEBUG' if accelerator.is_local_main_process else 'ERROR')
#     logger.add(f'log/{local_time}.log', level='DEBUG' if accelerator.is_local_main_process else 'ERROR')
#     logger.info(accelerator.state)
#     logger.info(config)

#     if accelerator.is_local_main_process:
#         transformers.utils.logging.set_verbosity_info()
#     else:
#         transformers.utils.logging.set_verbosity_error()
#     # wandb
#     if args.use_wandb:
#         name = args.name if args.name else local_time
#         name += '_' + str(accelerator.process_index)

#         if args.log_all:
#             group = args.name if args.name else 'DDP_' + local_time
#             run = wandb.init(entity=args.entity, project=args.project, group=group, config=config, name=name)
#         else:
#             if accelerator.is_local_main_process:
#                 run = wandb.init(entity=args.entity, project=args.project, config=config, name=name)
#             else:
#                 run = None
#     else:
#         run = None

#     # If passed along, set the training seed now.
#     if args.seed is not None:
#         set_seed(args.seed)

#     if args.output_dir is not None:
#         os.makedirs(args.output_dir, exist_ok=True)

#     kg = DBpedia(dataset=args.dataset, debug=args.debug).get_entity_kg_info()

#     tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
#     tokenizer.add_special_tokens(gpt2_special_tokens_dict)
#     model = PromptGPT2forCRS.from_pretrained(args.model)
#     model.resize_token_embeddings(len(tokenizer))
#     model.config.pad_token_id = tokenizer.pad_token_id
#     model = model.to(device)

#     text_tokenizer = AutoTokenizer.from_pretrained(args.text_tokenizer)
#     text_tokenizer.add_special_tokens(prompt_special_tokens_dict)
#     text_encoder = AutoModel.from_pretrained(args.text_encoder)
#     text_encoder.resize_token_embeddings(len(text_tokenizer))
#     text_encoder = text_encoder.to(device)

#     # ── [FIUP] 对话训练 FIUP 初始化 ──────────────────────────────────────────
#     if args.use_fiup:
#         sentiment_analyzer_conv = SentimentAnalyzer(backend=args.sentiment_backend)
#         fiup_managers_conv: dict = {}
#         EMB_DIM_CONV = text_encoder.config.hidden_size
#         logger.info(f"[FIUP-Conv] Initialized. EMB_DIM={EMB_DIM_CONV}, alpha={args.fiup_alpha}")
#     # ──────────────────────────────────────────────────────────────────────────

#     prompt_encoder = KGPrompt(
#         model.config.n_embd, text_encoder.config.hidden_size, model.config.n_head, model.config.n_layer, 2,
#         n_entity=kg['num_entities'], num_relations=kg['num_relations'], num_bases=args.num_bases,
#         edge_index=kg['edge_index'], edge_type=kg['edge_type'],
#         n_prefix_conv=args.n_prefix_conv
#     )
#     if args.prompt_encoder is not None:
#         prompt_encoder.load(args.prompt_encoder)
#     prompt_encoder = prompt_encoder.to(device)

#     fix_modules = [model, text_encoder]
#     for module in fix_modules:
#         module.requires_grad_(False)

#     # optim & amp
#     modules = [prompt_encoder]
#     no_decay = ["bias", "LayerNorm.weight"]
#     optimizer_grouped_parameters = [
#         {
#             "params": [p for model in modules for n, p in model.named_parameters()
#                        if not any(nd in n for nd in no_decay) and p.requires_grad],
#             "weight_decay": args.weight_decay,
#         },
#         {
#             "params": [p for model in modules for n, p in model.named_parameters()
#                        if any(nd in n for nd in no_decay) and p.requires_grad],
#             "weight_decay": 0.0,
#         },
#     ]
#     optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate)
#     # data
#     train_dataset = CRSConvDataset(
#         args.dataset, 'train', tokenizer, debug=args.debug,
#         context_max_length=args.context_max_length, resp_max_length=args.resp_max_length,
#         entity_max_length=args.entity_max_length,
#         prompt_tokenizer=text_tokenizer, prompt_max_length=args.prompt_max_length
#     )
#     valid_dataset = CRSConvDataset(
#         args.dataset, 'valid', tokenizer, debug=args.debug,
#         context_max_length=args.context_max_length, resp_max_length=args.resp_max_length,
#         entity_max_length=args.entity_max_length,
#         prompt_tokenizer=text_tokenizer, prompt_max_length=args.prompt_max_length
#     )
#     test_dataset = CRSConvDataset(
#         args.dataset, 'test', tokenizer, debug=args.debug,
#         context_max_length=args.context_max_length, resp_max_length=args.resp_max_length,
#         entity_max_length=args.entity_max_length,
#         prompt_tokenizer=text_tokenizer, prompt_max_length=args.prompt_max_length
#     )
#     # dataloader
#     data_collator_teacher = CRSConvDataCollator(
#         tokenizer=tokenizer, device=device, use_amp=args.fp16, debug=args.debug, gen=False,
#         ignore_pad_token_for_loss=args.ignore_pad_token_for_loss,
#         context_max_length=args.context_max_length + args.resp_max_length,
#         entity_max_length=args.entity_max_length, pad_entity_id=kg['pad_entity_id'],
#         prompt_tokenizer=text_tokenizer
#     )
#     train_dataloader = DataLoader(
#         train_dataset,
#         batch_size=args.per_device_train_batch_size,
#         shuffle=True,
#         num_workers=args.num_workers,
#         collate_fn=data_collator_teacher,
#     )
#     valid_dataloader = DataLoader(
#         valid_dataset,
#         batch_size=args.per_device_eval_batch_size,
#         num_workers=args.num_workers,
#         collate_fn=data_collator_teacher,
#     )
#     test_dataloader = DataLoader(
#         test_dataset,
#         batch_size=args.per_device_eval_batch_size,
#         num_workers=args.num_workers,
#         collate_fn=data_collator_teacher,
#     )
#     data_collator_generator = CRSConvDataCollator(
#         tokenizer=tokenizer, device=device, gen=True, use_amp=args.fp16, debug=args.debug,
#         ignore_pad_token_for_loss=args.ignore_pad_token_for_loss,
#         context_max_length=args.context_max_length, resp_max_length=args.resp_max_length,
#         entity_max_length=args.entity_max_length, pad_entity_id=kg['pad_entity_id'],
#         prompt_tokenizer=text_tokenizer
#     )
#     valid_gen_dataloader = DataLoader(
#         valid_dataset,
#         batch_size=args.per_device_eval_batch_size,
#         num_workers=args.num_workers,
#         collate_fn=data_collator_generator,
#     )
#     test_gen_dataloader = DataLoader(
#         test_dataset,
#         batch_size=args.per_device_eval_batch_size,
#         num_workers=args.num_workers,
#         collate_fn=data_collator_generator,
#     )
#     gen_file_path = os.path.join('log', f'gen_{local_time}.jsonl')
#     evaluator = ConvEvaluator(tokenizer=tokenizer, log_file_path=gen_file_path)
#     prompt_encoder, optimizer, train_dataloader = accelerator.prepare(prompt_encoder, optimizer, train_dataloader)
#     # step, epoch, batch size
#     num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
#     if args.max_train_steps is None:
#         args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
#     else:
#         args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)
#     total_batch_size = args.per_device_train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps
#     completed_steps = 0
#     # lr_scheduler
#     lr_scheduler = get_linear_schedule_with_warmup(optimizer, args.num_warmup_steps, args.max_train_steps)
#     lr_scheduler = accelerator.prepare(lr_scheduler)
#     # training info
#     logger.info("***** Running training *****")
#     logger.info(f"  Num examples = {len(train_dataset)}")
#     logger.info(f"  Num Epochs = {args.num_train_epochs}")
#     logger.info(f"  Instantaneous batch size per device = {args.per_device_train_batch_size}")
#     logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
#     logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
#     logger.info(f"  Total optimization steps = {args.max_train_steps}")
#     # Only show the progress bar once on each machine.
#     progress_bar = tqdm(range(args.max_train_steps), disable=not accelerator.is_local_main_process)

#     # save model with best metric
#     metric, mode = 'loss', -1
#     assert mode in (-1, 1)
#     if mode == 1:
#         best_metric = 0
#     else:
#         best_metric = float('inf')
#     best_metric_dir = os.path.join(args.output_dir, 'best')
#     os.makedirs(best_metric_dir, exist_ok=True)

#     # train loop
#     for epoch in range(args.num_train_epochs):
#         train_loss = []
#         prompt_encoder.train()
#         for step, batch in enumerate(train_dataloader):
#             # ── [FIUP] 对话阶段：静默积累用户画像，不修改 Prompt ─────────────
#             # 设计原则：FIUP 的目标是提升推荐性能。
#             # 对话训练阶段只负责建立画像状态（显性库 + 隐性库），
#             # 不向 Prompt 注入任何内容，保证对话任务的 BLEU/Distinct 不受影响。
#             # 积累好的 fiup_managers_conv 状态将在推荐任务训练阶段被复用。
#             if args.use_fiup:
#                 batch_size         = len(batch["context"]["input_ids"])
#                 user_ids           = batch.get("user_id",      [f"u{step}_{i}" for i in range(batch_size)])
#                 context_strs       = batch.get("context_str",  [""] * batch_size)
#                 entity_names_batch = batch.get("entity_names", [[] for _ in range(batch_size)])

#                 for uid, ctx_str, attrs in zip(user_ids, context_strs, entity_names_batch):
#                     if uid not in fiup_managers_conv:
#                         fiup_managers_conv[uid] = FIUPManager(emb_dim=EMB_DIM_CONV, alpha=args.fiup_alpha)
#                     mgr = fiup_managers_conv[uid]

#                     # ① 情感评分
#                     e_tau = sentiment_analyzer_conv.score(ctx_str)

#                     # ② 上下文语义向量（frozen RoBERTa，无梯度）
#                     tokenized = text_tokenizer(
#                         ctx_str,
#                         return_tensors="pt",
#                         max_length=128,
#                         truncation=True,
#                         padding=True,
#                     ).to(device)
#                     with torch.no_grad():
#                         context_emb = text_encoder(**tokenized).last_hidden_state[:, 0, :].squeeze(0).cpu()

#                     # ③ 更新双库（显性库 + 隐性库）—— 不调用 build_profile_prompt，不修改 batch
#                     mgr.update_profile(attrs, e_tau, context_emb)

#                 # batch['prompt'] 保持原样，对话任务训练流程完全不受影响
#             # ── [FIUP] 结束 ──────────────────────────────────────────────────
#             with torch.no_grad():
#                 token_embeds = text_encoder(**batch['prompt']).last_hidden_state
#             prompt_embeds = prompt_encoder(
#                 entity_ids=batch['entity'],
#                 token_embeds=token_embeds,
#                 output_entity=False,
#                 use_conv_prefix=True
#             )
#             batch['context']['prompt_embeds'] = prompt_embeds

#             loss = model(**batch['context'], conv=True,
#                          conv_labels=batch['resp']).conv_loss / args.gradient_accumulation_steps
#             accelerator.backward(loss)
#             train_loss.append(float(loss))
#             # optim step
#             if step % args.gradient_accumulation_steps == 0 or step == len(train_dataloader) - 1:
#                 if args.max_grad_norm is not None:
#                     accelerator.clip_grad_norm_(prompt_encoder.parameters(), args.max_grad_norm)
#                 optimizer.step()
#                 lr_scheduler.step()
#                 optimizer.zero_grad()

#                 progress_bar.update(1)
#                 completed_steps += 1
#                 if run:
#                     run.log({'loss': np.mean(train_loss) * args.gradient_accumulation_steps})

#             if completed_steps >= args.max_train_steps:
#                 break

#         # metric
#         train_loss = np.mean(train_loss) * args.gradient_accumulation_steps
#         logger.info(f'epoch {epoch} train loss {train_loss}')

#         del train_loss, batch

#         # dev
#         valid_loss = []
#         prompt_encoder.eval()
#         for batch in tqdm(valid_dataloader, disable=not accelerator.is_local_main_process):
#             with torch.no_grad():
#                 token_embeds = text_encoder(**batch['prompt']).last_hidden_state
#                 prompt_embeds = prompt_encoder(
#                     entity_ids=batch['entity'],
#                     token_embeds=token_embeds,
#                     output_entity=False,
#                     use_conv_prefix=True
#                 )
#                 batch['context']['prompt_embeds'] = prompt_embeds

#                 loss = model(**batch['context'], conv=True, conv_labels=batch['resp']).conv_loss
#                 valid_loss.append(float(loss))

#         evaluator.log_file.write(f'\n\n*** valid-{evaluator.log_cnt} ***\n\n')
#         for batch in tqdm(valid_gen_dataloader, disable=not accelerator.is_local_main_process):
#             with torch.no_grad():
#                 token_embeds = text_encoder(**batch['prompt']).last_hidden_state
#                 prompt_embeds = prompt_encoder(
#                     entity_ids=batch['entity'],
#                     token_embeds=token_embeds,
#                     output_entity=False,
#                     use_conv_prefix=True
#                 )
#                 batch['context']['prompt_embeds'] = prompt_embeds

#                 gen_seqs = accelerator.unwrap_model(model).generate(
#                     **batch['context'],
#                     max_new_tokens=args.max_gen_len,
#                     no_repeat_ngram_size=3
#                 )
#                 gen_resp_ids = []
#                 for gen_seq, length in zip(gen_seqs, batch['context_len']):
#                     gen_seq = [token_id for token_id in gen_seq if token_id != tokenizer.pad_token_id]
#                     gen_resp_ids.append(gen_seq[length:])
#                 evaluator.evaluate(gen_resp_ids, batch['resp'], log=accelerator.is_local_main_process)

#         # metric
#         accelerator.wait_for_everyone()
#         report = evaluator.report()
#         valid_report = {}
#         for k, v in report.items():
#             valid_report[f'valid/{k}'] = v
#         valid_loss = np.mean(valid_loss)
#         valid_report['valid/loss'] = valid_loss
#         valid_report['epoch'] = epoch
#         logger.info(valid_report)
#         if run:
#             run.log(valid_report)
#         evaluator.reset_metric()

#         if valid_report[f'valid/{metric}'] * mode > best_metric * mode:
#             best_metric = valid_report[f'valid/{metric}']
#             prompt_encoder.save(best_metric_dir)
#             logger.info(f'new best model with {metric}')

#         # test
#         test_loss = []
#         prompt_encoder.eval()
#         for batch in tqdm(test_dataloader, disable=not accelerator.is_local_main_process):
#             with torch.no_grad():
#                 token_embeds = text_encoder(**batch['prompt']).last_hidden_state
#                 prompt_embeds = prompt_encoder(
#                     entity_ids=batch['entity'],
#                     token_embeds=token_embeds,
#                     output_entity=False,
#                     use_conv_prefix=True
#                 )
#                 batch['context']['prompt_embeds'] = prompt_embeds

#                 loss = model(**batch['context'], conv=True, conv_labels=batch['resp']).conv_loss
#                 test_loss.append(float(loss))

#         evaluator.log_file.write(f'\n*** test-{evaluator.log_cnt} ***\n\n')
#         for batch in tqdm(test_gen_dataloader, disable=not accelerator.is_local_main_process):
#             with torch.no_grad():
#                 token_embeds = text_encoder(**batch['prompt']).last_hidden_state
#                 prompt_embeds = prompt_encoder(
#                     entity_ids=batch['entity'],
#                     token_embeds=token_embeds,
#                     output_entity=False,
#                     use_conv_prefix=True
#                 )
#                 batch['context']['prompt_embeds'] = prompt_embeds

#                 gen_seqs = accelerator.unwrap_model(model).generate(
#                     **batch['context'],
#                     max_new_tokens=args.max_gen_len,
#                     no_repeat_ngram_size=3,
#                 )
#                 gen_resp_ids = []
#                 for gen_seq, length in zip(gen_seqs, batch['context_len']):
#                     gen_seq = [token_id for token_id in gen_seq if token_id != tokenizer.pad_token_id]
#                     gen_resp_ids.append(gen_seq[length:])
#                 evaluator.evaluate(gen_resp_ids, batch['resp'], log=accelerator.is_local_main_process)

#         # metric
#         accelerator.wait_for_everyone()
#         report = evaluator.report()
#         test_report = {}
#         for k, v in report.items():
#             test_report[f'test/{k}'] = v
#         test_loss = np.mean(test_loss)
#         test_report['test/loss'] = test_loss
#         test_report['epoch'] = epoch
#         logger.info(test_report)
#         if run:
#             run.log(test_report)
#         evaluator.reset_metric()

#         evaluator.log_cnt += 1

#     final_dir = os.path.join(args.output_dir, 'final')
#     prompt_encoder.save(final_dir)
#     logger.info(f'save final model')

#     # ── [FIUP] 训练结束：序列化所有用户画像状态 ──────────────────────────────
#     # 保存后 train_rec.py 可通过 --fiup_states_dir 加载，
#     # 在对话阶段积累的画像基础上继续更新，避免推荐阶段从零重建。
#     if args.use_fiup and accelerator.is_local_main_process:
#         states_dir = args.fiup_states_dir or os.path.join(args.output_dir, 'fiup_states')
#         os.makedirs(states_dir, exist_ok=True)
#         saved_count = 0
#         for uid, mgr in fiup_managers_conv.items():
#             # uid 可能含特殊字符，做简单转义保证文件名合法
#             safe_uid = str(uid).replace('/', '_').replace('\\', '_')
#             mgr.save(os.path.join(states_dir, f'{safe_uid}.json'))
#             saved_count += 1
#         logger.info(f'[FIUP] Saved {saved_count} user profile states to {states_dir}')
#     # ─────────────────────────────────────────────────────────────────────────