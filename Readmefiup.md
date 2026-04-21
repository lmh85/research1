
## 完整运行命令（修改后版本）

### 第一步：数据处理（不变，与原版相同）

```bash
cd data
python dbpedia/extract_kg.py

# redial 数据集
python redial/extract_subkg.py
python redial/remove_entity.py

# inspired 数据集
python inspired/extract_subkg.py
python inspired/remove_entity.py
```

---

### 第二步：Prompt 预训练（不涉及 FIUP，命令不变）

```bash
cd src
python data/redial/process.py

export OMP_NUM_THREADS=1
accelerate launch train_pre.py \
--dataset redial \
--tokenizer microsoft/DialoGPT-small \
--model microsoft/DialoGPT-small \
--text_tokenizer roberta-base \
--text_encoder roberta-base \
--num_train_epochs 5 \
--gradient_accumulation_steps 1 \
--per_device_train_batch_size 64 \
--per_device_eval_batch_size 128 \
--num_warmup_steps 1389 \
--max_length 200 \
--prompt_max_length 200 \
--entity_max_length 32 \
--learning_rate 5e-4 \
--output_dir /root/autodl-tmp/UniCRS-main/pre_trained_prompt
```

---

### 第三步：对话任务训练（新增 `--use_fiup`）

```bash
cd src
python data/redial/process_mask.py

nohup accelerate launch train_conv.py \
--dataset redial \
--tokenizer microsoft/DialoGPT-small \
--model microsoft/DialoGPT-small \
--text_tokenizer roberta-base \
--text_encoder roberta-base \
--n_prefix_conv 20 \
--prompt_encoder /root/autodl-tmp/UniCRS-main/pre_trained_prompt/final \
--num_train_epochs 10 \
--gradient_accumulation_steps 1 \
--ignore_pad_token_for_loss \
--per_device_train_batch_size 16 \
--per_device_eval_batch_size 16 \
--num_warmup_steps 6345 \
--context_max_length 200 \
--resp_max_length 183 \
--prompt_max_length 200 \
--entity_max_length 32 \
--learning_rate 1e-4 \
--output_dir /root/autodl-tmp/UniCRS-main/output_conv_f \
--use_fiup \
--fiup_alpha 0.8 \
--sentiment_backend textblob \
> train_conv_fiup417.log 2>&1 &
```
#保存画像版
nohup accelerate launch train_conv.py \
  --dataset redial \
  --tokenizer microsoft/DialoGPT-small \
  --model microsoft/DialoGPT-small \
  --text_tokenizer roberta-base \
  --text_encoder roberta-base \
  --n_prefix_conv 20 \
  --prompt_encoder /root/autodl-tmp/UniCRS-main/pre_trained_prompt/final \
  --num_train_epochs 10 \
  --gradient_accumulation_steps 1 \
  --ignore_pad_token_for_loss \
  --per_device_train_batch_size 16 \
  --per_device_eval_batch_size 16 \
  --num_warmup_steps 6345 \
  --context_max_length 200 \
  --resp_max_length 183 \
  --prompt_max_length 200 \
  --entity_max_length 32 \
  --learning_rate 1e-4 \
  --output_dir /root/autodl-tmp/UniCRS-main/output_conv_fiup \
  --use_fiup \
  --fiup_alpha 0.8 \
  --sentiment_backend textblob \
  --fiup_states_dir /root/autodl-tmp/UniCRS-main/output_conv_fiup/fiup_states \
  > train_conv_pro421.log 2>&1 &
---

### 第四步：对话推理（不涉及 FIUP，命令不变）

```bash
nohup accelerate launch infer_conv.py \
--dataset redial \
--split test \
--tokenizer microsoft/DialoGPT-small \
--model microsoft/DialoGPT-small \
--text_tokenizer roberta-base \
--text_encoder roberta-base \
--n_prefix_conv 20 \
--prompt_encoder /root/autodl-tmp/UniCRS-main/output_conv/final \
--per_device_eval_batch_size 64 \
--context_max_length 200 \
--resp_max_length 183 \
--prompt_max_length 200 \
--entity_max_length 32 \
> infer_conv.log 2>&1 &
```

推理完成后合并生成结果：

```bash
cd src
cp -r data/redial/. data/redial_gen/
python data/redial_gen/merge.py --gen_file_prefix dialogpt_prompt-pre_prefix-20_redial_1e-4
```

---

### 第五步：推荐任务训练（新增 `--use_fiup` / `--fiup_lambda`）

```bash
nohup accelerate launch train_rec.py \
--dataset redial_gen \
--tokenizer microsoft/DialoGPT-small \
--model microsoft/DialoGPT-small \
--text_tokenizer roberta-base \
--text_encoder roberta-base \
--n_prefix_rec 10 \
--prompt_encoder /root/autodl-tmp/UniCRS-main/pre_trained_prompt/final \
--num_train_epochs 5 \
--per_device_train_batch_size 64 \
--per_device_eval_batch_size 64 \
--gradient_accumulation_steps 1 \
--num_warmup_steps 530 \
--context_max_length 200 \
--prompt_max_length 200 \
--entity_max_length 32 \
--learning_rate 1e-4 \
--output_dir /root/autodl-tmp/UniCRS-main/output_rec \
--use_fiup \
--fiup_alpha 0.8 \
--fiup_lambda 0.1 \
--sentiment_backend textblob \
> train_rec.log 2>&1 &
```

---

### 对比实验命令（不启用 FIUP，验证效果差异）

如果你需要跑一组**不带 FIUP 的基线**做对比，只需去掉三个参数即可：

```bash
# 对话训练基线（无 FIUP）
nohup accelerate launch train_conv.py \
--dataset redial \
--tokenizer microsoft/DialoGPT-small \
--model microsoft/DialoGPT-small \
--text_tokenizer roberta-base \
--text_encoder roberta-base \
--n_prefix_conv 20 \
--prompt_encoder /root/autodl-tmp/UniCRS-main/pre_trained_prompt/final \
--num_train_epochs 10 \
--gradient_accumulation_steps 1 \
--ignore_pad_token_for_loss \
--per_device_train_batch_size 16 \
--per_device_eval_batch_size 16 \
--num_warmup_steps 6345 \
--context_max_length 200 \
--resp_max_length 183 \
--prompt_max_length 200 \
--entity_max_length 32 \
--learning_rate 1e-4 \
--output_dir /root/autodl-tmp/UniCRS-main/output_conv_baseline \
> train_conv_baseline.log 2>&1 &

# 推荐训练基线（无 FIUP）
nohup accelerate launch train_rec.py \
--dataset redial_gen \
--tokenizer microsoft/DialoGPT-small \
--model microsoft/DialoGPT-small \
--text_tokenizer roberta-base \
--text_encoder roberta-base \
--n_prefix_rec 10 \
--prompt_encoder /root/autodl-tmp/UniCRS-main/pre_trained_prompt/final \
--num_train_epochs 5 \
--per_device_train_batch_size 64 \
--per_device_eval_batch_size 64 \
--gradient_accumulation_steps 1 \
--num_warmup_steps 530 \
--context_max_length 200 \
--prompt_max_length 200 \
--entity_max_length 32 \
--learning_rate 1e-4 \
--output_dir /root/autodl-tmp/UniCRS-main/output_rec_baseline \
> train_rec_baseline.log 2>&1 &
```

---

### 新增参数说明

| 参数 | 适用脚本 | 说明 |
|---|---|---|
| `--use_fiup` | `train_conv.py` / `train_rec.py` | 启用 FIUP 双库用户画像，不加此参数则完全等价于原始 UniCRS |
| `--fiup_alpha 0.8` | `train_conv.py` / `train_rec.py` | 遗忘衰减系数，值越大历史越重要，建议范围 0.7~0.9 |
| `--fiup_lambda 0.1` | `train_rec.py` | 显性库重排序融合权重，建议范围 0.05~0.2 |
| `--sentiment_backend textblob` | `train_conv.py` / `train_rec.py` | 情感分析后端，训练用 `textblob`（轻量），推理可换 `transformers`（精度高） |