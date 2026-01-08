# AutoSearch SFT 训练指南

## 数据格式说明

**重要**：`verl` 的 `SFTDataset` 期望 Parquet 文件包含**原始文本**列，而不是已 tokenized 的数据。

需要的列：
- `question`：用户问题（对应 `prompt_key`）
- `answer`：助手回复（对应 `response_key`）

`SFTDataset` 会在训练时自动进行 tokenization，并应用正确的 label masking。

## 训练步骤

### 1. 确认数据格式

检查生成的 parquet 文件是否包含 `question` 和 `answer` 列：

```bash
python -c "import pandas as pd; df = pd.read_parquet('data/autosearch_sft_processed/train.parquet'); print('Columns:', df.columns.tolist()); print('Sample row:', df.iloc[0].to_dict())"
```

如果只有 `input_ids`, `labels`, `attention_mask`，需要重新处理数据。

### 2. 修改数据处理脚本（如果需要）

最新的 `autosearch_sft.py` 已经更新，会同时保存原始文本和 tokenized 数据。

如果需要重新处理数据：

```bash
# 处理训练集
python scripts/data_process/autosearch_sft.py \
    --input_file data/autosearch_sft_train.jsonl \
    --output_dir data/autosearch_sft_processed \
    --model_name /home/hadoop-ai-search/dolphinfs_ssd_hadoop-ai-search/chongwenyue/model/Qwen2.5-3B-Instruct \
    --max_length 4096

# 处理验证集
python scripts/data_process/autosearch_sft.py \
    --input_file data/autosearch_sft_val.jsonl \
    --output_dir data/autosearch_sft_processed \
    --model_name /home/hadoop-ai-search/dolphinfs_ssd_hadoop-ai-search/chongwenyue/model/Qwen2.5-3B-Instruct \
    --max_length 4096
```

### 3. 修改配置文件

编辑 `verl/trainer/config/autosearch_sft.yaml`，确认：

```yaml
data:
  train_files: data/autosearch_sft_processed/train.parquet
  val_files: data/autosearch_sft_processed/val.parquet
  prompt_key: question  # 对应 parquet 中的列名
  response_key: answer  # 对应 parquet 中的列名
  
model:
  partial_pretrain: /home/hadoop-ai-search/dolphinfs_ssd_hadoop-ai-search/chongwenyue/model/Qwen2.5-3B-Instruct
```

### 4. 运行训练

```bash
# 单卡训练
torchrun --nproc_per_node=1 --standalone verl/trainer/fsdp_sft_trainer.py \
    --config-path=config \
    --config-name=autosearch_sft

# 8 卡训练（推荐）
bash train_autosearch_sft.sh
```

或者手动运行：

```bash
cd verl/trainer
PYTHONUNBUFFERED=1 torchrun --nproc_per_node=8 --standalone \
    python -m verl.trainer.fsdp_sft_trainer \
    --config-path=config \
    --config-name=autosearch_sft \
    trainer.project_name='AutoSearch-SFT' \
    trainer.experiment_name='autosearch-sft-qwen2.5-3b-v1' \
    2>&1 | tee ../../autosearch-sft-qwen2.5-3b-v1.log
```

### 5. 监控训练

训练日志会输出到控制台和 `wandb`（如果配置了）。

检查点会保存到 `checkpoints/autosearch_sft/` 目录。

## 常见问题

### Q: 为什么需要原始文本格式？

A: `SFTDataset` 会在训练时根据配置动态进行 tokenization，这样可以：
- 灵活调整 tokenization 参数
- 正确应用 chat template（如果需要）
- 统一数据处理流程

### Q: 如何处理已生成的数据？

A: 如果数据已经生成为 tokenized 格式，有两个选择：
1. 重新运行数据处理脚本（已更新，会同时保存原始文本）
2. 或者创建一个自定义 Dataset 类来直接使用 tokenized 数据（更复杂）

### Q: 如何调整 batch size？

A: 在 `verl/trainer/config/autosearch_sft.yaml` 中修改：
```yaml
data:
  train_batch_size: 64  # 总 batch size
  micro_batch_size: 4   # 每个 GPU 的 batch size
```

`gradient_accumulation_steps = train_batch_size / (micro_batch_size * num_gpus)`

### Q: 如何使用 LoRA？

A: 在配置文件中修改：
```yaml
model:
  lora_rank: 32  # 设置为非 0 值启用 LoRA
  lora_alpha: 16
  target_modules: [q_proj, v_proj, k_proj, o_proj]  # 根据需要调整
```
