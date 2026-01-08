# SFT 数据处理流程

本指南说明如何使用 `split_data.py` 和 `autosearch_sft.py` 处理 SFT 数据。

## 完整流程

### 1. 划分数据集

将原始 JSONL 数据划分为训练集和验证集：

```bash
python scripts/data_process/split_data.py \
    --input data/autosearch_sft_raw.jsonl \
    --train_output data/autosearch_sft_train.jsonl \
    --val_output data/autosearch_sft_val.jsonl \
    --val_ratio 0.2 \
    --seed 42 \
    --stratify
```

参数说明：
- `--input`: 输入的原始 JSONL 文件
- `--train_output`: 训练集输出路径
- `--val_output`: 验证集输出路径
- `--val_ratio`: 验证集比例（默认 0.2，即 20%）
- `--seed`: 随机种子（默认 42，保证可复现）
- `--stratify`: 按类型（A/B/C）进行分层划分，保持各类别比例

### 2. 处理训练集

将训练集 JSONL 转换为 Parquet 格式（用于训练）：

```bash
python scripts/data_process/autosearch_sft.py \
    --input_file data/autosearch_sft_train.jsonl \
    --output_dir data/autosearch_sft_processed \
    --model_name Qwen/Qwen2.5-3B \
    --max_length 4096
```

注意：脚本会自动根据输入文件名识别输出文件名：
- 如果输入文件名包含 `train` → 输出 `train.parquet`
- 如果输入文件名包含 `val` 或 `test` → 输出 `val.parquet`

### 3. 处理验证集

将验证集 JSONL 转换为 Parquet 格式：

```bash
python scripts/data_process/autosearch_sft.py \
    --input_file data/autosearch_sft_val.jsonl \
    --output_dir data/autosearch_sft_processed \
    --model_name Qwen/Qwen2.5-3B \
    --max_length 4096
```

### 4. 可选：手动指定输出文件名

如果你想手动指定输出文件名，可以使用 `--output_file` 参数：

```bash
python scripts/data_process/autosearch_sft.py \
    --input_file data/autosearch_sft_train.jsonl \
    --output_dir data/autosearch_sft_processed \
    --output_file custom_name.parquet \
    --model_name Qwen/Qwen2.5-3B \
    --max_length 4096
```

### 5. 调试模式

在第一步处理时，可以使用 `--debug` 查看第一条数据的 tokenization 和 masking 详情：

```bash
python scripts/data_process/autosearch_sft.py \
    --input_file data/autosearch_sft_train.jsonl \
    --output_dir data/autosearch_sft_processed \
    --model_name Qwen/Qwen2.5-3B \
    --max_length 4096 \
    --debug
```

## 输出结果

处理完成后，在 `data/autosearch_sft_processed/` 目录下会生成：
- `train.parquet`: 训练集（包含 input_ids, labels, attention_mask）
- `val.parquet`: 验证集（包含 input_ids, labels, attention_mask）

这些文件可以直接用于 `verl` 的 SFT 训练。

## 数据统计

划分完成后，脚本会显示：
- 总数据量
- 类型分布（A/B/C）
- 训练集和验证集的数量及比例
- 各类别在训练集和验证集中的分布

示例输出：
```
Total: 1199 examples
Type distribution: {'A': 500, 'C': 199, 'B': 500}
Type A: 400 train, 100 val
Type C: 159 train, 40 val
Type B: 400 train, 100 val

Split results:
  Train: 959 examples (80.0%)
  Val: 240 examples (20.0%)
  Train type distribution: {'A': 400, 'B': 400, 'C': 159}
  Val type distribution: {'B': 100, 'C': 40, 'A': 100}
```
