# AutoSearch SFT V2 训练指南

## 数据统计分析

基于 `autosearch_sft_final.jsonl`（5489条数据）：

### 数据来源分布
- **r1_rewritten** (GPT-4生成): 4790条 (87.3%)
- **旧数据** (Type A/C): 699条 (12.7%)

### 数据类型分布
- **Type A** (Internal Recall): 500条 (9.1%)
- **Type C** (Trap/Hallucination): 199条 (3.6%)
- **r1_rewritten** (无type字段): 4790条 (87.3%)

### 数据源分布
- **hotpotqa**: 2567条 (46.8%)
- **nq**: 2223条 (40.5%)
- **其他**: 699条 (12.7%)

### Action分布
- **包含<recall>**: 1577条 (28.7%) ✅ 相比之前有明显提升
- **包含<search>**: 4048条 (73.7%)

### Action模式
- **search_only**: 3912条 (71.3%)
- **recall_only**: 1441条 (26.3%)
- **mixed** (recall+search): 136条 (2.5%) ✅ 包含混合推理模式

## 训练步骤

### 步骤1: 分割数据集 (train/val)

```bash
python scripts/data_process/split_data.py \
    --input data/autosearch_sft_final.jsonl \
    --train_output data/autosearch_sft_final_train.jsonl \
    --val_output data/autosearch_sft_final_val.jsonl \
    --val_ratio 0.2 \
    --seed 42 \
    --stratify
```

**说明**：
- `--val_ratio 0.2`: 验证集占20%（约1100条）
- `--stratify`: 按数据来源/类型分层划分，保持分布平衡
- 如果split_data.py不支持stratify，可以不加这个参数

### 步骤2: 转换为Parquet格式

**处理训练集**：
```bash
python scripts/data_process/autosearch_sft.py \
    --input_file data/autosearch_sft_final_train.jsonl \
    --output_dir data/autosearch_sft_final_processed \
    --model_name /mnt/dolphinfs/ssd_pool/docker/user/hadoop-ai-search/chongwenyue/model/Qwen2.5-3B-Instruct \
    --max_length 4096
```

**处理验证集**：
```bash
python scripts/data_process/autosearch_sft.py \
    --input_file data/autosearch_sft_final_val.jsonl \
    --output_dir data/autosearch_sft_final_processed \
    --model_name /mnt/dolphinfs/ssd_pool/docker/user/hadoop-ai-search/chongwenyue/model/Qwen2.5-3B-Instruct \
    --max_length 4096
```

**说明**：
- 脚本会自动识别输出文件名（train.parquet 或 val.parquet）
- `--max_length 4096` 与训练配置保持一致

### 步骤3: 更新训练配置

编辑 `verl/trainer/config/autosearch_sft.yaml`，更新数据路径：

```yaml
data:
  train_files: /mnt/dolphinfs/ssd_pool/docker/user/hadoop-ai-search/chenqing30/cqproject/autosearch/data/autosearch_sft_final_processed/train.parquet
  val_files: /mnt/dolphinfs/ssd_pool/docker/user/hadoop-ai-search/chenqing30/cqproject/autosearch/data/autosearch_sft_final_processed/val.parquet
  # ... 其他配置保持不变
```

**其他配置建议**（无需调整，保持原样即可）：
- `train_batch_size: 64` - 8卡H20，每卡8条，合适
- `micro_batch_size: 64` - 等于train_batch_size，避免FSDP数值问题
- `total_epochs: 3` - 数据量约4400条训练集，3个epoch足够
- `lr: 1e-5` - 标准SFT学习率
- `max_length: 4096` - 与数据处理保持一致

### 步骤4: 启动训练

```bash
bash train_autosearch_sft.sh
```

训练脚本会：
- 使用8卡H20 GPU
- 自动从配置文件读取数据路径和训练参数
- 保存checkpoint到 `./checkpoints/autosearch_sft_outdata/`

## 参数调整建议

### ✅ 无需调整的参数

1. **Batch Size**: `train_batch_size=64`, `micro_batch_size=64`
   - 8卡H20，每卡8条，合理
   - 数据量约4400条，每个epoch约69步

2. **Learning Rate**: `lr=1e-5`
   - 标准SFT学习率，无需调整

3. **Epochs**: `total_epochs=3`
   - 约4400条训练数据，3个epoch足够
   - 总步数约207步（69步/epoch × 3）

4. **Max Length**: `max_length=4096`
   - 与数据处理保持一致

### ⚠️ 可选调整

如果发现训练loss下降缓慢或过拟合，可以：

1. **增加epochs到4-5**（如果数据量少，可以多训几个epoch）
2. **降低learning rate到5e-6**（如果loss不稳定）
3. **增加warmup_steps_ratio到0.15**（如果前期loss波动大）

## 训练监控

训练过程中关注：
1. **Train Loss**: 应该稳定下降（从1.5-2.0降到0.5-0.8）
2. **Val Loss**: 应该跟随train loss下降，但略高于train loss
3. **GPU利用率**: 8卡H20应该都在90%以上

## 预期训练时间

- **每个epoch**: 约30-60分钟（取决于数据加载速度）
- **总训练时间**: 约1.5-3小时（3个epoch）

## 验证训练效果

训练完成后，使用评估脚本验证：

```bash
python scripts/eval_autosearch_sft.py \
    --model_path verl/trainer/checkpoints/autosearch_sft_outdata/global_step_XXX \
    --questions "What is the capital of France?" "Who wrote 1984?"
```
