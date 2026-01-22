# AutoSearch SFT 阶段开发进度

## 当前进度概览

**状态**: SFT训练完成，Recall工具已实现并集成

**时间节点**: 2025年1月

**最新进展**:
- ✅ SFT训练已完成（3个epoch，loss从1.943降到0.225）
- ✅ Recall工具模块已实现并集成到评估和推理流程
- ✅ 双通道架构（recall + search）已完整实现

---

## 已完成工作

### 1. 数据生成与优化

#### 1.1 Hindsight Generation 策略实现
- ✅ 实现了 `rewrite_r1_data.py`，使用 GPT-4 从 (question, golden_answer) 生成训练轨迹
- ✅ 支持 NQ（单跳）和 HotpotQA（多跳）数据重写
- ✅ 实现了 NQ recall 比例控制（`--nq_target_recall_ratio`），提升 recall 使用率

#### 1.2 Prompt 工程优化
- ✅ 设计了详细的系统提示，包含明确的 action 决策逻辑
- ✅ 添加了 Few-shot 示例（单跳 recall、单跳 search、多跳推理）
- ✅ 区分单跳和多跳问题的不同提示策略
- ✅ 强化了 tag 配对规则说明（recall→memory, search→information）

#### 1.3 数据质量保证
- ✅ 实现了严格的验证机制：
  - Answer 匹配（支持 strict/substring 模式）
  - Tag 配对规则检查
  - Reasoning 标签存在性检查
- ✅ 实现了 JSON 解析容错（`safe_json_loads`）
- ✅ 实现了重试机制（每个样本最多重试3次）
- ✅ 实现了中间保存（每100条保存一次）

#### 1.4 数据生成结果
- ✅ 生成了 **4791条** r1_rewritten 数据（GPT-4生成）
- ✅ 合并了旧数据（Type A: 500条, Type C: 199条）
- ✅ **最终数据集**: 5489条
- ✅ 数据统计：
  - Recall 比例: 28.7%（1577条）✅ 相比之前有明显提升
  - Search 比例: 73.7%（4048条）
  - 混合模式: 2.5%（136条）✅ 包含 recall+search 混合推理
  - NQ/HotpotQA 比例: 40.5% / 46.8%（平衡）

#### 1.5 数据管道完善
- ✅ 实现了 `extract_r1_questions.py`（从 Search-R1 提取问题）
- ✅ 实现了 `filter_old_data.py`（过滤 Type B，保留 Type A/C）
- ✅ 实现了 `merge_sft_data.py`（合并数据，支持加权采样）
- ✅ 数据已随机打散，分布均匀

### 2. 训练配置准备

#### 2.1 配置文件
- ✅ `verl/trainer/config/autosearch_sft.yaml` 已配置完成
- ✅ 训练参数：
  - `train_batch_size: 64`（8卡H20，每卡8条）
  - `micro_batch_size: 64`（避免FSDP数值问题）
  - `lr: 1e-5`（标准SFT学习率）
  - `total_epochs: 3`（约4400条训练数据，3个epoch足够）
  - `max_length: 4096`（完全覆盖现有数据，最大token数1034）

#### 2.2 学习率调度
- ✅ 使用 **Cosine Decay with Warmup**
- ✅ Warmup 比例: 10%（前10%步数线性增加到lr，之后cosine衰减到0）

#### 2.3 Checkpoint 策略
- ✅ 每个 epoch 结束后保存一次
- ✅ 所有 epoch 完成后保存最终 checkpoint

### 3. 工程化改进

- ✅ 实现了 API 限流处理（429错误重试）
- ✅ 实现了并发处理（多 worker 并行调用 API）
- ✅ 实现了统计输出（action分布、验证通过率等）
- ✅ 实现了元数据追踪（记录数据来源、模型、desired_action等）

---

## 已完成工作（更新）

### 4. SFT训练执行 ✅

#### 4.1 训练配置
- ✅ 使用8卡H20 GPU进行分布式训练
- ✅ Batch size: 64 (每卡8条)
- ✅ Learning rate: 1e-5
- ✅ Epochs: 3
- ✅ Max length: 4096

#### 4.2 训练过程
- ✅ 数据加载：959条训练样本，240条验证样本
- ✅ 训练步数：42步（14步/epoch × 3）
- ✅ Loss下降趋势：
  - Train loss: 1.943 → 0.225
  - Val loss: 0.500 → 0.327
- ✅ Checkpoint保存：所有epoch完成后保存一次

#### 4.3 训练结果
- ✅ 模型已收敛，loss稳定下降
- ✅ 验证loss跟随训练loss，无过拟合迹象
- ✅ Checkpoint已保存，可用于后续评估和RL训练

### 5. Recall工具实现 ✅

详见"已完成工作"部分的"3. Recall工具实现"。

## 待完成工作

### 1. 模型评估与验证

#### 3.1 核心模块
- ✅ 创建 `search_r1/search/recall_tool.py`
  - `recall_internal_knowledge()`: 单个查询的知识回忆
  - `batch_recall_internal_knowledge()`: 批量查询的知识回忆
  - `extract_recall_queries()`: 从文本中提取recall查询

#### 3.2 集成到评估脚本
- ✅ 修改 `scripts/eval_autosearch_sft.py`
  - 集成recall工具，支持真实的知识回忆
  - 多轮交互支持recall和search动作
  - 检索服务器集成

#### 3.3 集成到RL训练模块
- ✅ 修改 `search_r1/llm_agent/generation.py`
  - `execute_predictions()` 添加 `do_recall` 参数
  - 验证模式：调用recall工具测试
  - 训练模式：使用占位符（模型基于SFT生成）

#### 3.4 集成到推理脚本
- ✅ 修改 `infer.py`
  - 完整的双通道架构支持
  - 支持recall和search混合使用

#### 3.5 设计理念
- **解耦推理和知识回忆**: `<think>` 只推理，`<recall>` 专门回忆
- **参数化记忆**: 激发模型已有知识，而非学习新知识
- **独立工具**: `<recall>` 和 `<search>` 都是工具调用，统一架构

### 4. 模型验证

#### 4.1 基础功能测试
```bash
# 启用recall工具（默认）
python scripts/eval_autosearch_sft.py \
    --model_path verl/trainer/checkpoints/autosearch_sft_outdata/global_step_42 \
    --questions "What is the capital of France?" "Who wrote 1984?"

# 禁用search（仅测试recall）
python scripts/eval_autosearch_sft.py \
    --model_path verl/trainer/checkpoints/autosearch_sft_outdata/global_step_42 \
    --questions "What is the capital of France?" \
    --disable_search
```

#### 4.2 评估指标
- Action 选择正确性（recall vs search）
- 答案准确性（EM score）
- 多跳推理能力（HotpotQA）
- 混合模式使用（recall→search, search→recall）
- Recall工具调用成功率
- 回忆内容准确性

#### 4.3 训练结果
- **训练loss**: 从1.943降到0.225（3个epoch）
- **验证loss**: 从0.500降到0.327
- **训练步数**: 42步（14步/epoch × 3）
- **Checkpoint**: 保存在 `verl/trainer/checkpoints/autosearch_sft_outdata/global_step_42/`

---

## 技术细节

### 数据格式

**输入格式** (JSONL):
```json
{
  "messages": [
    {"role": "user", "content": "What is the capital of France?"},
    {"role": "assistant", "content": "<think>...</think><recall>...</recall><memory>...</memory><answer>Paris</answer>"}
  ],
  "meta": {
    "source": "r1_rewritten",
    "data_source": "nq",
    "model": "gpt-4"
  }
}
```

**输出格式** (Parquet):
- `question`: 用户问题
- `answer`: 助手回复（包含完整XML协议）
- `input_ids`, `labels`, `attention_mask`: Tokenized数据（可选）

### 训练配置要点

1. **Batch Size**: `train_batch_size=64`, `micro_batch_size=64`
   - 8卡H20，每卡8条，合理
   - micro_batch_size等于train_batch_size，避免FSDP数值精度问题

2. **Max Length**: `4096`
   - 数据统计：最大token数1034，平均246，99%分位522
   - 4096完全覆盖，甚至可以考虑降低到2048节省显存

3. **Learning Rate**: `1e-5`
   - 标准SFT学习率
   - 如果loss下降慢，可以尝试`2e-5`

4. **Epochs**: `3`
   - 约4400条训练数据，3个epoch足够
   - 总步数约207步（69步/epoch × 3）

### Recall工具技术细节

1. **核心模块**: `search_r1/search/recall_tool.py`
   - `recall_internal_knowledge()`: 单个查询的知识回忆
   - `batch_recall_internal_knowledge()`: 批量查询的知识回忆
   - 使用专门的prompt引导模型回忆知识
   - 低温度（0.1）确保确定性回忆

2. **集成点**:
   - **评估脚本**: `scripts/eval_autosearch_sft.py` - 支持真实recall工具调用
   - **RL训练**: `search_r1/llm_agent/generation.py` - 验证模式调用工具，训练模式使用占位符
   - **推理脚本**: `infer.py` - 完整的双通道架构支持

3. **设计理念**:
   - **解耦推理和知识回忆**: `<think>` 只推理，`<recall>` 专门回忆
   - **参数化记忆**: 激发模型已有知识，而非学习新知识
   - **独立工具**: `<recall>` 和 `<search>` 都是工具调用，统一架构

### 数据质量亮点

1. **Recall比例提升**: 28.7%（相比之前有明显提升）
2. **混合模式**: 2.5%（136条），包含recall+search混合推理
3. **数据平衡**: NQ和HotpotQA比例接近（40.5% vs 46.8%）
4. **数据来源**: 87.3%为GPT-4生成的新数据，质量更高

---

## 下一步计划

### 短期（1-2周）

1. **模型评估** ✅
   - ✅ 使用评估脚本测试模型
   - ✅ 检查action选择正确性
   - ✅ 检查答案准确性
   - ⏳ 在NQ/HotpotQA测试集上评估
   - ⏳ 分析错误模式和action分布

2. **Recall工具验证**
   - ✅ Recall工具已实现并集成
   - ⏳ 测试recall工具在不同场景下的表现
   - ⏳ 优化recall prompt（如需要）

### 中期（2-4周）

1. **模型评估**
   - 在NQ测试集上评估
   - 在HotpotQA测试集上评估
   - 分析action分布和错误模式

2. **数据优化**（如需要）
   - 根据评估结果调整数据分布
   - 增加特定类型数据（如混合模式）
   - 优化prompt和验证逻辑

3. **准备RL阶段**
   - 设计reward函数
   - 准备RL训练环境
   - 集成检索服务器

### 长期（1-2月）

1. **RL训练**
   - 实现PPO/GRPO训练
   - 优化reward函数
   - 迭代改进模型

2. **系统集成**
   - 集成Google Search API
   - 实现完整的推理流程
   - 端到端测试

---

## 关键文件

### 数据生成脚本
- `scripts/data_process/sft_data_process/extract_r1_questions.py` - 提取问题
- `scripts/data_process/sft_data_process/rewrite_r1_data.py` - 重写数据
- `scripts/data_process/sft_data_process/filter_old_data.py` - 过滤数据
- `scripts/data_process/sft_data_process/merge_sft_data.py` - 合并数据

### 训练相关
- `verl/trainer/config/autosearch_sft.yaml` - 训练配置
- `train_autosearch_sft.sh` - 训练启动脚本
- `verl/trainer/fsdp_sft_trainer.py` - SFT训练器

### 评估相关
- `scripts/eval_autosearch_sft.py` - 模型评估脚本（已集成recall工具）

### Recall工具相关
- `search_r1/search/recall_tool.py` - Recall工具核心模块
- `docs/recall_tool_guide.md` - Recall工具使用指南
- `docs/recall_implementation_summary.md` - Recall工具实现总结

### 数据文件
- `data/autosearch_sft_final.jsonl` - 最终合并数据（5489条）
- `data/autosearch_sft_final_processed/` - Parquet格式数据（待生成）

---

## 注意事项

1. **数据打散**: 数据已随机打散，无需按类型分类训练
2. **Max Length**: 4096足够，但可以考虑降低到2048节省显存
3. **学习率**: 先用1e-5，观察loss曲线再决定是否调整
4. **Checkpoint**: 每个epoch保存已足够，无需每步保存
5. **API限流**: 如果生成更多数据，注意控制并发数避免429错误

---

**最后更新**: 2025年1月

**主要更新**:
- ✅ SFT训练已完成
- ✅ Recall工具已实现并集成
- ✅ 双通道架构（recall + search）已完整实现
- ✅ 评估脚本已支持recall工具调用
