# Recall 工具实现总结

## 已完成的修改

### 1. ✅ 创建 Recall 工具模块

**文件**: `search_r1/search/recall_tool.py`

实现了核心的 recall 工具功能：

- `recall_internal_knowledge()`: 单个查询的知识回忆
- `batch_recall_internal_knowledge()`: 批量查询的知识回忆
- `extract_recall_queries()`: 从文本中提取 recall 查询

**关键特性**：
- 使用专门的 prompt 引导模型回忆知识
- 低温度（0.1）确保确定性回忆
- 自动处理 chat template
- 错误处理和降级策略

### 2. ✅ 修改评估脚本

**文件**: `scripts/eval_autosearch_sft.py`

**修改内容**：
- 导入 `recall_internal_knowledge` 工具
- 在检测到 `<recall>` 动作时，调用 recall 工具
- 将回忆结果添加到 prompt 中，继续推理

**改进**：
- 从简单的占位符改为真实的工具调用
- 支持模型真正回忆知识
- 错误处理和日志记录

### 3. ✅ 修改 RL 训练模块

**文件**: `search_r1/llm_agent/generation.py`

**修改内容**：
- `execute_predictions()` 方法添加 `do_recall` 参数
- 支持批量处理 recall 查询
- 在验证模式下调用 recall 工具
- 在训练模式下使用占位符（模型基于 SFT 训练生成）

**设计决策**：
- **验证模式**：调用 recall 工具，测试模型回忆能力
- **训练模式**：使用占位符，让模型基于 SFT 训练自己生成记忆内容
- 支持不同的模型访问路径（`model`, `base_model`, `_model`）

### 4. ✅ 修改推理脚本

**文件**: `infer.py`

**修改内容**：
- 导入 recall 工具
- 扩展 `get_query()` 支持 recall 和 search
- 更新 stopping criteria 支持 recall 和 answer
- 在生成循环中处理 recall 动作

**改进**：
- 完整的双通道架构支持
- 支持 recall 和 search 混合使用
- 错误处理和降级策略

## 架构设计

### 工作流程

```
用户问题
    ↓
模型推理 (<think>)
    ↓
判断需要知识
    ↓
    ├─→ 常见知识 → <recall>query</recall>
    │                    ↓
    │            recall_internal_knowledge()
    │                    ↓
    │            <memory>content</memory>
    │                    ↓
    │            继续推理 → <answer>
    │
    └─→ 长尾知识 → <search>query</search>
                        ↓
                检索服务器 API
                        ↓
                 <information>results</information>
                        ↓
                 继续推理 → <answer>
```

### 解耦设计

**之前（Search-R1）**：
```
<think>
  推理 + 知识回忆（混合）
</think>
→ 容易编造知识（Sycophancy）
```

**现在（AutoSearch V2）**：
```
<think>
  纯逻辑推理
</think>
<recall>query</recall>
<memory>knowledge</memory>
→ 知识回忆与推理分离
```

## 使用示例

### 评估脚本

```bash
python scripts/eval_autosearch_sft.py \
    --model_path verl/trainer/checkpoints/autosearch_sft_outdata/global_step_42 \
    --questions "What is the capital of France?"
```

### 推理脚本

```bash
# 启动检索服务器（可选）
bash retrieval_launch.sh

# 运行推理
python infer.py
```

## 关键设计决策

### 1. 为什么在训练模式使用占位符？

- 在 RL 训练中，模型应该基于 SFT 训练自己生成记忆内容
- 占位符 `<memory>Retrieving internal knowledge...</memory>` 作为触发信号
- 模型在训练时已经学会了如何生成 `<memory>` 内容

### 2. 为什么在验证模式调用工具？

- 验证模式用于测试模型的实际能力
- 调用 recall 工具可以引导模型回忆，测试是否真正学会了知识
- 帮助诊断模型是否能够正确使用 recall 动作

### 3. 温度设置

- Recall: `temperature=0.1`（低温度，确定性回忆）
- Search: 使用检索服务器（不涉及生成）
- 推理: `temperature=0.7`（标准温度）

## 测试建议

### 1. 单元测试

测试 recall 工具的基本功能：
- 单个查询回忆
- 批量查询回忆
- 错误处理
- 边界情况（空查询、无知识等）

### 2. 集成测试

测试完整流程：
- 评估脚本中的 recall 集成
- RL 训练中的 recall 支持
- 推理脚本中的 recall 支持

### 3. 功能测试

测试模型行为：
- 模型是否能正确使用 `<recall>` 动作
- 回忆的内容是否准确
- 是否能区分何时使用 recall vs search

## 后续改进方向

1. **更智能的 Recall Prompt**：根据查询类型调整 prompt
2. **Recall 质量评估**：评估回忆内容的准确性
3. **Recall 缓存**：缓存常见查询的回忆结果
4. **混合策略**：支持 recall → search 的混合策略
5. **RL 奖励优化**：在奖励函数中更好地利用 recall 和 search 的区别

## 相关文档

- `docs/recall_tool_guide.md`: 详细使用指南
- `docs/eval_guide.md`: 评估脚本使用指南
- `Plan/AutoSearch.md`: 设计规范
