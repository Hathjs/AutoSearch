#!/bin/bash
# AutoSearch SFT Training Script

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

export WAND_PROJECT='AutoSearch-SFT'
export EXPERIMENT_NAME='autosearch-sft-qwen2.5-3b-v1'

# 使用 torchrun 启动分布式训练
# 如果使用单卡，可以改成: torchrun --nproc_per_node=1 python -m verl.trainer.fsdp_sft_trainer ...
# 如果使用 8 卡，使用: torchrun --nproc_per_node=8 python -m verl.trainer.fsdp_sft_trainer ...

# 注意：需要在 verl/trainer/ 目录下运行，因为 hydra 的 config_path 是相对路径
cd verl/trainer

PYTHONUNBUFFERED=1 torchrun --nproc_per_node=8 \
    --standalone \
    python -m verl.trainer.fsdp_sft_trainer \
    --config-path=config \
    --config-name=autosearch_sft \
    trainer.project_name=$WAND_PROJECT \
    trainer.experiment_name=$EXPERIMENT_NAME \
    2>&1 | tee ../../${EXPERIMENT_NAME}.log
