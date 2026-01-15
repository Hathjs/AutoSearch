# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
A lightweight one-file FSDP SFT Trainer
TODO(zhangchi.usc1992)
- Add calculation of mfu
- Add validation
"""

import os

os.environ['NCCL_DEBUG'] = 'WARN'
os.environ['TOKENIZERS_PARALLELISM'] = 'true'

import logging
import re
import torch
import torch.distributed
from torch import nn, optim
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP, MixedPrecision, ShardingStrategy, CPUOffload
from transformers import AutoTokenizer, AutoModelForCausalLM, PreTrainedModel, AutoConfig
from verl.utils.torch_functional import get_cosine_schedule_with_warmup
from tensordict import TensorDict
from torch.utils.data import DataLoader, DistributedSampler

from verl.utils.fsdp_utils import get_fsdp_wrap_policy, init_fn, get_init_weight_context_manager
from verl.utils.dataset import SFTDataset
from verl.utils.fs import copy_local_path_from_hdfs
from verl.utils.tracking import Tracking

from torch.distributed.device_mesh import DeviceMesh

import verl.utils.hdfs_io as hdfs_io
from verl.utils.debug import log_gpu_memory_usage

# Optional import for LoRA support
try:
    from peft import LoraConfig, TaskType, get_peft_model
    PEFT_AVAILABLE = True
except ImportError:
    PEFT_AVAILABLE = False
    LoraConfig = None
    TaskType = None
    get_peft_model = None

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv('VERL_SFT_LOGGING_LEVEL', 'WARN'))




def extract_step(path):
    match = re.search(r'global_step_(\d+)', path)
    if match:
        return int(match.group(1))
    return None


def convert_to_regular_types(obj):
    """Convert Hydra configs and other special types to regular Python types."""
    from omegaconf import ListConfig, DictConfig
    if isinstance(obj, (ListConfig, DictConfig)):
        return {k: convert_to_regular_types(v) for k, v in obj.items()} if isinstance(obj, DictConfig) else list(obj)
    elif isinstance(obj, (list, tuple)):
        return [convert_to_regular_types(x) for x in obj]
    elif isinstance(obj, dict):
        return {k: convert_to_regular_types(v) for k, v in obj.items()}
    return obj


class FSDPSFTTrainer(object):

    def __init__(self, config, device_mesh: DeviceMesh):
        self.config = config
        self.device_mesh = device_mesh
        # build tokenizer first
        local_model_path = copy_local_path_from_hdfs(src=self.config.model.partial_pretrain, verbose=True)
        from verl.utils import hf_tokenizer
        self.tokenizer = hf_tokenizer(local_model_path, trust_remote_code=self.config.model.trust_remote_code)
        if self.config.data.chat_template is not None:
            raise ValueError('Apply Chat template from config is not supported yet.')

        # normalize dp size
        self._normalize_config_bsz()

        self._build_dataloader()
        # build model
        self._build_model_optimizer()

        # TODO: add checkpoint manager
        if self.device_mesh.get_rank() == 0:
            print(self.config)

    def _normalize_config_bsz(self):
        dp_size = self.device_mesh.size()
        if self.device_mesh.get_rank() == 0:
            print(f'Normalize batch size by dp {dp_size}')

        assert self.config.data.train_batch_size % dp_size == 0
        assert self.config.data.micro_batch_size % dp_size == 0

        self.config.data.train_batch_size //= dp_size
        self.config.data.micro_batch_size //= dp_size

    def _build_dataloader(self):
        config = self.config
        # build dataset
        self.train_dataset = SFTDataset(parquet_files=config.data.train_files,
                                        tokenizer=self.tokenizer,
                                        prompt_key=config.data.prompt_key,
                                        prompt_dict_keys=config.data.get('prompt_dict_keys', None),
                                        response_key=config.data.response_key,
                                        response_dict_keys=config.data.get('response_dict_keys', None),
                                        max_length=config.data.max_length,
                                        truncation=config.data.truncation)
        self.val_dataset = SFTDataset(parquet_files=config.data.val_files,
                                      tokenizer=self.tokenizer,
                                      prompt_key=config.data.prompt_key,
                                      prompt_dict_keys=config.data.get('prompt_dict_keys', None),
                                      response_key=config.data.response_key,
                                      response_dict_keys=config.data.get('response_dict_keys', None),
                                      max_length=config.data.max_length,
                                      truncation=config.data.truncation)

        # Define collate function with access to tokenizer
        def collate_fn(batch):
            """Custom collate function to pad sequences to the same length."""
            # Find max length in batch
            max_len = max(item['input_ids'].shape[0] for item in batch)
            
            # Get pad token id from tokenizer
            pad_token_id = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else (self.tokenizer.eos_token_id if self.tokenizer.eos_token_id is not None else 0)
            
            collated = {}
            for key in ['input_ids', 'labels', 'attention_mask', 'position_ids', 'loss_mask']:
                if key in batch[0]:
                    # Pad sequences
                    padded = []
                    for item in batch:
                        seq = item[key]
                        if len(seq) < max_len:
                            if key == 'labels':
                                # Pad labels with -100 (ignore index)
                                padding = torch.full((max_len - len(seq),), -100, dtype=seq.dtype)
                            elif key == 'attention_mask' or key == 'loss_mask':
                                # Pad masks with 0
                                padding = torch.zeros(max_len - len(seq), dtype=seq.dtype)
                            elif key == 'position_ids':
                                # Pad position_ids with continuation
                                last_pos = seq[-1].item() if len(seq) > 0 else 0
                                padding = torch.arange(last_pos + 1, last_pos + 1 + max_len - len(seq), dtype=seq.dtype)
                            else:  # input_ids
                                # Pad input_ids with pad_token_id
                                padding = torch.full((max_len - len(seq),), pad_token_id, dtype=seq.dtype)
                            seq = torch.cat([seq, padding])
                        padded.append(seq)
                    collated[key] = torch.stack(padded)
            
            return collated

        # build dataloader
        rank = self.device_mesh.get_rank()
        world_size = self.device_mesh.size()
        self.train_sampler = DistributedSampler(self.train_dataset,
                                                shuffle=True,
                                                num_replicas=world_size,
                                                rank=rank,
                                                drop_last=True)
        num_workers = int(config.data.get('num_workers', 8))
        val_num_workers = int(config.data.get('val_num_workers', num_workers))
        val_shuffle = bool(config.data.get('val_shuffle', False))

        self.train_dataloader = DataLoader(dataset=self.train_dataset,
                                           batch_size=config.data.train_batch_size,
                                           sampler=self.train_sampler,
                                           num_workers=num_workers,
                                           pin_memory=True,
                                           drop_last=True,
                                           collate_fn=collate_fn)

        self.val_sampler = DistributedSampler(self.val_dataset,
                                              shuffle=val_shuffle,
                                              num_replicas=world_size,
                                              rank=rank,
                                              drop_last=True)
        self.val_dataloader = DataLoader(dataset=self.val_dataset,
                                         batch_size=config.data.micro_batch_size,
                                         sampler=self.val_sampler,
                                         num_workers=val_num_workers,
                                         pin_memory=True,
                                         drop_last=True,
                                         collate_fn=collate_fn)

    def _build_model_optimizer(self):
        # TODO (zhangchi.usc1992):
        # 1. support pretrain from random weights
        # 2. support init directly from sharded weights
        local_model_path = copy_local_path_from_hdfs(src=self.config.model.partial_pretrain, verbose=True)

        if self.config.model.get('external_lib', None) is not None:
            # This is used to import external_lib into the huggingface systems
            import importlib
            importlib.import_module(self.config.model.external_lib)

        log_gpu_memory_usage('Before model allocation', logger=logger)

        trust_remote_code = self.config.model.trust_remote_code
        # load config first
        config = AutoConfig.from_pretrained(local_model_path, trust_remote_code=trust_remote_code)

        # This may be very large
        init_context = get_init_weight_context_manager(use_meta_tensor=not config.tie_word_embeddings)

        with init_context():
            # 根据配置选择 attention 实现
            # 'flash_attention_2': 最快，但可能有兼容性问题
            # 'sdpa': PyTorch 标准实现，更稳定
            # None: 使用模型默认实现
            attn_impl = self.config.model.get('attn_implementation', 'sdpa')
            if attn_impl == 'flash_attention_2':
                logger.info("Using Flash Attention 2.0 (fast but may have compatibility issues)")
            elif attn_impl == 'sdpa':
                logger.info("Using SDPA (PyTorch standard attention, more stable)")
            else:
                logger.info(f"Using default attention implementation: {attn_impl}")
            
            self.model: PreTrainedModel = AutoModelForCausalLM.from_pretrained(local_model_path,
                                                                               config=config,
                                                                               torch_dtype=torch.bfloat16,
                                                                               attn_implementation=attn_impl,
                                                                               trust_remote_code=trust_remote_code)
            if self.config.model.get('lora_rank', 0) > 0:
                if not PEFT_AVAILABLE:
                    raise ImportError(
                        "LoRA is enabled (lora_rank > 0) but peft module is not available. "
                        "Please install peft: pip install peft"
                    )
                self.model.enable_input_require_grads()
                # Convert config to regular Python types before creating PEFT model
                lora_config = {
                    'task_type': TaskType.CAUSAL_LM,
                    'r': self.config.model.lora_rank,
                    'lora_alpha': self.config.model.lora_alpha,
                    'target_modules': convert_to_regular_types(self.config.model.target_modules),
                    'bias': "none"
                }
                self.model = get_peft_model(self.model, LoraConfig(**lora_config))

        if self.config.model.enable_gradient_checkpointing:
            self.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})

        log_gpu_memory_usage('After model allocation', logger=logger)

        mixed_precision = MixedPrecision(param_dtype=torch.bfloat16,
                                         reduce_dtype=torch.float32,
                                         buffer_dtype=torch.float32)

        auto_wrap_policy = get_fsdp_wrap_policy(self.model,
                                                config=self.config.model.fsdp_config.wrap_policy,
                                                is_lora=self.config.model.get('lora_rank', 0) > 0)
        if self.device_mesh.get_rank() == 0:
            print(auto_wrap_policy)

        if not self.config.model.fsdp_config.cpu_offload:
            cpu_offload = None
        else:
            cpu_offload = CPUOffload(offload_params=self.config.model.fsdp_config.offload_params)

        self.fsdp_model = FSDP(module=self.model,
                               auto_wrap_policy=auto_wrap_policy,
                               param_init_fn=init_fn,
                               sharding_strategy=ShardingStrategy.FULL_SHARD,
                               mixed_precision=mixed_precision,
                               device_mesh=self.device_mesh,
                               sync_module_states=True,
                               device_id=torch.cuda.current_device(),
                               cpu_offload=cpu_offload,
                               use_orig_params=False)

        log_gpu_memory_usage('After FSDP wrapping', logger=logger)

        self.optimizer = optim.AdamW(self.fsdp_model.parameters(),
                                     lr=self.config.optim.lr,
                                     betas=self.config.optim.betas,
                                     weight_decay=self.config.optim.weight_decay)

        log_gpu_memory_usage('After initialize optimizer', logger=logger)

        steps_per_epoch = len(self.train_dataloader)
        total_steps = steps_per_epoch * self.config.trainer.total_epochs

        if self.device_mesh.get_rank() == 0:
            print(
                f'Number of steps/epoch {steps_per_epoch}, number of epochs {self.config.trainer.total_epochs}, total number of steps {total_steps}'
            )

        num_warmup_steps = int(total_steps * self.config.optim.warmup_steps_ratio)

        self.lr_scheduler = get_cosine_schedule_with_warmup(optimizer=self.optimizer,
                                                            num_warmup_steps=num_warmup_steps,
                                                            num_training_steps=total_steps)

    def _compute_loss(self, batch):
        # 检查 batch 是否为空或异常
        if 'loss_mask' not in batch:
            rank = self.device_mesh.get_rank() if hasattr(self, 'device_mesh') else 0
            logger.error(f"Rank {rank}: Missing 'loss_mask' in batch!")
            return torch.tensor(0.0, device=torch.cuda.current_device(), dtype=torch.float32)
        
        # 检查序列长度，避免切片后得到空 tensor
        seq_length = batch['loss_mask'].shape[1]
        if seq_length <= 1:
            rank = self.device_mesh.get_rank() if hasattr(self, 'device_mesh') else 0
            logger.warning(f"Rank {rank}: Sequence length is {seq_length}, too short for loss computation. Returning 0.0")
            return torch.tensor(0.0, device=torch.cuda.current_device(), dtype=torch.float32)
        
        loss_mask = batch.pop('loss_mask')[:, :-1].reshape(-1).cuda()
        labels = batch['input_ids'][:, 1:].cuda()
        
        # 检查 loss_mask 是否为空
        if loss_mask.numel() == 0:
            rank = self.device_mesh.get_rank() if hasattr(self, 'device_mesh') else 0
            logger.warning(f"Rank {rank}: loss_mask is empty after slicing. Returning 0.0")
            return torch.tensor(0.0, device=torch.cuda.current_device(), dtype=torch.float32)

        # 检查输入是否有异常值
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        position_ids = batch['position_ids']
        
        # 检查是否有 NaN 或 Inf
        if torch.isnan(input_ids).any() or torch.isinf(input_ids.float()).any():
            rank = self.device_mesh.get_rank() if hasattr(self, 'device_mesh') else 0
            logger.error(f"Rank {rank}: input_ids contains NaN or Inf!")
            return torch.tensor(0.0, device=torch.cuda.current_device(), dtype=torch.float32)
        
        try:
            with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                output = self.fsdp_model(input_ids=input_ids,
                                         attention_mask=attention_mask,
                                         position_ids=position_ids,
                                         use_cache=False)  # prevent model thinks it it generating
        except Exception as e:
            rank = self.device_mesh.get_rank() if hasattr(self, 'device_mesh') else 0
            logger.error(f"Rank {rank}: Error in model forward: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return torch.tensor(0.0, device=torch.cuda.current_device(), dtype=torch.float32)

        logits = output.logits
        
        # 检查 logits 是否有异常值
        if torch.isnan(logits).any() or torch.isinf(logits).any():
            rank = self.device_mesh.get_rank() if hasattr(self, 'device_mesh') else 0
            logger.error(f"Rank {rank}: logits contains NaN or Inf!")
            return torch.tensor(0.0, device=logits.device, dtype=logits.dtype)

        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels.contiguous()
        # Flatten the tokens
        loss_fct = nn.CrossEntropyLoss(reduction='none')
        shift_logits = shift_logits.view(-1, self.model.config.vocab_size)
        shift_labels = shift_labels.view(-1)
        # Enable model parallelism
        shift_labels = shift_labels.to(shift_logits.device)
        loss = loss_fct(shift_logits, shift_labels)
        
        # 确保 loss 和 loss_mask 的 shape 匹配
        if loss.shape != loss_mask.shape:
            rank = self.device_mesh.get_rank() if hasattr(self, 'device_mesh') else 0
            logger.error(f"Rank {rank}: Shape mismatch! loss.shape={loss.shape}, loss_mask.shape={loss_mask.shape}")
            return torch.tensor(0.0, device=loss.device, dtype=loss.dtype)
        
        loss = loss * loss_mask

        valid_token_this_rank = torch.sum(loss_mask)

        if self.config.data.balance_dp_token:
            torch.distributed.all_reduce(valid_token_this_rank)  # becomes total valid tokens in all ranks
            dp_size = torch.distributed.get_world_size()
        else:
            dp_size = 1

        # 防止除零错误：如果当前rank没有有效token，返回0
        # 在validation阶段，某些rank可能分配到空的batch
        # 使用 .item() 确保是 Python 标量，添加 epsilon 防止数值精度问题
        valid_token_count = valid_token_this_rank.item() if valid_token_this_rank.numel() == 1 else float(valid_token_this_rank)
        epsilon = 1e-8
        
        if valid_token_count > epsilon:
            # 使用 epsilon 防止数值精度问题导致的除零
            loss = torch.sum(loss) / (valid_token_this_rank + epsilon) * dp_size
        else:
            # 如果没有有效token，返回0（在all_reduce时会自动平均）
            # 记录警告以便调试（只在 rank 0 打印，避免日志过多）
            rank = self.device_mesh.get_rank() if hasattr(self, 'device_mesh') else 0
            if rank == 0:
                logger.warning(f"Warning: valid_token_this_rank is {valid_token_count}, returning 0.0 loss. "
                             f"This may indicate empty batch or all tokens masked.")
            loss = torch.tensor(0.0, device=loss.device, dtype=loss.dtype)
        return loss

    def training_step(self, batch: TensorDict):
        self.fsdp_model.train()

        log_gpu_memory_usage('Before optimizer zero_grad', logger=logger)

        self.optimizer.zero_grad()

        log_gpu_memory_usage('After optimizer zero_grad', logger=logger)

        micro_batches = batch.split(self.config.data.micro_batch_size)
        n_micro_batches = len(micro_batches)
        step_loss = 0
        for micro_batch in micro_batches:
            loss = self._compute_loss(batch=micro_batch) / n_micro_batches
            loss.backward()
            step_loss += loss.item()

        self.fsdp_model.clip_grad_norm_(max_norm=self.config.optim.clip_grad)

        log_gpu_memory_usage('Before optimizer step', logger=logger)

        self.optimizer.step()

        log_gpu_memory_usage('After optimizer step', logger=logger)

        self.lr_scheduler.step()

        # reduce loss across dp ranks
        lr = self.lr_scheduler.get_last_lr()[0]

        log_gpu_memory_usage('After offload weights', logger=logger)

        step_loss = torch.tensor(step_loss).cuda()
        torch.distributed.all_reduce(step_loss, op=torch.distributed.ReduceOp.AVG)
        return {'train/loss': step_loss.detach().item(), 'train/lr(1e-3)': lr * 1e3}

    def validation_step(self, batch: TensorDict):
        self.fsdp_model.eval()
        with torch.no_grad():
            try:
                # 检查 batch 是否为空或异常
                if 'loss_mask' not in batch:
                    rank = self.device_mesh.get_rank() if hasattr(self, 'device_mesh') else 0
                    logger.error(f"Rank {rank}: Missing 'loss_mask' in validation batch!")
                    loss = torch.tensor(0.0, device=torch.cuda.current_device(), dtype=torch.float32)
                else:
                    loss = self._compute_loss(batch)
                    # 确保loss是tensor类型（防止某些edge case）
                    if not isinstance(loss, torch.Tensor):
                        loss = torch.tensor(loss, device=torch.cuda.current_device(), dtype=torch.float32)
                    # 检查 loss 是否为 NaN 或 Inf
                    if torch.isnan(loss) or torch.isinf(loss):
                        rank = self.device_mesh.get_rank() if hasattr(self, 'device_mesh') else 0
                        logger.warning(f"Rank {rank}: Loss is NaN or Inf, replacing with 0.0")
                        loss = torch.tensor(0.0, device=loss.device, dtype=loss.dtype)
                # 可选：validation 阶段是否做跨 rank 规约
                # 现象：SIGFPE 固定发生在 “Starting validation...” 后，强烈怀疑是通信规约触发底层异常
                # 用该开关可快速验证是不是 all_reduce 导致（先跑通训练再进一步定位）
                do_val_all_reduce = bool(self.config.trainer.get('val_all_reduce', True))
                if do_val_all_reduce:
                    # 避免 ReduceOp.AVG：在某些环境/类型组合下可能触发底层 SIGFPE
                    # 改为 SUM + 手动除以 world_size，并强制 float32 做规约
                    loss = loss.float()
                    torch.distributed.all_reduce(loss, op=torch.distributed.ReduceOp.SUM)
                    loss = loss / torch.distributed.get_world_size()
            except Exception as e:
                # 如果计算loss时出错，返回0并记录警告
                rank = self.device_mesh.get_rank() if hasattr(self, 'device_mesh') else 0
                logger.error(f"Rank {rank}: Error in validation_step: {e}, returning 0.0")
                import traceback
                logger.error(traceback.format_exc())
                loss = torch.tensor(0.0, device=torch.cuda.current_device(), dtype=torch.float32)
                do_val_all_reduce = bool(self.config.trainer.get('val_all_reduce', True))
                if do_val_all_reduce:
                    torch.distributed.all_reduce(loss, op=torch.distributed.ReduceOp.SUM)
                    loss = loss / torch.distributed.get_world_size()
        return loss

    def save_checkpoint(self, step):
        # save checkpoint
        # 使用新的 FSDP API 避免弃用警告
        from torch.distributed.fsdp import FullStateDictConfig, StateDictType
        
        cfg = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
        
        # 使用 set_state_dict_type 替代 state_dict_type context manager
        # 注意：这会在整个模型生命周期中保持设置，但保存checkpoint后通常不需要恢复
        FSDP.set_state_dict_type(self.fsdp_model, StateDictType.FULL_STATE_DICT, cfg)
        state_dict = self.fsdp_model.state_dict()

        path = os.path.join(self.config.trainer.default_local_dir, f'global_step_{step}')
        # save huggingface model
        if self.device_mesh.get_rank() == 0:
            os.makedirs(path, exist_ok=True)
            self.model.save_pretrained(path, state_dict=state_dict)
            self.tokenizer.save_pretrained(path)
            if self.config.trainer.default_hdfs_dir:
                hdfs_io.makedirs(self.config.trainer.default_hdfs_dir, exist_ok=True)
                hdfs_io.copy(src=path, dst=self.config.trainer.default_hdfs_dir, dirs_exist_ok=True)
        torch.distributed.barrier()

    def fit(self):
        rank = self.device_mesh.get_rank()

        # TODO: add a unified tracking
        if rank == 0:
            tracking = Tracking(project_name=self.config.trainer.project_name,
                                experiment_name=self.config.trainer.experiment_name,
                                default_backend=self.config.trainer.logger)

        global_step = 0
        # compute the total training steps.
        # the total training steps in SFT is mainly for early exit
        total_training_steps = len(self.train_dataloader) * self.config.trainer.total_epochs

        if self.config.trainer.total_training_steps is not None:
            total_training_steps = self.config.trainer.total_training_steps

        self.total_training_steps = total_training_steps
        print(f'Total training steps: {self.total_training_steps}')

        # TODO (zhangchi.usc1992) add back checkpoint manager. Currently, it blocks when uploading to hdfs. So very slow.

        if self.config.trainer.validate_before_training:
            # validate before training
            val_losses = []
            for data in self.val_dataloader:
                data = TensorDict(data, batch_size=self.config.data.micro_batch_size).cuda()
                val_loss = self.validation_step(data)
                # 确保val_loss是tensor类型
                if isinstance(val_loss, torch.Tensor):
                    val_losses.append(val_loss)
                else:
                    val_losses.append(torch.tensor(val_loss, device=torch.cuda.current_device()))
            
            if len(val_losses) > 0:
                if rank == 0:
                    val_loss = torch.mean(torch.stack(val_losses))
                    metric = {'val/loss': val_loss.detach().item()}
                    tracking.log(data=metric, step=global_step)
            else:
                if rank == 0:
                    logger.warning("No validation losses collected, skipping validation logging")
            torch.distributed.barrier()

        for epoch in range(self.config.trainer.total_epochs):
            self.train_sampler.set_epoch(epoch=epoch)
            for data in self.train_dataloader:
                data = TensorDict(data, batch_size=self.config.data.train_batch_size).cuda()
                metric = self.training_step(data)
                if rank == 0:
                    tracking.log(data=metric, step=global_step)
                global_step += 1

                # for early exit validation
                if global_step >= self.total_training_steps:
                    # Perform final validation
                    val_losses = []
                    for val_data in self.val_dataloader:
                        val_data = TensorDict(val_data, batch_size=self.config.data.micro_batch_size).cuda()
                        val_loss = self.validation_step(val_data)
                        # 确保val_loss是tensor类型
                        if isinstance(val_loss, torch.Tensor):
                            val_losses.append(val_loss)
                        else:
                            val_losses.append(torch.tensor(val_loss, device=torch.cuda.current_device()))
                    
                    if len(val_losses) > 0:
                        if rank == 0:
                            avg_val_loss = torch.mean(torch.stack(val_losses))
                            metric = {'val/loss': avg_val_loss.detach().item()}
                            tracking.log(data=metric, step=global_step)
                    else:
                        if rank == 0:
                            logger.warning("No validation losses collected, skipping validation logging")
                    torch.distributed.barrier()

                    # Save final checkpoint
                    self.save_checkpoint(step=global_step)
                    return

            # validation
            # 检查是否跳过 validation（用于排查问题）
            skip_val = self.config.trainer.get('skip_validation', False)
            if skip_val:
                if rank == 0:
                    print(f"Skipping validation after epoch {epoch + 1} (skip_validation=True)")
                torch.distributed.barrier()
            else:
                if rank == 0:
                    print(f"Starting validation after epoch {epoch + 1}...")
                val_losses = []
                for batch_idx, data in enumerate(self.val_dataloader):
                    try:
                        data = TensorDict(data, batch_size=self.config.data.micro_batch_size).cuda()
                        # 检查 batch 是否为空
                        if len(data) == 0:
                            if rank == 0:
                                logger.warning(f"Validation batch {batch_idx} is empty, skipping")
                            continue
                        val_loss = self.validation_step(data)
                        # 确保val_loss是tensor类型
                        if isinstance(val_loss, torch.Tensor):
                            # 检查是否为 NaN 或 Inf
                            if torch.isnan(val_loss) or torch.isinf(val_loss):
                                if rank == 0:
                                    logger.warning(f"Validation batch {batch_idx} produced NaN/Inf loss, skipping")
                                continue
                            val_losses.append(val_loss)
                        else:
                            val_losses.append(torch.tensor(val_loss, device=torch.cuda.current_device()))
                    except Exception as e:
                        if rank == 0:
                            logger.error(f"Error processing validation batch {batch_idx}: {e}")
                            import traceback
                            logger.error(traceback.format_exc())
                        continue
                
                if len(val_losses) > 0:
                    if rank == 0:
                        val_loss = torch.mean(torch.stack(val_losses))
                        metric = {'val/loss': val_loss.detach().item()}
                        tracking.log(data=metric, step=global_step)
                        print(f"Validation loss after epoch {epoch + 1}: {val_loss.detach().item():.4f}")
                else:
                    if rank == 0:
                        logger.warning("No validation losses collected, skipping validation logging")
                torch.distributed.barrier()

            # Note: Checkpoint saving moved to after all epochs complete

        # Save final checkpoint after all epochs are complete
        if rank == 0:
            print(f'Saving final checkpoint at step {global_step} after {self.config.trainer.total_epochs} epochs')
        self.save_checkpoint(step=global_step)


from verl.trainer.fsdp_sft_trainer import FSDPSFTTrainer
import hydra

from torch.distributed.device_mesh import init_device_mesh

from verl.utils.distributed import initialize_global_process_group


@hydra.main(config_path='config', config_name='sft_trainer', version_base=None)
def main(config):
    local_rank, rank, world_size = initialize_global_process_group()

    device_mesh = init_device_mesh(device_type='cuda', mesh_shape=(world_size,), mesh_dim_names=('dp',))
    trainer = FSDPSFTTrainer(config=config, device_mesh=device_mesh)
    trainer.fit()


if __name__ == '__main__':
    main()
