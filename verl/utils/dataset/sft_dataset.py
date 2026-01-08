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

from omegaconf import ListConfig
import os
from typing import List, Union

import pandas as pd

import torch
import numpy as np
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer

from verl.utils.fs import copy_local_path_from_hdfs
from verl.utils.model import compute_position_id_with_mask
import verl.utils.torch_functional as verl_F


def download_files_distributed(download_fn):
    rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
    if rank == 0:
        download_fn()
        if torch.distributed.is_initialized():
            torch.distributed.barrier()
    else:
        if torch.distributed.is_initialized():
            torch.distributed.barrier()
        # download anyway
        download_fn()


class SFTDataset(Dataset):
    """
    Supervised Fine-Tuning Dataset for training LLMs.
    
    Expects parquet files with columns specified by prompt_key and response_key.
    The dataset concatenates prompt and response, then tokenizes them.
    """

    def __init__(self,
                 parquet_files: Union[str, List[str]],
                 tokenizer: PreTrainedTokenizer,
                 prompt_key='question',
                 prompt_dict_keys=None,
                 response_key='answer',
                 response_dict_keys=None,
                 max_length=1024,
                 truncation='error',
                 cache_dir='~/.cache/verl/sft'):
        if not isinstance(parquet_files, (List, ListConfig)):
            parquet_files = [parquet_files]

        self.parquet_files = parquet_files
        self.cache_dir = os.path.expanduser(cache_dir)
        self.tokenizer = tokenizer

        self.prompt_key = prompt_key
        self.prompt_dict_keys = prompt_dict_keys
        self.response_key = response_key
        self.response_dict_keys = response_dict_keys
        self.max_length = max_length
        self.truncation = truncation

        self._download()
        self._read_files()

    def _download(self):
        def _download_files():
            from verl.utils.fs import copy_local_path_from_hdfs
            os.makedirs(self.cache_dir, exist_ok=True)
            for i, parquet_file in enumerate(self.parquet_files):
                self.parquet_files[i] = copy_local_path_from_hdfs(
                    src=parquet_file, 
                    cache_dir=self.cache_dir
                )

        download_files_distributed(_download_files)

    def _read_files(self):
        """Read parquet files and store prompts and responses."""
        dataframes = []
        for parquet_file in self.parquet_files:
            dataframe = pd.read_parquet(parquet_file)
            dataframes.append(dataframe)
        
        self.dataframe = pd.concat(dataframes, ignore_index=True)
        print(f'Loaded {len(self.dataframe)} examples from {len(self.parquet_files)} parquet file(s)')
        
        # Check required columns exist
        if self.prompt_key not in self.dataframe.columns:
            raise ValueError(f"Column '{self.prompt_key}' not found in parquet file. Available columns: {self.dataframe.columns.tolist()}")
        if self.response_key not in self.dataframe.columns:
            raise ValueError(f"Column '{self.response_key}' not found in parquet file. Available columns: {self.dataframe.columns.tolist()}")

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, item):
        """
        Get a single example and tokenize it.
        
        Returns:
            dict with keys: input_ids, attention_mask, position_ids, loss_mask
        """
        row_dict = self.dataframe.iloc[item].to_dict()
        
        # Extract prompt and response
        prompt = row_dict.pop(self.prompt_key)
        response = row_dict.pop(self.response_key)
        
        # Handle dict-style prompts/responses (if needed)
        if self.prompt_dict_keys is not None:
            if isinstance(prompt, dict):
                prompt = ' '.join([prompt.get(k, '') for k in self.prompt_dict_keys])
        
        if self.response_dict_keys is not None:
            if isinstance(response, dict):
                response = ' '.join([response.get(k, '') for k in self.response_dict_keys])
        
        # Convert to string if needed
        if not isinstance(prompt, str):
            prompt = str(prompt)
        if not isinstance(response, str):
            response = str(response)
        
        # Concatenate prompt and response
        # Format: prompt + response (with EOS token)
        full_text = prompt + response
        
        # Tokenize the full sequence
        input_ids, attention_mask = verl_F.tokenize_and_postprocess_data(
            prompt=full_text,
            tokenizer=self.tokenizer,
            max_length=self.max_length,
            pad_token_id=self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else self.tokenizer.eos_token_id,
            left_pad=False,  # Right pad for causal LM
            truncation=self.truncation
        )
        
        # Compute position IDs
        position_ids = compute_position_id_with_mask(attention_mask)
        
        # Create loss mask: 1 for response tokens (to compute loss), 0 for prompt tokens (masked)
        # We need to tokenize prompt separately to know where response starts
        prompt_ids, prompt_attention_mask = verl_F.tokenize_and_postprocess_data(
            prompt=prompt,
            tokenizer=self.tokenizer,
            max_length=self.max_length,
            pad_token_id=self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else self.tokenizer.eos_token_id,
            left_pad=False,
            truncation='right'  # Allow truncation for prompt to find boundary
        )
        
        # Get actual prompt length (excluding padding)
        prompt_length = prompt_attention_mask[0].sum().item()
        seq_length = input_ids.shape[1]
        
        # Create loss mask: 1 for response tokens (to train), 0 for prompt tokens and padding
        loss_mask = torch.zeros(seq_length, dtype=torch.float32)
        
        # Set mask to 1 for all tokens starting from prompt_length
        # This means we compute loss on the entire response
        if prompt_length < seq_length:
            loss_mask[prompt_length:] = 1.0
        
        # Also mask padding tokens (where attention_mask is 0)
        loss_mask = loss_mask * attention_mask[0].float()
        
        result = {
            'input_ids': input_ids[0],
            'attention_mask': attention_mask[0],
            'position_ids': position_ids[0],
            'loss_mask': loss_mask,
        }
        
        # Add any additional fields from the original row
        result.update(row_dict)
        
        return result
