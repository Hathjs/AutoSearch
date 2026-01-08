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
Split JSONL dataset into train and validation sets.

Usage:
    python scripts/data_process/split_data.py \
        --input data/autosearch_sft_raw.jsonl \
        --train_output data/autosearch_sft_train.jsonl \
        --val_output data/autosearch_sft_val.jsonl \
        --val_ratio 0.2 \
        --seed 42
"""

import json
import argparse
import random
from collections import Counter
from pathlib import Path


def split_data(input_file, train_output, val_output, val_ratio=0.2, seed=42, stratify_by_type=False):
    """
    Split JSONL data into train and validation sets.
    
    Args:
        input_file: Input JSONL file path
        train_output: Output path for training data
        val_output: Output path for validation data
        val_ratio: Ratio of validation data (default: 0.2)
        seed: Random seed for reproducibility
        stratify_by_type: Whether to stratify split by type (A/B/C)
    """
    # Set random seed
    random.seed(seed)
    
    # Load data
    print(f"Loading data from {input_file}...")
    data = []
    with open(input_file, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    
    print(f"Total: {len(data)} examples")
    
    # Show type distribution
    if data and 'meta' in data[0]:
        types = [d.get('meta', {}).get('type', 'unknown') for d in data]
        type_counts = Counter(types)
        print(f"Type distribution: {dict(type_counts)}")
    
    # Split data
    if stratify_by_type and data and 'meta' in data[0]:
        # Stratified split by type
        # Group data by type
        type_to_indices = {}
        for i, d in enumerate(data):
            data_type = d.get('meta', {}).get('type', 'unknown')
            if data_type not in type_to_indices:
                type_to_indices[data_type] = []
            type_to_indices[data_type].append(i)
        
        train_indices = []
        val_indices = []
        
        for data_type, indices in type_to_indices.items():
            # Shuffle indices
            shuffled = indices.copy()
            random.shuffle(shuffled)
            
            # Calculate split point
            split_point = int(len(shuffled) * (1 - val_ratio))
            
            train_idx = shuffled[:split_point]
            val_idx = shuffled[split_point:]
            
            train_indices.extend(train_idx)
            val_indices.extend(val_idx)
            print(f"Type {data_type}: {len(train_idx)} train, {len(val_idx)} val")
        
        train_data = [data[i] for i in train_indices]
        val_data = [data[i] for i in val_indices]
        
        # Shuffle both sets
        random.shuffle(train_data)
        random.shuffle(val_data)
    else:
        # Simple random split
        shuffled = data.copy()
        random.shuffle(shuffled)
        
        # Calculate split point
        split_point = int(len(shuffled) * (1 - val_ratio))
        
        train_data = shuffled[:split_point]
        val_data = shuffled[split_point:]
    
    print(f"\nSplit results:")
    print(f"  Train: {len(train_data)} examples ({len(train_data)/len(data)*100:.1f}%)")
    print(f"  Val: {len(val_data)} examples ({len(val_data)/len(data)*100:.1f}%)")
    
    # Show type distribution in train
    if train_data and 'meta' in train_data[0]:
        train_types = [d.get('meta', {}).get('type', 'unknown') for d in train_data]
        print(f"  Train type distribution: {dict(Counter(train_types))}")
    
    # Show type distribution in val
    if val_data and 'meta' in val_data[0]:
        val_types = [d.get('meta', {}).get('type', 'unknown') for d in val_data]
        print(f"  Val type distribution: {dict(Counter(val_types))}")
    
    # Save train data
    train_output_path = Path(train_output)
    train_output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(train_output, 'w', encoding='utf-8') as f:
        for item in train_data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    print(f"\nSaved train data to {train_output}")
    
    # Save val data
    val_output_path = Path(val_output)
    val_output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(val_output, 'w', encoding='utf-8') as f:
        for item in val_data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    print(f"Saved val data to {val_output}")


def main():
    parser = argparse.ArgumentParser(description='Split JSONL dataset into train and validation sets')
    parser.add_argument('--input', type=str, required=True,
                        help='Input JSONL file path')
    parser.add_argument('--train_output', type=str, default='data/autosearch_sft_train.jsonl',
                        help='Output path for training data')
    parser.add_argument('--val_output', type=str, default='data/autosearch_sft_val.jsonl',
                        help='Output path for validation data')
    parser.add_argument('--val_ratio', type=float, default=0.2,
                        help='Ratio of validation data (default: 0.2)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility (default: 42)')
    parser.add_argument('--stratify', action='store_true',
                        help='Stratify split by type (A/B/C) to maintain proportions')
    
    args = parser.parse_args()
    
    # Validate arguments
    if not Path(args.input).exists():
        raise FileNotFoundError(f"Input file not found: {args.input}")
    
    if args.val_ratio <= 0 or args.val_ratio >= 1:
        raise ValueError(f"val_ratio must be between 0 and 1, got {args.val_ratio}")
    
    split_data(
        input_file=args.input,
        train_output=args.train_output,
        val_output=args.val_output,
        val_ratio=args.val_ratio,
        seed=args.seed,
        stratify_by_type=args.stratify
    )


if __name__ == '__main__':
    main()
