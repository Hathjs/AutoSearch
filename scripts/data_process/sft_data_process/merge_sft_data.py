#!/usr/bin/env python3
"""
Merge multiple SFT data files into a single dataset.
This script combines:
- Rewritten NQ/HotpotQA data (from rewrite_r1_data.py)
- Filtered old Type A data (Internal Recall)
- Filtered old Type C data (Trap/Hallucination)
"""

import json
import argparse
import random
from pathlib import Path
from collections import Counter
from typing import List, Dict


def load_jsonl(file_path: str) -> List[Dict]:
    """Load samples from JSONL file"""
    samples = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                try:
                    samples.append(json.loads(line))
                except json.JSONDecodeError as e:
                    print(f"[Warning] Failed to parse line in {file_path}: {e}")
                    continue
    return samples


def merge_data(
    input_files: List[str],
    output_file: str,
    shuffle: bool = True,
    seed: int = 42
):
    """
    Merge multiple SFT data files.
    
    Args:
        input_files: List of input JSONL file paths (can include weights like "file1.jsonl:2" for 2x sampling)
        output_file: Output JSONL file path
        shuffle: Whether to shuffle the merged data
        seed: Random seed for shuffling
    """
    print(f"Merging {len(input_files)} data files...")
    
    all_samples = []
    
    for file_path in input_files:
        # Check if weight is specified (format: "file.jsonl:weight")
        if ':' in file_path:
            file_path, weight_str = file_path.rsplit(':', 1)
            try:
                weight = int(weight_str)
            except ValueError:
                weight = 1
        else:
            weight = 1
        
        if not Path(file_path).exists():
            print(f"[Warning] File not found: {file_path}, skipping...")
            continue
        
        samples = load_jsonl(file_path)
        print(f"  Loaded {len(samples)} samples from {file_path} (weight: {weight}x)")
        
        # Add samples with weight
        for _ in range(weight):
            all_samples.extend(samples.copy())
    
    print(f"\nTotal samples before merging: {len(all_samples)}")
    
    # Shuffle if requested
    if shuffle:
        random.seed(seed)
        random.shuffle(all_samples)
        print(f"Shuffled with seed {seed}")
    
    # Save merged data
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        for sample in all_samples:
            f.write(json.dumps(sample, ensure_ascii=False) + '\n')
    
    # Print statistics
    print(f"\n{'='*60}")
    print(f"Merge Complete!")
    print(f"{'='*60}")
    print(f"Total samples: {len(all_samples)}")
    
    # Analyze type distribution
    type_counts = Counter()
    source_counts = Counter()
    
    for sample in all_samples:
        sample_type = sample.get('meta', {}).get('type', 'unknown')
        source = sample.get('meta', {}).get('source', 'unknown')
        type_counts[sample_type] += 1
        source_counts[source] += 1
    
    if type_counts:
        print(f"\nType distribution:")
        for sample_type, count in type_counts.most_common():
            percentage = count / len(all_samples) * 100
            print(f"  Type {sample_type}: {count} ({percentage:.1f}%)")
    
    if source_counts:
        print(f"\nSource distribution:")
        for source, count in source_counts.most_common():
            percentage = count / len(all_samples) * 100
            print(f"  {source}: {count} ({percentage:.1f}%)")
    
    # Analyze action distribution (for rewritten data)
    recall_count = 0
    search_count = 0
    for sample in all_samples:
        messages = sample.get('messages', [])
        if len(messages) > 1:
            content = messages[1].get('content', '')
            if '<recall>' in content:
                recall_count += 1
            if '<search>' in content:
                search_count += 1
    
    if recall_count > 0 or search_count > 0:
        total_actions = recall_count + search_count
        print(f"\nAction distribution:")
        print(f"  <recall>: {recall_count} ({recall_count/total_actions*100:.1f}%)")
        print(f"  <search>: {search_count} ({search_count/total_actions*100:.1f}%)")
    
    print(f"\nSaved to: {output_file}")
    
    return all_samples


def main():
    parser = argparse.ArgumentParser(description='Merge multiple SFT data files')
    parser.add_argument('--inputs', type=str, nargs='+', required=True,
                        help='Input JSONL files (can specify weight like "file.jsonl:2" for 2x sampling)')
    parser.add_argument('--output', type=str, required=True,
                        help='Output JSONL file path')
    parser.add_argument('--shuffle', action='store_true', default=True,
                        help='Shuffle the merged data (default: True)')
    parser.add_argument('--no_shuffle', dest='shuffle', action='store_false',
                        help='Do not shuffle the merged data')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for shuffling (default: 42)')
    
    args = parser.parse_args()
    
    merge_data(args.inputs, args.output, args.shuffle, args.seed)


if __name__ == '__main__':
    main()
