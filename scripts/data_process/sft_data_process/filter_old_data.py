#!/usr/bin/env python3
"""
Filter old SFT data, keeping only Type A (Internal Recall) and Type C (Trap/Hallucination).
Discard Type B (Real-time External Search) to avoid time hallucination issues.
"""

import json
import argparse
from pathlib import Path
from collections import Counter


def filter_old_data(input_file: str, output_file: str, keep_types: list = ['A', 'C']):
    """
    Filter old SFT data, keeping only specified types.
    
    Args:
        input_file: Input JSONL file with old SFT data
        output_file: Output JSONL file path
        keep_types: List of types to keep (default: ['A', 'C'])
    """
    print(f"Filtering data from {input_file}...")
    print(f"Keeping types: {keep_types}")
    
    filtered = []
    discarded = []
    type_counts = Counter()
    
    with open(input_file, 'r', encoding='utf-8') as f:
        for idx, line in enumerate(f):
            if line.strip():
                try:
                    sample = json.loads(line)
                    sample_type = sample.get('meta', {}).get('type', '')
                    
                    type_counts[sample_type] += 1
                    
                    if sample_type in keep_types:
                        filtered.append(sample)
                    else:
                        discarded.append({
                            'index': idx,
                            'type': sample_type,
                            'question': sample.get('messages', [{}])[0].get('content', '')[:100]
                        })
                except json.JSONDecodeError as e:
                    print(f"[Warning] Failed to parse line {idx}: {e}")
                    continue
    
    # Save filtered data
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        for sample in filtered:
            f.write(json.dumps(sample, ensure_ascii=False) + '\n')
    
    # Print statistics
    print(f"\n{'='*60}")
    print(f"Filtering Complete!")
    print(f"{'='*60}")
    print(f"Total samples: {sum(type_counts.values())}")
    print(f"Type distribution (before filtering):")
    for sample_type, count in type_counts.items():
        percentage = count / sum(type_counts.values()) * 100
        print(f"  Type {sample_type}: {count} ({percentage:.1f}%)")
    
    print(f"\nFiltered results:")
    print(f"  Kept: {len(filtered)} samples")
    print(f"  Discarded: {len(discarded)} samples")
    
    if filtered:
        filtered_type_counts = Counter()
        for sample in filtered:
            sample_type = sample.get('meta', {}).get('type', '')
            filtered_type_counts[sample_type] += 1
        
        print(f"\nFiltered type distribution:")
        for sample_type, count in filtered_type_counts.items():
            percentage = count / len(filtered) * 100
            print(f"  Type {sample_type}: {count} ({percentage:.1f}%)")
    
    if discarded:
        print(f"\nDiscarded samples (first 10):")
        for item in discarded[:10]:
            print(f"  Type {item['type']}: {item['question']}...")
    
    print(f"\nSaved to: {output_file}")
    
    return filtered, discarded


def main():
    parser = argparse.ArgumentParser(description='Filter old SFT data, keeping only Type A and C')
    parser.add_argument('--input', type=str, required=True,
                        help='Input JSONL file with old SFT data')
    parser.add_argument('--output', type=str, required=True,
                        help='Output JSONL file path')
    parser.add_argument('--keep_types', type=str, nargs='+', default=['A', 'C'],
                        help='Types to keep (default: A C)')
    
    args = parser.parse_args()
    
    if not Path(args.input).exists():
        raise FileNotFoundError(f"Input file not found: {args.input}")
    
    filter_old_data(args.input, args.output, args.keep_types)


if __name__ == '__main__':
    main()
