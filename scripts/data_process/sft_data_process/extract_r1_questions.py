#!/usr/bin/env python3
"""
Extract questions and golden answers from Search-R1 parquet files.
This script reads the parquet files and extracts (question, golden_answer) pairs.
"""

import json
import argparse
import random
from collections import defaultdict, Counter
from datasets import Dataset
from pathlib import Path


def extract_from_parquet(parquet_path: str, output_path: str, max_samples: int = None, stratify: bool = True, seed: int = 42):
    """
    Extract questions and golden answers from parquet file.
    
    Args:
        parquet_path: Path to input parquet file
        output_path: Path to output JSONL file
        max_samples: Maximum number of samples to extract (None for all)
        stratify: Whether to stratify sampling by data_source (default: True)
        seed: Random seed for reproducibility
    """
    random.seed(seed)
    
    print(f"Loading parquet from {parquet_path}...")
    dataset = Dataset.from_parquet(parquet_path)
    
    print(f"Loaded {len(dataset)} samples")
    print(f"Columns: {dataset.column_names}")
    
    # Show first sample to understand structure
    if len(dataset) > 0:
        print("\nFirst sample structure:")
        sample = dataset[0]
        print(json.dumps(sample, indent=2, ensure_ascii=False, default=str)[:500])
    
    # First pass: collect all samples grouped by data_source
    samples_by_source = defaultdict(list)
    
    print("\nExtracting samples...")
    for idx, example in enumerate(dataset):
        # Extract question from prompt
        prompt = example.get('prompt', [])
        if isinstance(prompt, list) and len(prompt) > 0:
            # prompt is a list of messages
            user_content = None
            for msg in prompt:
                if isinstance(msg, dict) and msg.get('role') == 'user':
                    user_content = msg.get('content', '')
                    break
            
            # Extract the actual question from the prompt content
            # The prompt might contain instructions, so we extract just the question
            if user_content:
                # Find the question part (usually after "Question:")
                question = user_content
                if "Question:" in user_content:
                    question = user_content.split("Question:")[-1].strip()
                elif "question:" in user_content:
                    question = user_content.split("question:")[-1].strip()
                else:
                    # Use the whole content if no "Question:" marker
                    question = user_content.strip()
                
                # Remove question mark if already present, then ensure it ends with ?
                question = question.rstrip('?').strip()
                if not question.endswith('?'):
                    question += '?'
        else:
            # If prompt is a string
            question = str(prompt).strip()
            if "Question:" in question:
                question = question.split("Question:")[-1].strip()
            question = question.rstrip('?').strip()
            if not question.endswith('?'):
                question += '?'
        
        # Extract golden answer
        ground_truth = example.get('reward_model', {})
        if isinstance(ground_truth, dict):
            gt_target = ground_truth.get('ground_truth', {})
            if isinstance(gt_target, dict):
                golden_answers = gt_target.get('target', [])
            else:
                golden_answers = gt_target
        else:
            golden_answers = ground_truth
        
        # Handle list or string
        if isinstance(golden_answers, list):
            if len(golden_answers) > 0:
                golden_answer = str(golden_answers[0]).strip()
            else:
                print(f"Warning: Sample {idx} has empty golden_answers")
                continue
        else:
            golden_answer = str(golden_answers).strip()
        
        if not question or not golden_answer:
            print(f"Warning: Sample {idx} has empty question or answer, skipping")
            continue
        
        data_source = example.get('data_source', 'unknown')
        
        samples_by_source[data_source].append({
            'question': question,
            'golden_answer': golden_answer,
            'index': idx,
            'data_source': data_source
        })
    
    # Print data source distribution
    print(f"\nData source distribution in dataset:")
    for source, samples_list in samples_by_source.items():
        print(f"  {source}: {len(samples_list)} samples")
    
    # Stratified sampling
    if stratify and max_samples and len(samples_by_source) > 1:
        print(f"\nStratified sampling {max_samples} samples...")
        
        # Calculate samples per source (proportional)
        total_samples = sum(len(samples_list) for samples_list in samples_by_source.values())
        samples = []
        
        for source, samples_list in samples_by_source.items():
            # Calculate proportional sample size
            source_ratio = len(samples_list) / total_samples
            source_max = max(1, int(max_samples * source_ratio))
            
            # Shuffle and sample
            random.shuffle(samples_list)
            sampled = samples_list[:source_max]
            samples.extend(sampled)
            print(f"  {source}: sampled {len(sampled)}/{len(samples_list)} samples")
        
        # If we have fewer samples than requested, fill from remaining
        if len(samples) < max_samples:
            remaining_needed = max_samples - len(samples)
            all_remaining = []
            for source, samples_list in samples_by_source.items():
                # Get samples not yet selected
                selected_indices = {s['index'] for s in samples}
                remaining = [s for s in samples_list if s['index'] not in selected_indices]
                all_remaining.extend(remaining)
            
            random.shuffle(all_remaining)
            samples.extend(all_remaining[:remaining_needed])
            print(f"  Added {min(remaining_needed, len(all_remaining))} more samples to reach {max_samples}")
        
        # Shuffle final samples
        random.shuffle(samples)
        samples = samples[:max_samples]  # Ensure we don't exceed max_samples
        
    elif max_samples:
        # Simple sampling without stratification
        print(f"\nSampling {max_samples} samples (no stratification)...")
        all_samples = []
        for samples_list in samples_by_source.values():
            all_samples.extend(samples_list)
        
        random.shuffle(all_samples)
        samples = all_samples[:max_samples]
    else:
        # No max_samples, return all
        samples = []
        for samples_list in samples_by_source.values():
            samples.extend(samples_list)
        random.shuffle(samples)
    
    print(f"\nExtracted {len(samples)} valid samples")
    
    # Save to JSONL
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        for sample in samples:
            f.write(json.dumps(sample, ensure_ascii=False) + '\n')
    
    print(f"Saved to {output_path}")
    
    # Print statistics
    print(f"\nStatistics:")
    print(f"  Total samples: {len(samples)}")
    if samples:
        question_lengths = [len(s['question']) for s in samples]
        answer_lengths = [len(s['golden_answer']) for s in samples]
        print(f"  Question length - Min: {min(question_lengths)}, Max: {max(question_lengths)}, Avg: {sum(question_lengths)/len(question_lengths):.1f}")
        print(f"  Answer length - Min: {min(answer_lengths)}, Max: {max(answer_lengths)}, Avg: {sum(answer_lengths)/len(answer_lengths):.1f}")
        
        # Show data source distribution
        sources = {}
        for s in samples:
            src = s.get('data_source', 'unknown')
            sources[src] = sources.get(src, 0) + 1
        print(f"  Data sources: {sources}")
    
    return samples


def main():
    parser = argparse.ArgumentParser(description='Extract questions and golden answers from Search-R1 parquet files')
    parser.add_argument('--input', type=str, required=True,
                        help='Input parquet file path')
    parser.add_argument('--output', type=str, required=True,
                        help='Output JSONL file path')
    parser.add_argument('--max_samples', type=int, default=None,
                        help='Maximum number of samples to extract (default: all)')
    parser.add_argument('--stratify', action='store_true', default=True,
                        help='Stratify sampling by data_source (default: True)')
    parser.add_argument('--no_stratify', dest='stratify', action='store_false',
                        help='Do not stratify sampling')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility (default: 42)')
    
    args = parser.parse_args()
    
    extract_from_parquet(args.input, args.output, args.max_samples, args.stratify, args.seed)


if __name__ == '__main__':
    main()
