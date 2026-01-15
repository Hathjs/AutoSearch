#!/usr/bin/env python3
"""
Extract questions and golden answers from Search-R1 parquet files.
This script reads the parquet files and extracts (question, golden_answer) pairs.
"""

import json
import argparse
from datasets import Dataset
from pathlib import Path


def extract_from_parquet(parquet_path: str, output_path: str, max_samples: int = None):
    """
    Extract questions and golden answers from parquet file.
    
    Args:
        parquet_path: Path to input parquet file
        output_path: Path to output JSONL file
        max_samples: Maximum number of samples to extract (None for all)
    """
    print(f"Loading parquet from {parquet_path}...")
    dataset = Dataset.from_parquet(parquet_path)
    
    print(f"Loaded {len(dataset)} samples")
    print(f"Columns: {dataset.column_names}")
    
    # Show first sample to understand structure
    if len(dataset) > 0:
        print("\nFirst sample structure:")
        sample = dataset[0]
        print(json.dumps(sample, indent=2, ensure_ascii=False, default=str)[:500])
    
    samples = []
    
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
        
        samples.append({
            'question': question,
            'golden_answer': golden_answer,
            'index': idx,
            'data_source': example.get('data_source', 'unknown')
        })
        
        if max_samples and len(samples) >= max_samples:
            break
    
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
    
    args = parser.parse_args()
    
    extract_from_parquet(args.input, args.output, args.max_samples)


if __name__ == '__main__':
    main()
