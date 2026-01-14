#!/usr/bin/env python3
"""
检查 SFT 训练数据是否有问题（可能导致 SIGFPE 的样本）
"""

import pandas as pd
import argparse
from transformers import AutoTokenizer

def check_data(data_path, model_path, tokenizer_path=None):
    """检查数据中是否有问题样本"""
    if tokenizer_path is None:
        tokenizer_path = model_path
    
    print(f"Loading tokenizer from {tokenizer_path}...")
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_path,
        trust_remote_code=True
    )
    
    print(f"Loading data from {data_path}...")
    df = pd.read_parquet(data_path)
    
    print(f"\nTotal examples: {len(df)}")
    print(f"Columns: {df.columns.tolist()}")
    print("\n" + "="*80)
    print("Checking for problematic samples...")
    print("="*80)
    
    problematic_indices = []
    empty_answer_count = 0
    zero_token_count = 0
    all_masked_count = 0
    
    for idx, row in df.iterrows():
        issues = []
        
        # 检查是否有 labels 字段（预tokenized数据）
        if 'labels' in row:
            labels = row['labels']
            if isinstance(labels, list):
                valid_tokens = sum(1 for l in labels if l != -100)
                if valid_tokens == 0:
                    issues.append(f"all_tokens_masked(valid={valid_tokens})")
                    all_masked_count += 1
            elif hasattr(labels, '__iter__'):
                # 如果是 numpy array 或其他可迭代对象
                try:
                    valid_tokens = sum(1 for l in labels if l != -100)
                    if valid_tokens == 0:
                        issues.append(f"all_tokens_masked(valid={valid_tokens})")
                        all_masked_count += 1
                except:
                    pass
        
        # 检查 answer 字段（如果存在）
        if 'answer' in row:
            answer = str(row['answer']) if pd.notna(row['answer']) else ""
            if not answer or len(answer.strip()) == 0:
                issues.append("empty_answer")
                empty_answer_count += 1
            else:
                # 检查 tokenize 后的长度
                try:
                    tokens = tokenizer(answer, add_special_tokens=False)['input_ids']
                    if len(tokens) == 0:
                        issues.append(f"zero_tokens_after_tokenize")
                        zero_token_count += 1
                except Exception as e:
                    issues.append(f"tokenize_error: {e}")
        
        if issues:
            problematic_indices.append((idx, issues))
            if len(problematic_indices) <= 10:  # 只显示前10个
                print(f"Row {idx}: {', '.join(issues)}")
                if 'answer' in row:
                    answer_preview = str(row['answer'])[:100] if pd.notna(row['answer']) else "N/A"
                    print(f"  Answer preview: {answer_preview}...")
    
    print("\n" + "="*80)
    print("Summary:")
    print("="*80)
    print(f"Total problematic samples: {len(problematic_indices)}")
    print(f"  - Empty answer: {empty_answer_count}")
    print(f"  - Zero tokens after tokenize: {zero_token_count}")
    print(f"  - All tokens masked: {all_masked_count}")
    
    if len(problematic_indices) > 10:
        print(f"\n(Showing first 10, total {len(problematic_indices)} problematic samples)")
        print(f"All problematic indices: {[idx for idx, _ in problematic_indices]}")
    
    if len(problematic_indices) > 0:
        print("\n⚠️  WARNING: Found problematic samples that may cause SIGFPE!")
        print("   These samples should be removed or fixed before training.")
    else:
        print("\n✓ No problematic samples found.")
    
    return problematic_indices

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Check SFT training data for problematic samples')
    parser.add_argument('--data_path', type=str, required=True,
                        help='Path to parquet file')
    parser.add_argument('--model_path', type=str, 
                        default='/mnt/dolphinfs/ssd_pool/docker/user/hadoop-ai-search/chongwenyue/model/Qwen2.5-3B-Instruct',
                        help='Path to model (for tokenizer)')
    parser.add_argument('--tokenizer_path', type=str, default=None,
                        help='Path to tokenizer (if different from model_path)')
    
    args = parser.parse_args()
    
    problematic = check_data(args.data_path, args.model_path, args.tokenizer_path)
    
    if len(problematic) > 0:
        exit(1)
    else:
        exit(0)
