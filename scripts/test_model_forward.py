#!/usr/bin/env python3
"""
Test model forward pass only (no generation).
This helps isolate if the issue is in generation or in the model itself.
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import sys

def test_forward_only(model_path: str):
    """Test model forward pass without generation"""
    
    print("=" * 60)
    print("Model Forward Pass Test (No Generation)")
    print("=" * 60)
    print(f"Model Path: {model_path}")
    print("=" * 60)
    
    # Load tokenizer
    print("\n[1/3] Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    print(f"✓ Tokenizer loaded")
    
    # Load model
    print("\n[2/3] Loading model...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        trust_remote_code=True,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        device_map="auto" if device == "cuda" else None,
        attn_implementation="eager",
    )
    if device == "cpu":
        model = model.to(device)
    model.eval()
    print(f"✓ Model loaded on {device} with dtype={model.dtype}")
    
    # Create simple input
    print("\n[3/3] Testing forward pass...")
    test_text = "Hello, how are you?"
    input_ids = tokenizer.encode(test_text, return_tensors='pt').to(device)
    attention_mask = torch.ones_like(input_ids)
    
    print(f"  Input text: '{test_text}'")
    print(f"  Input shape: {input_ids.shape}")
    print(f"  Testing forward pass...")
    sys.stdout.flush()
    
    try:
        with torch.no_grad():
            # Single forward pass
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            
            if device == "cuda":
                torch.cuda.synchronize()
            
            print(f"  ✓ Forward pass successful!")
            print(f"  Output logits shape: {outputs.logits.shape}")
            
            # Test multiple forward passes
            print("\n  Testing 5 forward passes...")
            for i in range(5):
                with torch.no_grad():
                    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                if device == "cuda":
                    torch.cuda.synchronize()
                print(f"  ✓ Forward pass {i+1}/5 completed")
            
            print("\n  ✓ All forward passes successful!")
            print("\n  If forward passes work but generate() hangs,")
            print("  the issue is likely in the generation loop/optimizations.")
            
    except Exception as e:
        print(f"\n  ✗ Forward pass FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    print("\n" + "=" * 60)
    print("Test Complete!")
    print("=" * 60)
    return True

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    args = parser.parse_args()
    test_forward_only(args.model_path)
