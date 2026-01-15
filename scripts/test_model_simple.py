#!/usr/bin/env python3
"""
Simple test script to verify the trained model works as a normal language model.
Tests basic generation without AutoSearch protocol.
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import sys
import time

def test_model_simple(model_path: str, question: str = "1+1等于几？", max_new_tokens: int = 100, force_cpu: bool = False):
    """Test model with a simple question"""
    
    print("=" * 60)
    print("Simple Model Test (No AutoSearch Protocol)")
    print("=" * 60)
    print(f"Model Path: {model_path}")
    print(f"Question: {question}")
    print(f"Max New Tokens: {max_new_tokens}")
    print("=" * 60)
    
    # Load tokenizer
    print("\n[1/4] Loading tokenizer...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        print(f"✓ Tokenizer loaded. Vocab size: {len(tokenizer)}")
    except Exception as e:
        print(f"✗ Failed to load tokenizer: {e}")
        return
    
    # Load model
    print("\n[2/4] Loading model...")
    if force_cpu:
        device = "cpu"
        print("  Forced CPU mode")
    else:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    try:
        # Try to load with specific dtype and disable flash attention to avoid issues
        # Flash Attention requires float16/bfloat16, but may cause hangs if not properly configured
        print("  Attempting to load with float16 and disabled flash attention...")
        
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            trust_remote_code=True,
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
            device_map="auto" if device == "cuda" else None,
            attn_implementation="eager",  # Disable Flash Attention, use eager implementation
        )
        if device == "cpu":
            model = model.to(device)
        model.eval()
        print(f"✓ Model loaded on {device} with dtype={model.dtype}")
        
        # Check GPU memory
        if device == "cuda":
            allocated = torch.cuda.memory_allocated() / 1024**3
            reserved = torch.cuda.memory_reserved() / 1024**3
            print(f"  GPU Memory: {allocated:.2f}GB allocated, {reserved:.2f}GB reserved")
            
            # Test a simple forward pass before generation
            print("\n  Testing a simple forward pass...")
            test_input = torch.randint(0, 1000, (1, 5)).to(device)
            try:
                with torch.no_grad():
                    _ = model(test_input)
                torch.cuda.synchronize()
                print("  ✓ Forward pass successful")
            except Exception as e:
                print(f"  ✗ Forward pass failed: {e}")
                import traceback
                traceback.print_exc()
                return
    except Exception as e:
        print(f"✗ Failed to load model: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Prepare input
    print("\n[3/4] Preparing input...")
    try:
        # Use chat template if available
        if tokenizer.chat_template:
            messages = [{"role": "user", "content": question}]
            prompt = tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=False
            )
            print(f"✓ Using chat template")
        else:
            # Simple prompt format
            prompt = f"User: {question}\nAssistant: "
            print(f"✓ Using simple prompt format")
        
        print(f"\nPrompt (first 200 chars):\n{prompt[:200]}...")
        
        # Tokenize
        input_ids = tokenizer.encode(prompt, return_tensors='pt').to(device)
        attention_mask = torch.ones_like(input_ids)
        print(f"✓ Tokenized to {input_ids.shape[1]} tokens")
    except Exception as e:
        print(f"✗ Failed to prepare input: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Generate
    print("\n[4/4] Generating response...")
    print("=" * 60)
    print("Generating (this may take 10-60 seconds)...")
    sys.stdout.flush()
    
    start_time = time.time()
    
    try:
        # Clear generation config
        if hasattr(model, 'generation_config'):
            from transformers import GenerationConfig
            model.generation_config = GenerationConfig(
                do_sample=False,  # Greedy decoding
                temperature=None,
                top_p=None,
                top_k=None,
            )
        
        # Prepare generation kwargs
        gen_kwargs = {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'max_new_tokens': max_new_tokens,
            'do_sample': False,  # Greedy decoding
            'pad_token_id': tokenizer.pad_token_id,
            'eos_token_id': tokenizer.eos_token_id,
        }
        
        print(f"Generation kwargs: do_sample={gen_kwargs['do_sample']}, "
              f"max_new_tokens={gen_kwargs['max_new_tokens']}")
        sys.stdout.flush()
        
        # Test CUDA sync before generation
        if device == "cuda":
            print("Testing CUDA synchronization before generation...")
            sync_start = time.time()
            try:
                torch.cuda.synchronize()
                sync_elapsed = time.time() - sync_start
                print(f"✓ CUDA sync completed in {sync_elapsed:.3f}s")
                if sync_elapsed > 1.0:
                    print(f"  WARNING: CUDA sync took {sync_elapsed:.1f}s (unusually long)")
            except Exception as e:
                print(f"✗ CUDA sync FAILED: {e}")
                print("  This indicates CUDA operations may be blocked!")
        
        # Generate
        print("\nCalling model.generate()...")
        print(f"  Input shape: {input_ids.shape}")
        print(f"  Model dtype: {model.dtype}")
        print(f"  Input dtype: {input_ids.dtype}")
        sys.stdout.flush()
        
        # Ensure input and model are on same device and dtype
        if device == "cuda":
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            # Sync before generation
            torch.cuda.synchronize()
            print("  ✓ CUDA synchronized before generation")
            sys.stdout.flush()
        
        # Add timeout mechanism using threading
        import threading
        generation_result = [None]
        generation_error = [None]
        generation_done = threading.Event()
        
        def run_generation():
            try:
                with torch.no_grad():
                    result = model.generate(**gen_kwargs)
                generation_result[0] = result
                generation_done.set()
            except Exception as e:
                generation_error[0] = e
                generation_done.set()
        
        # Start generation in a separate thread
        gen_thread = threading.Thread(target=run_generation, daemon=False)
        gen_thread.start()
        
        # Wait with timeout
        print("  Waiting for generation (max 120s)...")
        sys.stdout.flush()
        
        timeout = 120  # 2 minutes
        check_interval = 2  # Check every 2 seconds
        
        for i in range(timeout // check_interval):
            if generation_done.wait(timeout=check_interval):
                break
            elapsed = (i + 1) * check_interval
            print(f"  Still generating... {elapsed}s elapsed")
            sys.stdout.flush()
            
            # Check CUDA status
            if device == "cuda":
                try:
                    torch.cuda.synchronize()
                except Exception as e:
                    print(f"  ⚠ CUDA sync check failed: {e}")
        
        # Check if thread is still alive
        if gen_thread.is_alive():
            elapsed = time.time() - start_time
            print(f"\n  ✗ Generation timeout after {elapsed:.1f}s!")
            print(f"  Thread still alive: {gen_thread.is_alive()}")
            print(f"  This suggests model.generate() is stuck or blocked")
            
            # Try to get some info about what's happening
            if device == "cuda":
                try:
                    allocated = torch.cuda.memory_allocated() / 1024**3
                    print(f"  GPU Memory: {allocated:.2f}GB")
                    torch.cuda.synchronize()
                    print(f"  CUDA sync: OK")
                except Exception as e:
                    print(f"  CUDA sync failed: {e}")
            
            print("\n  Possible causes:")
            print("    1. Flash Attention issue (try disabling)")
            print("    2. CUDA driver/kernel issue")
            print("    3. Model architecture incompatibility")
            print("    4. Memory corruption")
            return
        
        # Check for errors
        if generation_error[0] is not None:
            raise generation_error[0]
        
        if generation_result[0] is None:
            raise RuntimeError("Generation did not complete")
        
        outputs = generation_result[0]
        
        elapsed = time.time() - start_time
        print(f"\n✓ Generation completed in {elapsed:.1f}s")
        print(f"  Output shape: {outputs.shape}")
        print(f"  Generated {outputs.shape[1] - input_ids.shape[1]} new tokens")
        
    except KeyboardInterrupt:
        elapsed = time.time() - start_time
        print(f"\n✗ Generation interrupted by user after {elapsed:.1f}s")
        return
    except RuntimeError as e:
        elapsed = time.time() - start_time
        error_str = str(e)
        print(f"\n✗ Generation failed after {elapsed:.1f}s")
        if "CUDA" in error_str or "cuda" in error_str or "out of memory" in error_str.lower():
            print(f"  CUDA ERROR: {error_str}")
            print("  Trying to clear CUDA cache...")
            torch.cuda.empty_cache()
        else:
            print(f"  RuntimeError: {error_str}")
        import traceback
        traceback.print_exc()
        return
    except Exception as e:
        elapsed = time.time() - start_time
        print(f"\n✗ Generation failed after {elapsed:.1f}s: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Decode and print
    print("\n" + "=" * 60)
    print("RESULT:")
    print("=" * 60)
    
    # Decode full output
    full_text = tokenizer.decode(outputs[0], skip_special_tokens=False)
    print(f"\nFull Output:\n{full_text}")
    
    # Decode only generated part
    generated_tokens = outputs[0][input_ids.shape[1]:]
    generated_text = tokenizer.decode(generated_tokens, skip_special_tokens=True)
    print(f"\nGenerated Text Only:\n{generated_text}")
    
    print("\n" + "=" * 60)
    print("Test Complete!")
    print("=" * 60)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Simple model test without AutoSearch protocol")
    parser.add_argument("--model_path", type=str, required=True,
                        help="Path to the trained model")
    parser.add_argument("--question", type=str, default="1+1等于几？",
                        help="Test question (default: '1+1等于几？')")
    parser.add_argument("--max_new_tokens", type=int, default=100,
                        help="Maximum number of new tokens to generate (default: 100)")
    parser.add_argument("--cpu", action="store_true",
                        help="Force CPU mode (slower but useful for debugging)")
    
    args = parser.parse_args()
    
    test_model_simple(
        model_path=args.model_path,
        question=args.question,
        max_new_tokens=args.max_new_tokens,
        force_cpu=args.cpu
    )
