#!/usr/bin/env python3
"""
Quick script to check CUDA status and diagnose potential blocking issues.
Run this in a separate terminal while eval_autosearch_sft.py is running.
"""

import torch
import time
import sys

def check_cuda():
    """Check CUDA availability and status"""
    print("=" * 60)
    print("CUDA Diagnostic Check")
    print("=" * 60)
    
    # Check if CUDA is available
    print(f"CUDA Available: {torch.cuda.is_available()}")
    if not torch.cuda.is_available():
        print("ERROR: CUDA is not available!")
        return
    
    # Check device count
    print(f"CUDA Device Count: {torch.cuda.device_count()}")
    
    # Check current device
    if torch.cuda.is_available():
        print(f"Current Device: {torch.cuda.current_device()}")
        print(f"Device Name: {torch.cuda.get_device_name(0)}")
    
    # Check memory
    print("\n--- GPU Memory Status ---")
    for i in range(torch.cuda.device_count()):
        allocated = torch.cuda.memory_allocated(i) / 1024**3
        reserved = torch.cuda.memory_reserved(i) / 1024**3
        max_allocated = torch.cuda.max_memory_allocated(i) / 1024**3
        print(f"GPU {i}:")
        print(f"  Allocated: {allocated:.2f} GB")
        print(f"  Reserved: {reserved:.2f} GB")
        print(f"  Max Allocated: {max_allocated:.2f} GB")
    
    # Test synchronization (this will hang if CUDA operations are blocked)
    print("\n--- Testing CUDA Synchronization ---")
    print("Attempting torch.cuda.synchronize()...")
    sys.stdout.flush()
    
    start = time.time()
    try:
        torch.cuda.synchronize()
        elapsed = time.time() - start
        print(f"✓ CUDA synchronize() completed in {elapsed:.3f}s")
        if elapsed > 1.0:
            print(f"  WARNING: Synchronization took {elapsed:.1f}s, which is unusually long!")
    except Exception as e:
        elapsed = time.time() - start
        print(f"✗ CUDA synchronize() FAILED after {elapsed:.1f}s: {e}")
        print("  This indicates CUDA operations may be blocked!")
    
    # Test a simple CUDA operation
    print("\n--- Testing Simple CUDA Operation ---")
    print("Creating a small tensor on GPU...")
    sys.stdout.flush()
    
    start = time.time()
    try:
        x = torch.randn(10, 10).cuda()
        torch.cuda.synchronize()
        elapsed = time.time() - start
        print(f"✓ Tensor creation completed in {elapsed:.3f}s")
        del x
        torch.cuda.empty_cache()
    except Exception as e:
        elapsed = time.time() - start
        print(f"✗ Tensor creation FAILED after {elapsed:.1f}s: {e}")
        print("  This indicates CUDA operations are blocked!")
    
    print("\n" + "=" * 60)
    print("Diagnostic complete!")
    print("=" * 60)
    print("\nIf synchronize() or tensor creation hangs or takes >5s,")
    print("it indicates CUDA operations are blocked.")
    print("\nPossible causes:")
    print("  1. Another process is holding the GPU")
    print("  2. GPU driver issue")
    print("  3. CUDA context corruption")
    print("  4. Model.generate() is stuck in a long computation")


if __name__ == "__main__":
    check_cuda()
