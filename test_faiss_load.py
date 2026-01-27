#!/usr/bin/env python3
"""
Test script to diagnose FAISS index loading issues
"""
import faiss
import time
import os
import sys

index_path = "/mnt/dolphinfs/ssd_pool/docker/user/hadoop-ai-search/chongwenyue/RL-Factory/corpus/e5_Flat.index"

print(f"Testing FAISS index loading...")
print(f"Index path: {index_path}")
print(f"File exists: {os.path.exists(index_path)}")
print(f"File size: {os.path.getsize(index_path) / (1024**3):.2f} GB")
print(f"File readable: {os.access(index_path, os.R_OK)}")

print(f"\n[1/3] Testing file access (read first 1MB)...")
try:
    with open(index_path, 'rb') as f:
        data = f.read(1024 * 1024)
    print(f"✓ File access OK, read {len(data)} bytes")
except Exception as e:
    print(f"✗ File access failed: {e}")
    sys.exit(1)

print(f"\n[2/3] Testing FAISS read_index with mmap (timeout: 60s)...")
start_time = time.time()
try:
    # Set a timeout using signal (Unix only)
    import signal
    
    def timeout_handler(signum, frame):
        raise TimeoutError("FAISS read_index timeout after 60 seconds")
    
    signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(60)  # 60 second timeout
    
    index = faiss.read_index(index_path, faiss.IO_FLAG_MMAP)
    signal.alarm(0)  # Cancel timeout
    
    elapsed = time.time() - start_time
    print(f"✓ FAISS mmap mode loaded in {elapsed:.2f} seconds")
    print(f"  Index type: {type(index)}")
    print(f"  Index dimension: {index.d}")
    print(f"  Index size: {index.ntotal}")
    
except TimeoutError as e:
    elapsed = time.time() - start_time
    print(f"✗ {e} (elapsed: {elapsed:.2f}s)")
    print(f"  Process may be stuck. Check with: ps aux | grep python")
    sys.exit(1)
except Exception as e:
    signal.alarm(0)  # Cancel timeout
    elapsed = time.time() - start_time
    print(f"✗ FAISS mmap mode failed after {elapsed:.2f}s: {e}")
    print(f"\n[3/3] Trying normal mode (timeout: 120s)...")
    start_time = time.time()
    try:
        signal.alarm(120)  # 2 minute timeout for normal mode
        index = faiss.read_index(index_path)
        signal.alarm(0)
        elapsed = time.time() - start_time
        print(f"✓ FAISS normal mode loaded in {elapsed:.2f} seconds")
    except TimeoutError:
        elapsed = time.time() - start_time
        print(f"✗ Normal mode also timed out after {elapsed:.2f}s")
        sys.exit(1)
    except Exception as e2:
        signal.alarm(0)
        elapsed = time.time() - start_time
        print(f"✗ Normal mode also failed after {elapsed:.2f}s: {e2}")
        sys.exit(1)

print(f"\n✓ All tests passed!")
