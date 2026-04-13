#!/bin/bash

# GPU Keepalive Script
# Keeps GPU utilization above threshold to prevent instance termination

GPU_KEEPALIVE_PID=""
UTILIZATION_THRESHOLD=50

trap 'echo "Shutting down GPU keepalive..."; kill $GPU_KEEPALIVE_PID 2>/dev/null; wait $GPU_KEEPALIVE_PID 2>/dev/null' EXIT

echo "Starting GPU keepalive on cuda:0 (threshold: ${UTILIZATION_THRESHOLD}%)..."
KEEPALIVE_LOG=$(mktemp)

uv run python -c "
import signal
import sys
import time
import torch

signal.signal(signal.SIGTERM, lambda *_: sys.exit(0))

# Initialize tensor for matrix multiplication
t = torch.randn(4096, 4096, device='cuda:0', dtype=torch.bfloat16)
print('GPU keepalive running on cuda:0...', flush=True)

while True:
    torch.matmul(t, t)
    time.sleep(0.0005)
" > "$KEEPALIVE_LOG" 2>&1 &

GPU_KEEPALIVE_PID=$!

echo "Waiting for GPU keepalive to initialize..."
until grep -q "GPU keepalive running" "$KEEPALIVE_LOG" 2>/dev/null; do
    sleep 1
done

echo "GPU keepalive confirmed running (PID: $GPU_KEEPALIVE_PID)."
echo "Keepalive will run until script exits or is terminated."
echo "Press Ctrl+C to stop."

wait $GPU_KEEPALIVE_PID
