#!/bin/bash
# Find CUDA installation on this system

echo "=== CUDA Diagnostic ==="
echo ""

echo "1. Check nvcc location:"
which nvcc 2>/dev/null || echo "   nvcc not in PATH"

echo ""
echo "2. Check nvidia-smi:"
nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null | head -1 || echo "   nvidia-smi failed"

echo ""
echo "3. Common CUDA paths:"
for path in /usr/local/cuda /usr/local/cuda-* /usr/lib/cuda /opt/cuda; do
    if [ -d "$path" ]; then
        echo "   FOUND: $path"
        if [ -f "$path/bin/nvcc" ]; then
            echo "          Has nvcc: YES"
        fi
    fi
done

echo ""
echo "4. Environment variables:"
echo "   CUDA_HOME=$CUDA_HOME"
echo "   CUDA_PATH=$CUDA_PATH"
echo "   LD_LIBRARY_PATH=$LD_LIBRARY_PATH"

echo ""
echo "5. Find nvcc anywhere:"
find /usr -name "nvcc" -type f 2>/dev/null | head -5

echo ""
echo "=== Suggested fix ==="
NVCC_PATH=$(which nvcc 2>/dev/null)
if [ -n "$NVCC_PATH" ]; then
    CUDA_DIR=$(dirname $(dirname $NVCC_PATH))
    echo "Add to your shell or run before python:"
    echo "  export CUDA_HOME=$CUDA_DIR"
else
    echo "nvcc not found in PATH. Check paths above and set CUDA_HOME manually."
fi
