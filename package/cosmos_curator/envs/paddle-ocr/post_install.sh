#!/bin/bash
# Replace cu130 torch (from core) with CPU-only torch so it doesn't conflict
# with paddlepaddle-gpu's bundled CUDA runtime.
pip uninstall torch torchvision -y
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install paddlepaddle-gpu==3.3.0 -i https://www.paddlepaddle.org.cn/packages/stable/cu129/
