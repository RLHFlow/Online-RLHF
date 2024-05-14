#!/bin/bash

# 检查是否提供了参数（模型路径）
if [ $# -eq 0 ]; then
    echo "Usage: $0 <model_path>"
    exit 1
fi

# 将第一个命令行参数赋值给MODEL_PATH变量
MODEL_PATH=$1

# 使用for循环启动8个服务实例，每个实例使用不同的GPU和端口
for i in {0..7}
do
    CUDA_VISIBLE_DEVICES=$i python -m vllm.entrypoints.api_server \
        --model $MODEL_PATH \
        --gpu-memory-utilization=0.9 \
        --max-num-seqs=200 \
        --host 127.0.0.1 --tensor-parallel-size 1 \
        --port $((8000+i)) \
    &
done
