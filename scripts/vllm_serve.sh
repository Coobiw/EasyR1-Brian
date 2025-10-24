CKPT_PATH=$1
MODEL_NAME=$2
PORT=${MODEL_PORT:-8000}  # 使用环境变量或默认 8000

vllm serve $CKPT_PATH \
  --served-model-name $MODEL_NAME \
  --chat-template scripts/think_chat_template.jina \
  --max-model-len 8192 \
  --tensor-parallel-size 4 \
  --port $PORT \
  --host 0.0.0.0 \
  --dtype bfloat16 \
  --limit-mm-per-prompt image=5,video=5 \
  --mm-processor-kwargs '{"max_pixels": 1048576, "min_pixels": 262144}'