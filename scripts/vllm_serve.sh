CKPT_PATH=$1
MODEL_NAME=$2

vllm serve $CKPT_PATH \
  --served-model-name $MODEL_NAME \
  --enable-auto-tool-choice \
  --tool-call-parser hermes \
  --max-model-len 8192 \
  --tensor-parallel-size 4 \
  --port 8000 \
  --host 0.0.0.0 \
  --dtype bfloat16 \
  --limit-mm-per-prompt image=5,video=5 \
  --mm-processor-kwargs '{"max_pixels": 1048576, "min_pixels": 262144}'