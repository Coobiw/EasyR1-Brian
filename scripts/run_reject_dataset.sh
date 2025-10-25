#!/bin/bash
# 基于 reject sampling 结果过滤数据集并上传到 HuggingFace

set -e  # 遇到错误立即退出

# 默认配置
DATASET_NAME="${DATASET_NAME:-Coobiw/agiqa3k_prompt_1013}"
RESULTS_DIR="${RESULTS_DIR:-./reject_sample_results}"
CHECKPOINT_NAME="${CHECKPOINT_NAME:-qwen2_5_vl_7b_baseline_agiqa3k}"
MAX_ERROR="${MAX_ERROR:-0.75}"
OUTPUT_REPO="${OUTPUT_REPO:-Coobiw/agiqa3k_prompt_rejected_1025}"
LOCAL_CACHE="${LOCAL_CACHE:-./filtered_dataset_cache}"

echo "===================================================="
echo "Reject Sampling Dataset Filter"
echo "===================================================="
echo ""
echo "Configuration:"
echo "  Source Dataset: $DATASET_NAME"
echo "  Results Dir: $RESULTS_DIR"
echo "  Checkpoint Name: $CHECKPOINT_NAME"
echo "  Max Error: $MAX_ERROR"
echo "  Output Repo: $OUTPUT_REPO"
echo "  Local Cache: $LOCAL_CACHE"
echo ""
echo "===================================================="
echo ""

# 检查结果目录是否存在
if [ ! -d "$RESULTS_DIR" ]; then
    echo "❌ Error: Results directory not found: $RESULTS_DIR"
    exit 1
fi

# 运行过滤脚本
python scripts/reject_sample_dataset.py \
    --dataset_name "$DATASET_NAME" \
    --results_dir "$RESULTS_DIR" \
    --checkpoint_name "$CHECKPOINT_NAME" \
    --max_error "$MAX_ERROR" \
    --output_repo "$OUTPUT_REPO" \
    --local_cache "$LOCAL_CACHE"

echo ""
echo "===================================================="
echo "✅ All done!"
echo "===================================================="
