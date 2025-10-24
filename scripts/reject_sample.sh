#!/bin/bash
# Reject Sampling Complete Pipeline
# 
# 这个脚本展示了如何从启动服务到获得结果的完整流程

# 全局变量
VLLM_PID=""

# 清理函数
cleanup() {
    local exit_code=$?
    echo ""
    echo "=================================================="
    echo "Cleanup: Stopping vLLM Service..."
    echo "=================================================="
    
    if [ -n "$VLLM_PID" ]; then
        echo "Killing vLLM process (PID: $VLLM_PID)..."
        
        # 尝试杀掉进程组
        pkill -P $VLLM_PID 2>/dev/null || true
        kill $VLLM_PID 2>/dev/null || true
        
        # 等待一下
        sleep 2
        
        # 强制杀掉（如果还活着）
        if ps -p $VLLM_PID > /dev/null 2>&1; then
            echo "Force killing vLLM process..."
            kill -9 $VLLM_PID 2>/dev/null || true
        fi
        
        echo "✅ vLLM service stopped"
    fi
    
    # 清理端口占用
    PORT_PID=$(lsof -ti:${MODEL_PORT:-8000} 2>/dev/null)
    if [ -n "$PORT_PID" ]; then
        echo "Killing process on port ${MODEL_PORT:-8000} (PID: $PORT_PID)..."
        kill -9 $PORT_PID 2>/dev/null || true
    fi
    
    echo "=================================================="
    
    # 如果是错误退出，显示提示信息
    if [ $exit_code -ne 0 ]; then
        echo ""
        echo "❌ Script exited with error code: $exit_code"
        echo "   vLLM service has been cleaned up."
        echo ""
        echo "To manually clean up any remaining processes, run:"
        echo "   MODEL_PORT=${MODEL_PORT:-8000} bash scripts/kill_vllm.sh"
        echo "   or: bash scripts/kill_vllm.sh ${MODEL_PORT:-8000}"
    fi
    
    exit $exit_code
}

# 捕获退出信号
trap cleanup EXIT INT TERM

set -e  # 遇到错误立即退出

# ==================== 配置部分 ====================
# 修改这些变量以适配你的环境

# 模型路径（修改为你的实际路径）
MODEL_PATH="${MODEL_PATH:-/code/All-In-One/qbw/EasyR1-20250410/cache/ckpt/Qwen2.5-VL-7B-Instruct}"

# 模型名称（用于标识结果）
MODEL_NAME="${MODEL_NAME:-qwen2_5_vl_7b_baseline}"

# vLLM 服务端口
MODEL_PORT="${MODEL_PORT:-8000}"

# Reject Sampling 参数
NUM_ROLLOUT="${NUM_ROLLOUT:-16}"      # 每条数据 rollout 次数
TEMPERATURE="${TEMPERATURE:-1.0}"     # 采样温度
CONCURRENCY="${CONCURRENCY:-16}"      # 并发数
MAX_RETRIES="${MAX_RETRIES:-10}"      # 最大重试次数
MAX_DATA="${MAX_DATA:-}"              # 留空表示处理全部数据，设置数字表示只处理前 N 条

# 输出目录
OUTPUT_DIR="${OUTPUT_DIR:-./reject_sample_results}"

# vLLM 日志目录
LOG_DIR="${LOG_DIR:-./logs}"

# ==================================================

echo "=================================================="
echo "Reject Sampling Complete Pipeline"
echo "=================================================="
echo ""
echo "Configuration:"
echo "  Model Path: $MODEL_PATH"
echo "  Model Name: $MODEL_NAME"
echo "  Model Port: $MODEL_PORT"
echo "  Rollouts per data: $NUM_ROLLOUT"
echo "  Temperature: $TEMPERATURE"
echo "  Concurrency: $CONCURRENCY"
echo "  Max Retries: $MAX_RETRIES"
if [ -n "$MAX_DATA" ]; then
    echo "  Max Data: $MAX_DATA"
else
    echo "  Max Data: All"
fi
echo "  Output Dir: $OUTPUT_DIR"
echo ""
echo "=================================================="
echo ""

# 创建日志目录
mkdir -p $LOG_DIR

# Step 1: 启动 vLLM 服务
echo "Step 1: Starting vLLM Service..."
echo "-----------------------------------------------"

LOG_FILE="$LOG_DIR/${MODEL_NAME}_vllm.log"
echo "Server logs will be saved to: $LOG_FILE"

# 启动 vLLM 服务（后台运行）
nohup bash scripts/vllm_serve.sh "$MODEL_PATH" "$MODEL_NAME" > "$LOG_FILE" 2>&1 &
VLLM_PID=$!

echo "vLLM service started with PID: $VLLM_PID"
echo ""

# Step 2: 等待服务启动
echo "Step 2: Waiting for vLLM Service to be Ready..."
echo "-----------------------------------------------"

MAX_WAIT=600  # 最多等待 10 分钟
WAIT_COUNT=0
WAIT_INTERVAL=10

while [ $WAIT_COUNT -lt $MAX_WAIT ]; do
    if curl -s http://localhost:$MODEL_PORT/health > /dev/null 2>&1; then
        echo "✅ vLLM service is ready!"
        break
    fi
    echo "Waiting... ($WAIT_COUNT/$MAX_WAIT seconds)"
    sleep $WAIT_INTERVAL
    WAIT_COUNT=$((WAIT_COUNT + WAIT_INTERVAL))
done

if [ $WAIT_COUNT -ge $MAX_WAIT ]; then
    echo "❌ Timeout: vLLM service did not start within $MAX_WAIT seconds"
    echo "   Check logs at: $LOG_FILE"
    kill $VLLM_PID 2>/dev/null || true
    exit 1
fi

# 额外等待确保模型完全加载
echo "Waiting additional 30 seconds for model to fully load..."
sleep 30
echo ""

# Step 3: 运行 Reject Sampling
echo "Step 3: Running Reject Sampling..."
echo "-----------------------------------------------"

CMD="python scripts/reject_sample.py \
    --model_path \"$MODEL_PATH\" \
    --model_name \"$MODEL_NAME\" \
    --model_port $MODEL_PORT \
    --num_rollout $NUM_ROLLOUT \
    --temperature $TEMPERATURE \
    --concurrency $CONCURRENCY \
    --max_retries $MAX_RETRIES \
    --output_dir \"$OUTPUT_DIR\" \
    --skip_vllm_launch"

if [ -n "$MAX_DATA" ]; then
    CMD="$CMD --max_data $MAX_DATA"
fi

echo "Command: $CMD"
echo ""

# 执行 reject sampling
eval $CMD
SAMPLING_STATUS=$?

echo ""

# Step 4: 显示结果（清理会由 trap 自动处理）
echo "=================================================="
echo "Pipeline Completed!"
echo "=================================================="
echo ""

if [ $SAMPLING_STATUS -eq 0 ]; then
    echo "✅ Reject Sampling completed successfully!"
    echo ""
    echo "Results are available at:"
    echo "  Directory: $OUTPUT_DIR/$MODEL_NAME/"
    echo "  - reject_sampling_results.json (detailed results)"
    echo "  - error_distribution.png (histogram)"
    echo "  - statistics.txt (summary)"
    echo ""
    echo "Quick view of statistics:"
    echo "-----------------------------------------------"
    if [ -f "$OUTPUT_DIR/$MODEL_NAME/statistics.txt" ]; then
        head -n 20 "$OUTPUT_DIR/$MODEL_NAME/statistics.txt"
    fi
else
    echo "❌ Reject Sampling failed with exit code: $SAMPLING_STATUS"
    echo "   Check vLLM logs at: $LOG_FILE"
fi

echo ""
echo "=================================================="

# 退出（cleanup 函数会自动被 trap 调用）
exit $SAMPLING_STATUS

