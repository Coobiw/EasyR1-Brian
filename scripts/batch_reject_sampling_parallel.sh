#!/bin/bash
# 并行批量 Reject Sampling 脚本
# 同时运行2个任务，使用不同的GPU组和端口

set -e

# ==================== 配置部分 ====================

# 基础路径
BASE_PATH="/code/All-In-One/qbw/EasyR1-20250410/cache/output/agiqa3k_qual_n16_temp1_gaussian-default_format0p5_bs128-mbs64_kl0_chat-template_20251021_on-policy_val-temp1_promptdataset_merge-data"

# 检查点列表
CHECKPOINTS=(
    "global_step_37"
    "global_step_74"
    "global_step_111"
    "global_step_148"
    "global_step_185"
)

# 对应的模型名称
MODEL_NAMES=(
    "qwen_2_5_vl_mergedata_step37"
    "qwen_2_5_vl_mergedata_step74"
    "qwen_2_5_vl_mergedata_step111"
    "qwen_2_5_vl_mergedata_step148"
    "qwen_2_5_vl_mergedata_step185"
)

# 并行配置
# GPU组1: 卡0-3, 端口12000
GPU_GROUP_1="0,1,2,3"
PORT_1=12000

# GPU组2: 卡4-7, 端口12001
GPU_GROUP_2="4,5,6,7"
PORT_2=12001

# 公共配置
NUM_ROLLOUT=16
TEMPERATURE=1.0
CONCURRENCY=16
MAX_RETRIES=10
OUTPUT_DIR="./reject_sample_results"

# ==================================================

echo "=================================================="
echo "Parallel Batch Reject Sampling"
echo "=================================================="
echo ""
echo "Configuration:"
echo "  Base Path: $BASE_PATH"
echo "  Total Checkpoints: ${#CHECKPOINTS[@]}"
echo "  Parallel Jobs: 2"
echo ""
echo "GPU Group 1: $GPU_GROUP_1 (Port: $PORT_1)"
echo "GPU Group 2: $GPU_GROUP_2 (Port: $PORT_2)"
echo ""
echo "  Rollouts per sample: $NUM_ROLLOUT"
echo "  Temperature: $TEMPERATURE"
echo "  Concurrency: $CONCURRENCY"
echo ""
echo "=================================================="
echo ""

# 创建日志目录
LOG_DIR="./logs/batch_parallel"
mkdir -p "$LOG_DIR"

# 记录成功和失败的检查点
declare -A JOB_STATUS
SUCCESS_CKPTS=()
FAILED_CKPTS=()

# 运行单个任务的函数
run_single_task() {
    local CKPT=$1
    local MODEL_NAME=$2
    local GPU_GROUP=$3
    local PORT=$4
    local JOB_ID=$5
    
    local MODEL_PATH="$BASE_PATH/$CKPT/actor/huggingface"
    local LOG_FILE="$LOG_DIR/${MODEL_NAME}.log"
    
    echo "[Job $JOB_ID] Starting: $CKPT (GPU: $GPU_GROUP, Port: $PORT)" | tee -a "$LOG_FILE"
    
    # 检查检查点是否存在
    if [ ! -d "$MODEL_PATH" ]; then
        echo "[Job $JOB_ID] ⚠️  Checkpoint not found: $MODEL_PATH" | tee -a "$LOG_FILE"
        return 1
    fi
    
    # 设置环境变量并运行
    MODEL_PATH="$MODEL_PATH" \
    MODEL_NAME="$MODEL_NAME" \
    MODEL_PORT="$PORT" \
    CUDA_VISIBLE_DEVICES="$GPU_GROUP" \
    NUM_ROLLOUT="$NUM_ROLLOUT" \
    TEMPERATURE="$TEMPERATURE" \
    CONCURRENCY="$CONCURRENCY" \
    MAX_RETRIES="$MAX_RETRIES" \
    OUTPUT_DIR="$OUTPUT_DIR" \
    bash scripts/reject_sample.sh >> "$LOG_FILE" 2>&1
    
    local EXIT_CODE=$?
    
    if [ $EXIT_CODE -eq 0 ]; then
        echo "[Job $JOB_ID] ✅ Completed: $CKPT" | tee -a "$LOG_FILE"
        return 0
    else
        echo "[Job $JOB_ID] ❌ Failed: $CKPT (exit code: $EXIT_CODE)" | tee -a "$LOG_FILE"
        return 1
    fi
}

# 主循环：每次处理2个检查点
total_processed=0
for ((i=0; i<${#CHECKPOINTS[@]}; i+=2)); do
    echo ""
    echo "###################################################"
    echo "# Batch $((i/2 + 1)): Processing 2 checkpoints in parallel"
    echo "###################################################"
    echo ""
    
    # 第一个任务
    idx1=$i
    if [ $idx1 -lt ${#CHECKPOINTS[@]} ]; then
        CKPT1="${CHECKPOINTS[$idx1]}"
        MODEL_NAME1="${MODEL_NAMES[$idx1]}"
        
        echo "Starting Job 1: $CKPT1"
        echo "  GPU: $GPU_GROUP_1"
        echo "  Port: $PORT_1"
        echo "  Log: $LOG_DIR/${MODEL_NAME1}.log"
        
        # 后台运行任务1
        (
            if run_single_task "$CKPT1" "$MODEL_NAME1" "$GPU_GROUP_1" "$PORT_1" "1"; then
                echo "SUCCESS:$CKPT1" > "$LOG_DIR/status_${MODEL_NAME1}.txt"
            else
                echo "FAILED:$CKPT1" > "$LOG_DIR/status_${MODEL_NAME1}.txt"
            fi
        ) &
        PID1=$!
        echo "  Started with PID: $PID1"
    else
        PID1=""
    fi
    
    # 第二个任务
    idx2=$((i+1))
    if [ $idx2 -lt ${#CHECKPOINTS[@]} ]; then
        CKPT2="${CHECKPOINTS[$idx2]}"
        MODEL_NAME2="${MODEL_NAMES[$idx2]}"
        
        echo ""
        echo "Starting Job 2: $CKPT2"
        echo "  GPU: $GPU_GROUP_2"
        echo "  Port: $PORT_2"
        echo "  Log: $LOG_DIR/${MODEL_NAME2}.log"
        
        # 后台运行任务2
        (
            if run_single_task "$CKPT2" "$MODEL_NAME2" "$GPU_GROUP_2" "$PORT_2" "2"; then
                echo "SUCCESS:$CKPT2" > "$LOG_DIR/status_${MODEL_NAME2}.txt"
            else
                echo "FAILED:$CKPT2" > "$LOG_DIR/status_${MODEL_NAME2}.txt"
            fi
        ) &
        PID2=$!
        echo "  Started with PID: $PID2"
    else
        PID2=""
    fi
    
    echo ""
    echo "Waiting for both jobs to complete..."
    echo "You can monitor progress with:"
    if [ -n "$PID1" ]; then
        echo "  tail -f $LOG_DIR/${MODEL_NAME1}.log"
    fi
    if [ -n "$PID2" ]; then
        echo "  tail -f $LOG_DIR/${MODEL_NAME2}.log"
    fi
    echo ""
    
    # 等待两个任务完成
    if [ -n "$PID1" ]; then
        wait $PID1
        STATUS1=$(cat "$LOG_DIR/status_${MODEL_NAME1}.txt" 2>/dev/null || echo "FAILED:$CKPT1")
        if [[ $STATUS1 == SUCCESS:* ]]; then
            SUCCESS_CKPTS+=("$CKPT1")
        else
            FAILED_CKPTS+=("$CKPT1")
        fi
        total_processed=$((total_processed + 1))
    fi
    
    if [ -n "$PID2" ]; then
        wait $PID2
        STATUS2=$(cat "$LOG_DIR/status_${MODEL_NAME2}.txt" 2>/dev/null || echo "FAILED:$CKPT2")
        if [[ $STATUS2 == SUCCESS:* ]]; then
            SUCCESS_CKPTS+=("$CKPT2")
        else
            FAILED_CKPTS+=("$CKPT2")
        fi
        total_processed=$((total_processed + 1))
    fi
    
    echo ""
    echo "---------------------------------------------------"
    echo "Batch $((i/2 + 1)) completed"
    echo "Progress: $total_processed/${#CHECKPOINTS[@]} checkpoints"
    echo "---------------------------------------------------"
    echo ""
    
    # 清理两个端口
    echo "Cleaning up ports..."
    bash scripts/kill_vllm.sh $PORT_1 $PORT_2 || true
    
    # 等待一下再继续
    if [ $total_processed -lt ${#CHECKPOINTS[@]} ]; then
        echo "Waiting 15 seconds before next batch..."
        sleep 15
    fi
done

# 打印总结
echo ""
echo "=================================================="
echo "PARALLEL BATCH PROCESSING SUMMARY"
echo "=================================================="
echo ""
echo "Total checkpoints: ${#CHECKPOINTS[@]}"
echo "Successfully processed: ${#SUCCESS_CKPTS[@]}"
echo "Failed: ${#FAILED_CKPTS[@]}"
echo ""

if [ ${#SUCCESS_CKPTS[@]} -gt 0 ]; then
    echo "✅ Successful checkpoints:"
    for ckpt in "${SUCCESS_CKPTS[@]}"; do
        echo "   - $ckpt"
    done
    echo ""
fi

if [ ${#FAILED_CKPTS[@]} -gt 0 ]; then
    echo "❌ Failed checkpoints:"
    for ckpt in "${FAILED_CKPTS[@]}"; do
        echo "   - $ckpt"
    done
    echo ""
fi

echo "Results saved to: $OUTPUT_DIR"
echo "Logs saved to: $LOG_DIR"
echo ""
echo "=================================================="
echo "All done!"
echo "=================================================="

