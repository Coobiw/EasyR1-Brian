#!/usr/bin/env bash

# 可以同时启用三项“安全模式”，让脚本更健壮、更易发现问题
# -e (errexit)：任何命令只要返回非零状态（失败），脚本就会立即退出
# -u (nounset)：脚本里引用到未定义的变量就会报错并退出
# -o pipefail：默认情况下，管道 cmd1 | cmd2 | cmd3 的退出状态是最后一个命令 cmd3 的状态；即使 cmd1 或 cmd2 失败了，只要 cmd3 成功，整个管道也算成功
set -euo pipefail

# 检查脚本启动时传入的参数个数，确保你提供了恰好 5 个参数；如果不是，就给出使用提示并以错误状态退出
if [ $# -ne 6 ]; then
  echo "Usage: $0 INTERVAL START END MODEL_NAME BASE_DIR LOG_DIR"
  exit 1
fi

# 参数
INTERVAL=$1       # 步长，比如 20
START=$2          # 起始 global_step
END=$3            # 结束 global_step
MODEL_NAME=$4     # 保存的.jsonl文件名：模型/experiment的代表
BASE_DIR=$5       # 模型输出根目录
LOG_DIR=$6        # 日志目录

num_tasks=$(( (END - START) / INTERVAL + 1 ))
if (( num_tasks > 8 )); then
  echo "Error: 任务数量 ${num_tasks} 超出可用 GPU 数量 8" >&2
  exit 1
fi

this_cuda=0
mkdir -p "$LOG_DIR"

# 遍历所有 global_step_* 目录
for d in "$BASE_DIR"/global_step_*; do
  # 提取数字部分
  step=${d##*global_step_}

  # 只处理：能被 INTERVAL 整除，且 START <= step <= END
  if (( step % INTERVAL == 0 && step >= START && step <= END )); then
    ACTOR_DIR="$d/actor/huggingface"
    if (( this_cuda > 7 )); then
      echo "Error: this_cuda (${this_cuda}) 超出最大 GPU 索引 7" >&2
      exit 1
    fi
    if [ -d "$ACTOR_DIR" ]; then
      LOG_FILE="$LOG_DIR/step_${step}.log"
      echo "Launching evaluation on GPU $this_cuda: step $step → $ACTOR_DIR"
      nohup bash eval_scripts/agiqa3k_eval.sh \
        $this_cuda "$ACTOR_DIR" "${MODEL_NAME}_step${step}" \
        > "$LOG_FILE" 2>&1 &
      # 自增 GPU 索引，并检查不超过 7
      this_cuda=$(( this_cuda + 1 ))
    else
      echo "Warning: 目录 $ACTOR_DIR 不存在，跳过"
    fi
  fi
done

echo "所有符合条件（step ∈ [$START,$END] 且 % $INTERVAL == 0）的 eval 任务均已放入后台，日志保存在 $LOG_DIR。"
