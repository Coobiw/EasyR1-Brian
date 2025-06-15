#!/usr/bin/env bash

# 可以同时启用三项“安全模式”，让脚本更健壮、更易发现问题
# -e (errexit)：任何命令只要返回非零状态（失败），脚本就会立即退出
# -u (nounset)：脚本里引用到未定义的变量就会报错并退出
# -o pipefail：默认情况下，管道 cmd1 | cmd2 | cmd3 的退出状态是最后一个命令 cmd3 的状态；即使 cmd1 或 cmd2 失败了，只要 cmd3 成功，整个管道也算成功
set -euo pipefail

# 检查脚本启动时传入的参数个数：4 个
if [ $# -ne 4 ]; then
  echo "Usage: $0 INTERVAL START END MODEL_NAME"
  exit 1
fi

# 参数
INTERVAL=$1       # 步长，比如 20
START=$2          # 起始 global_step
END=$3            # 结束 global_step
MODEL_NAME=$4     # 输出文件名前缀

# 遍历从 START 到 END，步长为 INTERVAL 的整数序列
for (( step=START; step<=END; step+=INTERVAL )); do
  echo "Computing metrics for step $step ..."
  python eval_scripts/compute_metric.py \
    --out_name "${MODEL_NAME}_step${step}"
done

echo "所有符合条件（step ∈ [$START,$END] 且 step % $INTERVAL == 0）的 compute_metrics 任务均已完成。"
