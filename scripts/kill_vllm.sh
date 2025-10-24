#!/bin/bash
# 清理残留的 vLLM 服务进程
#
# 使用方法：
#   bash scripts/kill_vllm.sh [port1] [port2] ...
#
# 示例：
#   bash scripts/kill_vllm.sh              # 清理默认端口 8000
#   bash scripts/kill_vllm.sh 8000         # 清理指定端口
#   bash scripts/kill_vllm.sh 8000 8001    # 清理多个端口
#   MODEL_PORT=8080 bash scripts/kill_vllm.sh  # 使用环境变量

# 获取端口列表
if [ $# -gt 0 ]; then
    # 使用命令行参数
    PORTS=("$@")
elif [ -n "$MODEL_PORT" ]; then
    # 使用环境变量
    PORTS=("$MODEL_PORT")
else
    # 默认端口
    PORTS=(8000)
fi

echo "=================================================="
echo "Cleaning up vLLM processes"
echo "=================================================="
echo ""

# 查找所有 vLLM 相关进程
echo "Searching for vLLM processes..."
VLLM_PIDS=$(ps aux | grep -E "vllm|vllm_serve" | grep -v grep | awk '{print $2}')

if [ -z "$VLLM_PIDS" ]; then
    echo "✅ No vLLM processes found."
else
    echo "Found vLLM processes:"
    ps aux | grep -E "vllm|vllm_serve" | grep -v grep
    echo ""
    
    echo "Killing processes..."
    for PID in $VLLM_PIDS; do
        echo "  Killing PID: $PID"
        kill -9 $PID 2>/dev/null || true
    done
    
    echo ""
    echo "✅ All vLLM processes killed."
fi

# 检查并清理端口占用
echo ""
echo "Checking ports: ${PORTS[@]}..."

for PORT in "${PORTS[@]}"; do
    echo ""
    echo "Checking port $PORT..."
    PORT_PID=$(lsof -ti:$PORT 2>/dev/null)
    
    if [ -z "$PORT_PID" ]; then
        echo "  ✅ Port $PORT is free."
    else
        echo "  Port $PORT is occupied by PID: $PORT_PID"
        echo "  Killing process on port $PORT..."
        kill -9 $PORT_PID 2>/dev/null || true
        
        # 再次检查
        sleep 1
        PORT_PID=$(lsof -ti:$PORT 2>/dev/null)
        if [ -z "$PORT_PID" ]; then
            echo "  ✅ Port $PORT freed."
        else
            echo "  ⚠️  Port $PORT still occupied, may need manual cleanup."
        fi
    fi
done

echo ""
echo "=================================================="
echo "Cleanup completed!"
echo "=================================================="

