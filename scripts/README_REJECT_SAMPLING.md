# Reject Sampling 使用指南

## 🎯 功能

对每条样本进行 N 次 rollout，取与 ground truth 差值最小的作为误差，统计误差分布。

## 🚀 快速开始

```bash
# 方式1: 使用脚本（推荐）
bash scripts/reject_sample.sh

# 方式2: 直接运行 Python
python scripts/reject_sample.py \
    --model_path /path/to/model \
    --model_name my_model \
    --num_rollout 16
```

## 📋 主要参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--model_path` | **必需** | 模型路径 |
| `--model_name` | **必需** | 模型名称 |
| `--num_rollout` | 16 | 每条样本 rollout 次数 |
| `--temperature` | 1.0 | 采样温度 |
| `--max_retries` | 10 | 失败重试次数 |
| `--concurrency` | 8 | 样本级并发数 (同时处理多少个样本) |
| `--skip_vllm_launch` | False | 跳过 vLLM 启动（服务已运行时用） |

## 💡 使用示例

### 测试运行（10条数据）
```bash
python scripts/reject_sample.py \
    --model_path /path/to/model \
    --model_name test \
    --num_rollout 4 \
    --max_data 10
```

### 使用环境变量配置
```bash
export MODEL_PATH=/path/to/model
export MODEL_NAME=my_model
export NUM_ROLLOUT=32
export MAX_DATA=100
bash scripts/reject_sample.sh
```

### 批量处理多个检查点
```bash
for step in 18 36 54 72 90; do
    python scripts/reject_sample.py \
        --model_path /path/to/model/global_step_${step}/actor/huggingface \
        --model_name model_step${step} \
        --num_rollout 16
done
```

## 📊 输出结果

在 `reject_sampling_results/<model_name>/` 生成：

- **reject_sampling_results.json** - 详细结果
- **error_distribution.png** - 误差分布直方图
- **statistics.txt** - 统计摘要

## 🔧 关键特性

- ✅ **两层并发机制**：样本级并发 + rollout 级并发
  - 同时处理 N 个样本（`--concurrency`）
  - 每个样本同时进行 M 次 rollout（`--num_rollout`）
  - 最大并发请求数：N × M
- ✅ 自动管理 vLLM 服务（启动/停止）
- ✅ 失败自动重试（指数退避：1s, 2s, 4s...）
- ✅ 实时进度显示（批次、成功率、平均误差）
- ✅ 完整的日志记录

### 并发性能示例

```bash
# 示例配置
--concurrency 8      # 同时处理 8 个样本
--num_rollout 16     # 每个样本 16 次 rollout

# 实际效果
- 样本级并发：8 个样本同时处理
- Rollout 级并发：每个样本内 16 次 rollout 并发
- 最大并发请求：8 × 16 = 128 个请求

# 性能提升
- 旧版本：串行处理样本，只在 rollout 维度并发 → 慢
- 新版本：两层并发 → 快 8 倍（取决于 concurrency）
```

## 📝 数据集

默认使用 `Coobiw/merged_agiqa5k_prompt_1022` (train split)

需要 HuggingFace 访问，可设置镜像：
```bash
export HF_ENDPOINT=https://hf-mirror.com
```

## ⚙️ 并发参数调优

根据你的硬件配置调整并发参数：

```bash
# GPU 显存充足（如 A100 80GB）
--concurrency 16 --num_rollout 16  # 256 并发请求

# GPU 显存中等（如 A100 40GB）
--concurrency 8 --num_rollout 16   # 128 并发请求

# GPU 显存较小（如 V100 32GB）
--concurrency 4 --num_rollout 16   # 64 并发请求

# 如果出现 OOM，降低 concurrency
--concurrency 2 --num_rollout 16   # 32 并发请求
```

**监控建议**：
```bash
# 观察 GPU 利用率
watch -n 1 nvidia-smi

# 如果 GPU 利用率 < 80%，可以增加 concurrency
# 如果出现 OOM，减少 concurrency
```

## 🐛 故障排查

**vLLM 启动超时**: 检查日志 `logs/<model_name>_vllm.log`

**大量 rollout 失败**: 检查模型输出格式（需要 `<answer>` 标签）

**OOM 错误**: 降低 `--concurrency` 参数

**端口占用**: 更改 `--model_port` 或手动清理 `lsof -ti:8000 | xargs kill -9`

## 🆕 更新说明

**v2.1 - 并发优化**
- ⚡ **两层并发**：样本级 + rollout 级，性能提升数倍
- 📊 改进进度显示：批次信息、平均误差、成功率

**v2.0 - 重大更新**
- 参数名称：`num_samples` → `num_rollout`
- 新增：`--max_retries` 失败重试机制
- 新增：`--model_path` 必需参数
- 集成：自动 vLLM 服务管理

