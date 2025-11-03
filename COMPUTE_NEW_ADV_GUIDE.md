# compute_new_adv 使用指南

## 概述

`compute_new_adv` 是 GRPO 的一个新功能，用于在样本过滤后重新计算 advantages。当使用 `keep_pos_ratio` 和 `keep_neg_ratio` 过滤样本后，可以选择用保留的样本重新计算归一化的 advantages，而不是直接使用原始的 advantages。

## 功能说明

### 标准 GRPO 流程
1. 计算所有样本的 GRPO advantages（使用所有样本的均值和标准差归一化）
2. 根据 `keep_pos_ratio` 和 `keep_neg_ratio` 过滤样本
3. 使用原始 advantages 进行训练

### 启用 compute_new_adv 后的流程
1. 计算所有样本的 GRPO advantages（使用所有样本的均值和标准差归一化）
2. 根据 `keep_pos_ratio` 和 `keep_neg_ratio` 过滤样本
3. **重新计算保留样本的 advantages**（使用保留样本的均值和标准差重新归一化）
4. 使用重新计算的 advantages 进行训练

## 为什么需要这个功能？

### 动机
当我们过滤掉一些样本后，保留样本的分布可能会发生变化。例如：

- **原始样本**：rewards = [0.1, 0.5, 0.9, 1.3, 1.7]
  - mean = 0.9, std = 0.6
  - advantages = [-1.33, -0.67, 0, 0.67, 1.33]

- **过滤后**（假设只保留最好的3个）：rewards = [0.9, 1.3, 1.7]
  - 使用原始 advantages：[0, 0.67, 1.33]
  - 使用重新计算的 advantages：[-1, 0, 1]（mean=1.3, std=0.4）

### 优势
1. **更强的训练信号**：保留样本之间的 advantage 差异更明显
2. **更好的归一化**：advantages 更符合标准正态分布
3. **更集中的学习**：模型更专注于保留样本之间的相对差异

## 配置参数

在 `algorithm` 配置中添加：

```yaml
algorithm:
  adv_estimator: grpo
  keep_neg_ratio: 0.5      # 保留最差的 50% 负样本
  keep_pos_ratio: 0.7      # 保留最好的 70% 正样本
  compute_new_adv: true    # 启用重新计算 advantages
```

### 参数说明

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `compute_new_adv` | bool | `false` | 是否用保留样本重新计算 advantages |
| `keep_pos_ratio` | float | `1.0` | 正样本保留比例（0.0-1.0） |
| `keep_neg_ratio` | float | `1.0` | 负样本保留比例（0.0-1.0） |

## 使用示例

### 示例 1：基础用法
```yaml
algorithm:
  adv_estimator: grpo
  keep_neg_ratio: 0.5
  keep_pos_ratio: 0.7
  compute_new_adv: true
```

这会：
1. 保留最差的 50% 负样本（advantage 最低）
2. 保留最好的 70% 正样本（advantage 最高）
3. 用这些保留的样本重新计算 advantages

### 示例 2：命令行配置
```bash
python3 -m verl.trainer.main \
    config=examples/config_agiqa3k.yaml \
    algorithm.keep_pos_ratio=0.7 \
    algorithm.keep_neg_ratio=0.5 \
    algorithm.compute_new_adv=true
```

### 示例 3：只过滤负样本
```yaml
algorithm:
  adv_estimator: grpo
  keep_neg_ratio: 0.3      # 只保留最差的 30% 负样本
  keep_pos_ratio: 1.0      # 保留所有正样本
  compute_new_adv: true    # 重新计算
```

### 示例 4：只过滤正样本
```yaml
algorithm:
  adv_estimator: grpo
  keep_neg_ratio: 1.0      # 保留所有负样本
  keep_pos_ratio: 0.5      # 只保留最好的 50% 正样本
  compute_new_adv: true    # 重新计算
```

## 注意事项

### 1. 仅在有过滤时有效
如果 `keep_neg_ratio=1.0` 且 `keep_pos_ratio=1.0`（没有过滤任何样本），设置 `compute_new_adv=true` 不会有任何效果，系统会给出警告。

### 2. 最小样本数要求
每个 group 中至少需要保留 2 个样本才能重新计算标准差。如果保留样本少于 2 个，会使用原始 advantages。

### 3. 算法限制
`compute_new_adv` 仅支持 GRPO 算法，不支持其他算法（GAE, RLOO, REINFORCE++, REMAX）。

### 4. 性能考虑
重新计算 advantages 会增加少量计算开销（通常可忽略），但可能带来更好的训练效果。

## 实现细节

### 重新计算逻辑

对于每个 group（同一个 prompt 的多个 responses）：

1. **找出保留的样本**：根据 `keep_pos_ratio` 和 `keep_neg_ratio` 确定
2. **重新计算统计量**：
   ```python
   kept_scores = [score for kept samples]
   kept_mean = mean(kept_scores)
   kept_std = std(kept_scores)
   ```
3. **重新归一化**：
   ```python
   new_advantage = (score - kept_mean) / kept_std
   ```

### 特殊情况处理

- **所有保留样本得分相同**（std ≈ 0）：advantage 设为 0
- **保留样本少于 2 个**：使用原始 advantages（不重新计算）
- **无过滤**（keep_pos_ratio=1.0 且 keep_neg_ratio=1.0）：直接返回原始 advantages

## Metrics 变化

启用 `compute_new_adv` 后，metrics 中的含义：

| Metric | 含义 |
|--------|------|
| `critic/advantages/*` | 原始 GRPO advantages（用于与其他实验对比） |
| `critic/advantages_processed/*` | 实际用于训练的 advantages（重新计算后的） |
| `critic/advantages_processed/std` | 保留样本的 advantage 标准差 |

注意：
- `critic/advantages_processed/*` 统计时会排除被过滤掉的样本（advantage=0）
- 如果 `compute_new_adv=false`，`advantages_processed` 就是原始 advantages（但已过滤）

## 推荐设置

### 保守设置（推荐初次使用）
```yaml
algorithm:
  keep_neg_ratio: 0.7
  keep_pos_ratio: 0.8
  compute_new_adv: true
```

### 激进设置（更强的样本选择）
```yaml
algorithm:
  keep_neg_ratio: 0.3
  keep_pos_ratio: 0.5
  compute_new_adv: true
```

### 对比实验
建议进行 A/B 测试：
- **Baseline**: `compute_new_adv: false`（使用原始 advantages）
- **Treatment**: `compute_new_adv: true`（重新计算 advantages）

## 与其他功能的组合

### 与 KL 惩罚结合
```yaml
algorithm:
  adv_estimator: grpo
  keep_neg_ratio: 0.5
  keep_pos_ratio: 0.7
  compute_new_adv: true
  disable_kl: false
  kl_coef: 1.0e-2
```

### 与 DAPO 结合
```yaml
algorithm:
  adv_estimator: grpo
  keep_neg_ratio: 0.5
  keep_pos_ratio: 0.7
  compute_new_adv: true

worker:
  actor:
    clip_ratio_low: 0.2
    clip_ratio_high: 0.28
    clip_ratio_dual: 10.0
```

## FAQ

### Q1: compute_new_adv 会影响收敛速度吗？
A: 可能会。重新计算后的 advantages 通常更极端（方差更大），可能加快收敛，但也可能导致不稳定。建议结合较小的学习率使用。

### Q2: 应该先调整过滤比例还是先启用 compute_new_adv？
A: 建议先调整过滤比例（`keep_pos_ratio` 和 `keep_neg_ratio`），找到合适的过滤策略后，再尝试启用 `compute_new_adv`。

### Q3: 可以只启用 compute_new_adv 而不过滤样本吗？
A: 技术上可以，但没有意义。如果不过滤样本，重新计算的 advantages 和原始 advantages 完全相同。

### Q4: compute_new_adv 和原始 advantages 有多大差异？
A: 差异取决于过滤强度。过滤越强（keep_ratio 越小），差异越大。可以通过 wandb 的 `critic/advantages/*` 和 `critic/advantages_processed/*` 对比观察。

### Q5: 出现 "Not enough samples to compute std" 警告怎么办？
A: 说明某些 group 保留的样本太少（< 2）。可以：
- 增加 `keep_pos_ratio` 和 `keep_neg_ratio`
- 增加 `rollout.n`（每个 prompt 生成更多 responses）

## 完整配置示例

```yaml
data:
  train_files: your/dataset
  rollout_batch_size: 128
  max_prompt_length: 2048
  max_response_length: 2048

algorithm:
  adv_estimator: grpo
  gamma: 1.0
  lam: 1.0
  disable_kl: false
  kl_coef: 1.0e-2
  keep_neg_ratio: 0.5      # 保留最差的 50% 负样本
  keep_pos_ratio: 0.7      # 保留最好的 70% 正样本
  compute_new_adv: true    # 🆕 重新计算 advantages

worker:
  actor:
    global_batch_size: 64
    micro_batch_size_per_device_for_update: 8
    model:
      model_path: Qwen/Qwen2.5-7B-Instruct
    optim:
      lr: 1.0e-6
  
  rollout:
    n: 16                  # 每个 prompt 生成 16 个 responses
    temperature: 1.0

trainer:
  total_episodes: 10
  logger: ["console", "wandb"]
  project_name: grpo-compute-new-adv
  experiment_name: test
```

## 版本历史

- **v1.0** (2025-01): 初始实现
  - 支持基于保留样本重新计算 GRPO advantages
  - 添加 `compute_new_adv` 配置参数
  - 添加相关验证和警告

## 参考资料

- GRPO 论文: [Group Relative Policy Optimization](https://arxiv.org/abs/2402.03300)
- 相关代码: `verl/trainer/core_algos.py::compute_grpo_outcome_advantage`

