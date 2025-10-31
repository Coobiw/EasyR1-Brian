# Dynamic Sampling (DAPO) 实现说明

## 概述

本实现参考DAPO论文中的dynamic sampling策略，在训练过程中**过滤掉所有rollout得分完全相同(std=0)的prompt**，并**自动累积多个generation batch**直到有足够的有效样本。

## 核心机制

### 1. 过滤标准

对每个prompt的n个rollout样本：
- 计算`token_level_scores`的序列和（sum over tokens）
- 计算这n个得分的标准差（std）
- **如果std=0**：说明所有rollout完全相同（全对或全错），**过滤掉该prompt**
- **如果std>0**：说明rollout有差异，**保留该prompt**

### 2. 批次累积机制

这是DAPO的关键特性：

```python
while num_accumulated_prompts < target_prompt_num:
    # 1. 生成一个batch
    new_batch = generate_and_compute_reward(batch_dict)
    
    # 2. 过滤std=0的prompts
    filtered_batch = filter_by_score_std(new_batch)
    
    # 3. 累积到accumulated_batch
    accumulated_batch = concat(accumulated_batch, filtered_batch)
    num_accumulated_prompts += num_kept_prompts
    
    # 4. 检查是否足够
    if num_accumulated_prompts >= target_prompt_num:
        batch = accumulated_batch[:target_traj_num]  # 截取
        break
    else:
        # 继续生成下一个batch
        batch_dict = next(dataloader)
        continue
```

**关键点**：
- 如果一个batch过滤后不够target_prompt_num，会自动从dataloader取下一个batch
- 继续生成、过滤、累积，直到有足够的prompts
- 最后截取前`rollout_batch_size * n`个trajectories用于训练

## 配置方式

```yaml
algorithm:
  use_dynamic_sample: true           # 是否启用（默认false）
  dynamic_sample_max_gen_batches: 3  # 最大累积批次数（0=无限制）
```

### 参数说明

- **`use_dynamic_sample`** (bool, default: `false`)
  - 是否启用dynamic sampling
  - `false`: 不过滤，不累积，行为与原代码完全一致

- **`dynamic_sample_max_gen_batches`** (int, default: `3`)
  - 最大累积的generation batch数量
  - `0`: 不限制，会一直累积直到dataloader耗尽
  - `> 0`: 累积到此数量后，即使不够target_prompt_num也停止
  - 推荐: `3-5`，防止在难数据集上卡太久

## 工作流程

```
Step N:
├─ Gen Batch 1 (128 prompts)
│  ├─ Generate 128*16 trajectories
│  ├─ Compute rewards
│  ├─ Filter: 45 prompts kept (std>0), 83 prompts filtered (std=0)
│  └─ Accumulated: 45/128 prompts ❌ 不够
│
├─ Gen Batch 2 (128 prompts)
│  ├─ Generate 128*16 trajectories  
│  ├─ Compute rewards
│  ├─ Filter: 62 prompts kept, 66 prompts filtered
│  └─ Accumulated: 107/128 prompts ❌ 仍不够
│
├─ Gen Batch 3 (128 prompts)
│  ├─ Generate 128*16 trajectories
│  ├─ Compute rewards
│  ├─ Filter: 50 prompts kept, 78 prompts filtered
│  └─ Accumulated: 157/128 prompts ✅ 够了！
│
└─ Truncate to 128*16=2048 trajectories
   ├─ Compute advantages
   ├─ Update actor & critic
   └─ Log metrics
```

## 输出日志示例

```
[Dynamic Sampling] Step 42, Gen batch 1: Kept 45/128 prompts (filtered 1328 trajectories), Accumulated: 45/128
[Dynamic Sampling] Need more prompts, fetching next batch...
[Dynamic Sampling] Step 42, Gen batch 2: Kept 62/128 prompts (filtered 1056 trajectories), Accumulated: 107/128
[Dynamic Sampling] Need more prompts, fetching next batch...
[Dynamic Sampling] Step 42, Gen batch 3: Kept 50/128 prompts (filtered 1248 trajectories), Accumulated: 157/128
[Dynamic Sampling] Accumulated enough prompts, proceeding with 2048 trajectories
```

## Metrics记录

训练过程中会记录以下metrics：
- `dynamic_sampling/num_gen_batches`: 累积了多少个generation batch
- `dynamic_sampling/num_accumulated_prompts`: 累积了多少个有效prompts
- `dynamic_sampling/target_prompt_num`: 目标prompt数量（通常等于rollout_batch_size）

## 与原实现的对比

### 参考代码（DAPO）特点

```python
# 参考代码的结构
metric_name = "seq_reward"  # 使用token_level_scores
new_batch.non_tensor_batch["seq_reward"] = scores.sum(dim=-1)

# 过滤
for uid, std in prompt_uid2metric_std.items():
    if std > 0 or len(metric_vals) == 1:
        keep_prompt(uid)

# 累积
batch = concat([batch, new_batch])
if num_prompt_in_batch < prompt_bsz:
    continue  # 继续生成
else:
    batch = batch[:traj_bsz]  # 截取
```

### 本实现特点

- ✅ 使用`token_level_scores`（与参考代码一致）
- ✅ 计算std并过滤std=0的prompts（与参考代码一致）
- ✅ 多批次累积机制（与参考代码一致）
- ✅ 截取到精确batch size（与参考代码一致）
- ✅ 配置化的max_gen_batches控制

## 适用场景

Dynamic sampling特别适合：

1. **高variance任务**：
   - 同一个prompt的不同rollout容易产生完全相同的结果
   - 例如：分类任务、简单QA任务

2. **训练后期**：
   - 模型已经收敛，部分prompt的所有rollout都给出相同质量答案
   - 过滤这些"无信息"样本可以提高训练效率

3. **大n值训练**：
   - `rollout.n >= 16`时，更容易出现全对/全错情况
   - Dynamic sampling可以有效过滤这些情况

## 注意事项

### 1. 训练时间

启用dynamic sampling后：
- ✅ 好处：过滤无信息样本，提高样本质量
- ⚠️ 代价：需要生成更多batch来累积够样本数
- 建议：设置合理的`max_gen_batches`（如3-5）防止卡太久

### 2. 与rollout.n配合

- `rollout.n`必须 > 1（无法计算std）
- 推荐`rollout.n >= 8`才能观察到明显效果
- `n`越大，过滤比例可能越高（更容易全对/全错）

### 3. max_gen_batches设置

```yaml
# 保守设置（推荐）
dynamic_sample_max_gen_batches: 3  # 最多累积3个batch

# 激进设置
dynamic_sample_max_gen_batches: 0  # 不限制，直到dataloader耗尽

# 中等设置
dynamic_sample_max_gen_batches: 5  # 平衡效果和速度
```

### 4. 边界情况

- 如果累积到`max_gen_batches`仍不够prompt数：使用现有的
- 如果dataloader耗尽：使用现有的
- 如果所有prompts都被过滤（极少见）：会警告但继续训练

## 使用示例

### 示例1：标准配置

```yaml
data:
  rollout_batch_size: 128

algorithm:
  adv_estimator: grpo
  use_dynamic_sample: true
  dynamic_sample_max_gen_batches: 3

worker:
  rollout:
    n: 16
```

### 示例2：激进过滤

```yaml
data:
  rollout_batch_size: 64  # 较小batch

algorithm:
  use_dynamic_sample: true
  dynamic_sample_max_gen_batches: 0  # 无限制

worker:
  rollout:
    n: 32  # 大n值
```

### 示例3：禁用（默认）

```yaml
algorithm:
  use_dynamic_sample: false  # 或不设置
  # 其他参数会被忽略
```

## 实现细节

### 代码位置

- **配置**: `verl/trainer/config.py` - `AlgorithmConfig`
- **过滤函数**: `verl/trainer/ray_trainer.py` - `_filter_batch_by_score_std()`
- **累积逻辑**: `verl/trainer/ray_trainer.py` - `fit()` 方法中的accumulation loop

### 过滤时机

```
生成 → 计算reward → 【过滤&累积】 → balance_batch → 计算log_probs → 
计算values → 计算advantage → 更新模型
```

关键：在reward计算后**立即**过滤，不等到advantage计算

### 为什么用token_level_scores？

1. **时机早**：reward计算后就有，不需要等KL penalty或advantage
2. **原始信号**：反映模型对样本的原始评分，未被KL修改
3. **与DAPO一致**：参考实现中使用`seq_reward`就是token_level_scores的sum

## FAQ

**Q: 为什么要累积多个batch？**
A: 因为过滤后可能不够一个完整的batch size，继续生成可以保证batch size稳定。

**Q: 会不会很慢？**
A: 取决于过滤比例。如果过滤比例高（如>50%），会需要生成多个batch，训练会变慢。建议设置max_gen_batches=3-5来限制。

**Q: 和keep_neg_ratio有什么区别？**
A: 
- `keep_neg_ratio`: 过滤advantage为负的样本（基于排序）
- `use_dynamic_sample`: 过滤std=0的prompt（基于variance）
- 两者可以同时使用

**Q: 什么时候会停止累积？**
A:
1. 累积的prompt数 >= target_prompt_num
2. 达到max_gen_batches限制
3. dataloader耗尽

**Q: 会影响训练稳定性吗？**
A: 一般不会。batch size在最后会被截取到精确值，保证每个step的batch size一致。

## 参考资料

- DAPO论文：[链接]
- 参考实现：`verl.trainer.ppo.ray_trainer.RayDAPOTrainer`
- 示例配置：`examples/config_agiqa3k_dynamic_sampling.yaml`

