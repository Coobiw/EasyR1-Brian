# Reject Sampling Dataset Filter

基于已有的reject sampling结果过滤数据集，只保留min_error <= 阈值的样本。

## 功能

1. 加载reject sampling结果（从`reject_sample_results`目录）
2. 使用`(problem, answer)`作为唯一标识匹配数据集样本
3. 过滤掉min_error > 阈值的样本
4. 保持原始数据集格式上传到HuggingFace
5. 显示详细的误差分布统计

## 使用方法

### 方法1: 使用Shell脚本（推荐）

```bash
# 使用默认配置
bash scripts/run_reject_dataset.sh

# 或自定义配置
DATASET_NAME=Coobiw/merged_agiqa5k_prompt_1022 \
RESULTS_DIR=./reject_sample_results \
MAX_ERROR=0.75 \
OUTPUT_REPO=Coobiw/agiqa3k_prompt_rejected_1025 \
bash scripts/run_reject_dataset.sh
```

### 方法2: 直接运行Python脚本

```bash
python scripts/reject_sample_dataset.py \
    --dataset_name Coobiw/merged_agiqa5k_prompt_1022 \
    --results_dir ./reject_sample_results \
    --max_error 0.75 \
    --output_repo Coobiw/agiqa3k_prompt_rejected_1025
```

## 参数说明

### 必需参数

无（所有参数都有默认值）

### 可选参数

- `--dataset_name`: 源数据集名称（默认：`Coobiw/merged_agiqa5k_prompt_1022`）
- `--results_dir`: reject sampling结果目录（默认：`./reject_sample_results`）
- `--max_error`: 最大允许误差阈值（默认：`0.75`）
- `--output_repo`: 输出数据集仓库（默认：`Coobiw/agiqa3k_prompt_rejected_1025`）
- `--local_cache`: 本地缓存目录（默认：`./filtered_dataset_cache`）
- `--private`: 设置为私有数据集
- `--keep_cache`: 保留本地缓存

## 示例

### 示例1: 使用默认配置

```bash
bash scripts/run_reject_dataset.sh
```

### 示例2: 自定义误差阈值

```bash
MAX_ERROR=0.5 bash scripts/run_reject_dataset.sh
```

### 示例3: 保留本地缓存

```bash
python scripts/reject_sample_dataset.py \
    --dataset_name Coobiw/merged_agiqa5k_prompt_1022 \
    --results_dir ./reject_sample_results \
    --max_error 0.75 \
    --keep_cache
```

### 示例4: 使用不同的结果目录

```bash
RESULTS_DIR=./my_custom_results \
OUTPUT_REPO=Coobiw/my_filtered_dataset \
bash scripts/run_reject_dataset.sh
```

## 输出

### 过滤结果统计

```
================================================================================
Filtering Results:
  Original samples: 4785
  Kept (error <= 0.75): 4520 (94.46%)
  Discarded (error > 0.75): 265 (5.54%)
  No reject sampling result: 0 (0.00%)
  Final dataset size: 4520
================================================================================
```

### 误差分布

```
================================================================================
Error Distribution (from matched samples):
================================================================================
  [0, 0.25)        3508 (73.31%)
  [0.25, 0.5)       769 (16.07%)
  [0.5, 0.75)       243 ( 5.08%)
  [0.75, 1.0)       162 ( 3.39%)
  [1.0, 1.5)         88 ( 1.84%)
  [1.5, 2.0)         13 ( 0.27%)
  [2.0, inf)          2 ( 0.04%)
================================================================================
```

### HuggingFace数据集

过滤后的数据集会上传到指定的HuggingFace仓库，格式与原始数据集完全一致。

## 匹配逻辑

脚本使用 `(problem, answer)` 元组作为唯一标识：
- `problem`: 数据集中的problem字段（去除首尾空格）
- `answer`: 数据集中的answer字段（MOS分数）

这样可以准确匹配每个样本，即使problem相同但answer（MOS分数）不同。

## 前置条件

1. **Reject Sampling结果**: 需要先运行`reject_sample.py`生成结果
2. **结果目录结构**:
   ```
   reject_sample_results/
   ├── checkpoint1/
   │   └── reject_sampling_results.json
   ├── checkpoint2/
   │   └── reject_sampling_results.json
   └── ...
   ```
3. **HuggingFace认证**: 确保已登录 (`huggingface-cli login`)

## 注意事项

1. **数据集匹配**: 确保reject sampling使用的数据集与要过滤的数据集一致
2. **唯一标识**: 使用`(problem, answer)`作为key，确保匹配准确性
3. **多个结果**: 如果同一个key有多个结果，取最小的min_error
4. **磁盘空间**: 本地缓存需要足够的磁盘空间
5. **HuggingFace Token**: 需要有写权限的token

## 错误排查

### 找不到reject sampling结果

检查：
- 结果目录路径是否正确
- 子目录中是否有`reject_sampling_results.json`文件
- JSON文件格式是否正确

### 匹配率很低

可能原因：
- reject sampling使用的数据集与当前数据集不一致
- problem或answer字段格式不匹配
- JSON文件中缺少必要字段

### 数据集上传失败

数据集会保存在本地缓存目录，可以手动上传：

```python
from datasets import load_from_disk
ds = load_from_disk("./filtered_dataset_cache")
ds.push_to_hub("your_repo_name", token="your_token")
```

## 工作流程

```
1. 加载reject sampling结果
   - 遍历results_dir中的所有子目录
   - 读取reject_sampling_results.json
   - 构建 {(problem, answer): min_error} 映射
   ↓
2. 加载源数据集
   ↓
3. 对每个样本:
   - 提取 (problem, answer)
   - 在映射中查找min_error
   - 判断是否保留 (min_error <= 阈值)
   ↓
4. 统计误差分布
   ↓
5. 创建过滤后的数据集
   ↓
6. 保存到本地缓存
   ↓
7. 推送到HuggingFace
   ↓
8. 清理资源（可选）
```

## 与完整Reject Sampling的区别

### 本脚本（基于已有结果）
- ✅ 快速（秒级完成）
- ✅ 不需要GPU
- ✅ 不需要启动vLLM
- ✅ 复用已有结果
- ❌ 需要先运行reject sampling

### 完整Reject Sampling（实时rollout）
- ❌ 慢（小时级）
- ❌ 需要GPU资源
- ❌ 需要启动vLLM服务
- ✅ 不依赖已有结果
- ✅ 可以用于新数据集

## 相关文件

- `reject_sample_dataset.py`: 主Python脚本（基于已有结果）
- `run_reject_dataset.sh`: Shell包装脚本
- `reject_sample.py`: 生成reject sampling结果的脚本
- `plot_reject_sampling_boxplot.py`: 可视化reject sampling结果
