from datasets import load_dataset, concatenate_datasets
from huggingface_hub import login
import os

# —— 1. 加载两个数据集的 train split —— #
ds1 = load_dataset("Coobiw/agiqa3k_finale_1013", split="train")
ds2 = load_dataset("Coobiw/aigciqa2023_qual", split="train")

print(f"Dataset 1 size: {len(ds1)}")
print(f"Dataset 2 size: {len(ds2)}")

# 打印两个数据集的 features 以便调试
print("\n" + "="*60)
print("ds1 features:")
print(ds1.features)
print("\n" + "="*60)
print("ds2 features:")
print(ds2.features)
print("="*60 + "\n")

# —— 1.5. 给 ds1 添加 source 字段 —— #
def add_source_ds1(example):
    example["source"] = "agiqa3k_train"
    return example

ds1 = ds1.map(add_source_ds1)
print(f"✅ 已给 ds1 添加 source 字段")

# —— 2. 修改 ds2 的字段 —— #
def modify_item(example):
    # 将 images 从 Sequence 转换为列表（确保与 ds1 格式一致）
    if "images" in example:
        example["images"] = list(example["images"])
    
    # 修改 problem 字段
    example["problem"] = \
"""
<image>What is your overall rating on the quality of this AI-generated picture? The rating should be a float between 1 and 5, rounded to two decimal places, with 1 representing very poor quality and 5 representing excellent quality. Return the final answer like: <answer> the score </answer>.
""".strip()
    
    # 将 answer 保留两位小数
    example["answer"] = round(float(example["answer"]), 2)
    
    # 添加 source 字段
    example["source"] = "aigcqa2023_train"
    
    return example

# 应用修改到 ds2
ds2 = ds2.map(modify_item)
print(f"✅ 已完成 ds2 的 images、problem、answer、source 字段修改")

# —— 2.5. 更新 ds2 的 features 定义以匹配 ds1 —— #
from datasets import Features, Image, Value
# 创建与 ds1 相同的 features 定义（包含 source 字段）
new_features = Features({
    'images': [Image()],
    'problem': Value('string'),
    'answer': Value('float64'),
    'source': Value('string')
})
ds2 = ds2.cast(new_features)
print(f"✅ 已更新 ds2 的 features 定义（包含 source 字段）")

# —— 3. 合并 —— #
merged = concatenate_datasets([ds1, ds2])

# —— 4. （可选）打乱 —— #
merged = merged.shuffle(seed=42)

print(f"Merged dataset size: {len(merged)}")

# —— 5. 先保存到本地 —— #
local_path = "./merged_dataset_cache"
print(f"\n保存数据集到本地: {local_path}")
merged.save_to_disk(local_path)
print(f"✅ 已保存到本地")

# —— 6. 从本地推送到 Hugging Face Hub —— #
from datasets import load_from_disk
print(f"\n从本地加载数据集并推送...")
merged_local = load_from_disk(local_path)
repo_id = "Coobiw/merged-agiqa5k-1022"
merged_local.push_to_hub(repo_id, private=False, token=True)

print(f"✅ 已成功推送合并后的数据集到 https://huggingface.co/datasets/{repo_id}")

# —— 7. 清理本地缓存（可选）—— #
import shutil
shutil.rmtree(local_path)
print(f"✅ 已清理本地缓存")
