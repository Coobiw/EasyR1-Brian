"""
基于 reject sampling 结果过滤数据集
使用 (problem, answer) 作为唯一标识
保留 min_error <= 0.75 的样本，推送到 HuggingFace
"""
import json
import argparse
from pathlib import Path
from datasets import load_dataset, Dataset
import shutil
import numpy as np

def load_reject_sampling_results(results_dir, checkpoint_name="qwen2_5_vl_7b_baseline"):
    """
    从指定checkpoint的reject sampling结果中加载数据
    返回: dict {(problem, answer): min_error}
    """
    results_dir = Path(results_dir)
    checkpoint_dir = results_dir / checkpoint_name
    result_file = checkpoint_dir / "reject_sampling_results.json"
    
    print(f"Loading reject sampling results from: {checkpoint_dir}")
    
    if not result_file.exists():
        print(f"❌ Result file not found: {result_file}")
        return {}
    
    with open(result_file, 'r') as f:
        data = json.load(f)
    
    print(f"  Loaded {len(data)} samples from {checkpoint_name}")
    
    # 调试：打印第一个item的keys
    if data:
        print(f"  Sample keys: {list(data[0].keys())}")
    
    # 构建映射
    all_results = {}
    for item in data:
        problem = item.get("problem", "")
        # JSON中使用ground_truth字段
        ground_truth = item.get("ground_truth")
        min_error = item.get("min_error", float('inf'))
        
        if problem and ground_truth is not None:
            # 使用 (problem, ground_truth) 作为唯一标识
            key = (problem.strip(), float(ground_truth))
            all_results[key] = min_error
    
    print(f"  Total samples with reject sampling results: {len(all_results)}")
    return all_results

def filter_dataset(dataset_name, reject_results, max_error=0.75):
    """
    过滤数据集，只保留 min_error <= max_error 的样本
    
    Args:
        dataset_name: HuggingFace 数据集名称
        reject_results: dict {(problem, answer): min_error}
        max_error: 最大允许误差阈值
    """
    print(f"\n{'='*80}")
    print(f"Loading dataset: {dataset_name}")
    print(f"{'='*80}")
    
    # 加载原始数据集
    ds = load_dataset(dataset_name, split="train")
    print(f"Original dataset size: {len(ds)}")
    
    # 打印数据集特征
    print(f"\nDataset features:")
    print(ds.features)
    
    # 过滤数据
    print(f"\n{'='*80}")
    print(f"Filtering samples with min_error <= {max_error}")
    print(f"{'='*80}")
    
    filtered_data = []
    all_errors = []  # 记录所有找到的误差
    kept_count = 0
    discarded_count = 0
    no_result_count = 0
    
    for idx, item in enumerate(ds):
        problem = item.get("problem", "").strip()
        answer = item.get("answer")
        
        if not problem or answer is None:
            no_result_count += 1
            continue
        
        # 构建key
        key = (problem, float(answer))
        
        # 检查是否有 reject sampling 结果
        if key not in reject_results:
            no_result_count += 1
            continue
        
        min_error = reject_results[key]
        all_errors.append(min_error)
        
        if min_error <= max_error:
            filtered_data.append(item)
            kept_count += 1
        else:
            discarded_count += 1
        
        if (idx + 1) % 1000 == 0:
            print(f"  Processed: {idx + 1}/{len(ds)} | Kept: {kept_count} | Discarded: {discarded_count} | No result: {no_result_count}")
    
    print(f"\n{'='*80}")
    print(f"Filtering Results:")
    print(f"  Original samples: {len(ds)}")
    print(f"  Kept (error <= {max_error}): {kept_count} ({kept_count/len(ds)*100:.2f}%)")
    print(f"  Discarded (error > {max_error}): {discarded_count} ({discarded_count/len(ds)*100:.2f}%)")
    print(f"  No reject sampling result: {no_result_count} ({no_result_count/len(ds)*100:.2f}%)")
    print(f"  Final dataset size: {len(filtered_data)}")
    print(f"{'='*80}\n")
    
    # 打印详细的误差分布
    if all_errors:
        print(f"{'='*80}")
        print(f"Error Distribution (from matched samples):")
        print(f"{'='*80}")
        
        # 定义分布区间
        dist_bins = [0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, float('inf')]
        dist_labels = ['[0, 0.25)', '[0.25, 0.5)', '[0.5, 0.75)', '[0.75, 1.0)', 
                       '[1.0, 1.5)', '[1.5, 2.0)', '[2.0, inf)']
        
        # 过滤掉 inf 的误差
        valid_errors = [e for e in all_errors if e != float('inf')]
        
        if valid_errors:
            hist, _ = np.histogram(valid_errors, bins=dist_bins)
            for label, count in zip(dist_labels, hist):
                percentage = (count / len(valid_errors)) * 100
                print(f"  {label:<15} {count:>5} ({percentage:>5.2f}%)")
            
            if len(all_errors) > len(valid_errors):
                inf_count = len(all_errors) - len(valid_errors)
                inf_percentage = (inf_count / len(all_errors)) * 100
                print(f"  {'[inf]':<15} {inf_count:>5} ({inf_percentage:>5.2f}%)")
        
        print(f"{'='*80}\n")
    
    # 创建新的数据集，保持原始特征
    filtered_ds = Dataset.from_list(filtered_data, features=ds.features)
    
    return filtered_ds

def main():
    parser = argparse.ArgumentParser(description="Filter dataset based on reject sampling results")
    parser.add_argument("--dataset_name", type=str, default="Coobiw/merged_agiqa5k_prompt_1022",
                        help="Source dataset name on HuggingFace")
    parser.add_argument("--results_dir", type=str, default="./reject_sample_results",
                        help="Directory containing reject sampling results")
    parser.add_argument("--checkpoint_name", type=str, default="qwen2_5_vl_7b_baseline",
                        help="Checkpoint name to use for filtering")
    parser.add_argument("--max_error", type=float, default=0.75,
                        help="Maximum error threshold (samples with error > this will be discarded)")
    parser.add_argument("--output_repo", type=str, default="Coobiw/agiqa3k_prompt_rejected_1025",
                        help="Output repository name on HuggingFace")
    parser.add_argument("--local_cache", type=str, default="./filtered_dataset_cache",
                        help="Local cache directory for saving dataset before upload")
    parser.add_argument("--private", action="store_true",
                        help="Make the dataset private on HuggingFace")
    parser.add_argument("--keep_cache", action="store_true",
                        help="Keep local cache after upload")
    args = parser.parse_args()
    
    print("="*80)
    print("Reject Sampling Dataset Filter")
    print("="*80)
    print(f"\nConfiguration:")
    print(f"  Source dataset: {args.dataset_name}")
    print(f"  Results directory: {args.results_dir}")
    print(f"  Checkpoint name: {args.checkpoint_name}")
    print(f"  Max error threshold: {args.max_error}")
    print(f"  Output repository: {args.output_repo}")
    print(f"  Local cache: {args.local_cache}")
    print(f"  Private: {args.private}")
    print()
    
    # 1. 加载 reject sampling 结果
    reject_results = load_reject_sampling_results(args.results_dir, args.checkpoint_name)
    
    if not reject_results:
        print("\n❌ No reject sampling results found!")
        return
    
    # 2. 过滤数据集
    filtered_ds = filter_dataset(args.dataset_name, reject_results, args.max_error)
    
    if len(filtered_ds) == 0:
        print("\n❌ No samples passed the filter!")
        return
    
    # 3. 保存到本地
    print(f"Saving filtered dataset to local: {args.local_cache}")
    filtered_ds.save_to_disk(args.local_cache)
    print(f"✅ Saved to local")
    
    # 4. 推送到 HuggingFace
    print(f"\n{'='*80}")
    print(f"Pushing to HuggingFace: {args.output_repo}")
    print(f"{'='*80}")
    
    from datasets import load_from_disk
    ds_to_upload = load_from_disk(args.local_cache)
    
    try:
        ds_to_upload.push_to_hub(
            args.output_repo,
            private=args.private,
            token=True
        )
        print(f"\n✅ Successfully pushed dataset to: https://huggingface.co/datasets/{args.output_repo}")
    except Exception as e:
        print(f"\n❌ Failed to push to HuggingFace: {e}")
        print(f"   Dataset is saved locally at: {args.local_cache}")
        return
    
    # 5. 清理本地缓存（可选）
    if not args.keep_cache:
        print(f"\nCleaning up local cache: {args.local_cache}")
        shutil.rmtree(args.local_cache)
        print(f"✅ Local cache cleaned")
    else:
        print(f"\nLocal cache kept at: {args.local_cache}")
    
    print(f"\n{'='*80}")
    print("Done!")
    print(f"{'='*80}")

if __name__ == "__main__":
    main()
