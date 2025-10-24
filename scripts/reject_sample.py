#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Reject Sampling 逻辑：
1. 加载 HuggingFace 数据集 Coobiw/merged_agiqa5k_prompt_1022 (train split)
2. 启动 vLLM 服务
3. 对每条样本通过 vLLM 服务 rollout N 次（默认16次）
4. 计算每次输出和 ground truth answer 的差值
5. 取最小差值作为该样本的误差
6. 画直方图统计误差分布
7. 停止 vLLM 服务
"""

import os
import io
import argparse
import json
import time
import signal
import subprocess
import requests
from pathlib import Path
from tqdm import tqdm
import base64
import asyncio
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

from openai import AsyncOpenAI
from datasets import load_dataset

# ----------- vLLM 服务管理 ----------------------------------------------------
def wait_for_server(port=8000, timeout=600, check_interval=10):
    """等待 vLLM 服务启动"""
    print(f"Waiting for vLLM server to start on port {port}...")
    start_time = time.time()
    while time.time() - start_time < timeout:
        try:
            response = requests.get(f"http://localhost:{port}/health", timeout=5)
            if response.status_code == 200:
                print("✅ vLLM server is ready!")
                return True
        except Exception:
            pass
        time.sleep(check_interval)
    raise TimeoutError(f"vLLM server did not start within {timeout} seconds")

def kill_process_group(process):
    """安全地 kill 进程组"""
    try:
        # 发送 SIGTERM 到整个进程组
        os.killpg(os.getpgid(process.pid), signal.SIGTERM)
        # 等待进程结束，最多等待10秒
        process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        # 如果进程还没结束，强制 kill
        print("Force killing vLLM server with SIGKILL...")
        try:
            os.killpg(os.getpgid(process.pid), signal.SIGKILL)
            process.wait()
        except:
            pass
    except (ProcessLookupError, PermissionError):
        # 进程已经不存在了
        pass

# ----------- 工具函数 ---------------------------------------------------------
def pil2base64(image):
    """将 PIL Image 转换为 base64 编码"""
    with io.BytesIO() as output:
        image.save(output, format="PNG")
        return base64.b64encode(output.getvalue()).decode("utf-8")

def extract_answer_value(response_text: str) -> float:
    """从模型响应中提取答案数值"""
    try:
        # 尝试提取 <answer> 标签中的内容
        answer_start = response_text.find("<answer>")
        answer_end = response_text.find("</answer>", answer_start + len("<answer>"))
        
        if answer_end == -1:
            if answer_start == -1:
                # 没有找到标签，直接用整个响应
                answer_text = response_text
            else:
                # 只找到开始标签
                answer_text = response_text[answer_start + len("<answer>"):]
        else:
            # 找到完整标签对
            answer_text = response_text[answer_start + len("<answer>"):answer_end]
        
        # 转换为浮点数
        value = float(answer_text.strip())
        return value
    except Exception as e:
        # 解析失败返回 None
        return None

def compute_error(pred_value: float, gt_value: float) -> float:
    """计算预测值和真实值的差值（绝对误差）"""
    if pred_value is None:
        return float('inf')  # 解析失败视为无穷大误差
    return abs(pred_value - gt_value)

# ----------- 数据生成器 -------------------------------------------------------
def prepare_dataset(dataset, sys_prompt: str):
    """准备数据集，转换为模型输入格式"""
    prepared_data = []
    for data_item in tqdm(dataset, desc="Preparing Dataset: "):
        # 检查数据项结构
        if 'images' in data_item and len(data_item['images']) > 0:
            image = data_item['images'][0]  # PIL.Image
            query = f"{data_item['problem'].replace('<image>', '').strip()} {sys_prompt}"
            gt_answer = data_item['answer']
            
            prepared_data.append({
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image_url",
                                "image_url": {"url": f"data:image;base64,{pil2base64(image)}"}
                            },
                            {"type": "text", "text": query}
                        ],
                    }
                ],
                "ground_truth": gt_answer,
                "problem": data_item['problem'],
            })
    return prepared_data

# ----------- Async 推理（支持多次采样）----------------------------------------
async def async_single_rollout(
    client: AsyncOpenAI,
    item: dict,
    model_name: str,
    temperature: float = 1.0,
    max_retries: int = 10,
):
    """单次异步推理（带重试逻辑）"""
    for retry in range(max_retries):
        try:
            resp = await client.chat.completions.create(
                model=model_name,
                messages=item["messages"],
                temperature=temperature,
                max_tokens=4096,
            )
            ret = resp.choices[0].message.content
            print(f"\033[91m{ret}\033[0m")
            return ret
        except Exception as e:
            if retry < max_retries - 1:
                wait_time = min(2 ** retry, 30)  # 指数退避，最多等待30秒
                # 只在前3次重试时打印详细信息，避免刷屏
                if retry < 3:
                    tqdm.write(f"⚠️  Retry {retry+1}/{max_retries}: {str(e)[:50]}... (wait {wait_time}s)")
                await asyncio.sleep(wait_time)
            else:
                tqdm.write(f"❌ Failed after {max_retries} retries: {str(e)[:80]}")
                return None

async def async_reject_sample_item(
    client: AsyncOpenAI,
    item: dict,
    model_name: str,
    num_rollout: int = 16,
    temperature: float = 0.8,
    max_retries: int = 10,
):
    """对单个样本进行 reject sampling（多次 rollout）- 不再使用 semaphore"""
    return await _async_reject_sample_item_impl(
        client, item, model_name, num_rollout, temperature, max_retries
    )

async def _async_reject_sample_item_impl(
    client: AsyncOpenAI,
    item: dict,
    model_name: str,
    num_rollout: int,
    temperature: float,
    max_retries: int,
):
    """实际的 reject sampling 实现"""
    gt_value = float(item["ground_truth"])
    
    # 并发进行 num_rollout 次 rollout
    tasks = [
        asyncio.create_task(
            async_single_rollout(client, item, model_name, temperature, max_retries)
        )
        for _ in range(num_rollout)
    ]
    responses = await asyncio.gather(*tasks)
    
    # 计算每次的误差
    errors = []
    valid_responses = []
    for resp in responses:
        if resp is not None:
            pred_value = extract_answer_value(resp)
            error = compute_error(pred_value, gt_value)
            errors.append(error)
            valid_responses.append({
                "response": resp,
                "pred_value": pred_value,
                "error": error,
            })
    
    # 找到最小误差
    if len(errors) > 0:
        min_error = min(errors)
        min_error_idx = errors.index(min_error)
        best_response = valid_responses[min_error_idx]
    else:
        min_error = float('inf')
        best_response = None
    
    return {
        "ground_truth": gt_value,
        "min_error": min_error,
        "num_valid_rollouts": len(errors),
        "num_total_rollouts": num_rollout,
        "best_response": best_response,
        "all_errors": errors,
    }

# ----------- 主入口 -----------------------------------------------------------
async def main():
    parser = argparse.ArgumentParser(description="Reject Sampling for AGIQA Dataset")
    parser.add_argument("--model_path", type=str, required=True, help="模型路径")
    parser.add_argument("--model_name", type=str, required=True, help="模型名称")
    parser.add_argument("--model_port", type=int, default=8000, help="vLLM Server 端口")
    parser.add_argument("--num_rollout", type=int, default=16, help="每条数据 rollout 次数")
    parser.add_argument("--temperature", type=float, default=1.0, help="采样温度")
    parser.add_argument("--concurrency", type=int, default=8, help="并发处理的数据条数")
    parser.add_argument("--max_retries", type=int, default=10, help="rollout 失败时的重试次数")
    parser.add_argument("--max_data", type=int, default=None, help="最大处理数据量（用于测试）")
    parser.add_argument("--output_dir", type=str, default="./reject_sampling_results", 
                        help="结果保存目录")
    parser.add_argument("--log_dir", type=str, default="./logs", help="vLLM 服务日志目录")
    parser.add_argument("--skip_vllm_launch", action="store_true", 
                        help="跳过 vLLM 服务启动（假设服务已在运行）")
    args = parser.parse_args()
    
    print("="*80)
    print("Reject Sampling with vLLM")
    print("="*80)
    print(f"Model Path: {args.model_path}")
    print(f"Model Name: {args.model_name}")
    print(f"Model Port: {args.model_port}")
    print(f"Rollouts per sample: {args.num_rollout}")
    print(f"Temperature: {args.temperature}")
    print(f"Max Retries: {args.max_retries}")
    print(f"Concurrency: {args.concurrency}")
    print("="*80)
    print()
    
    # 系统提示词
    sys_prompt = (
        "A conversation between User and Assistant. The user asks a question, and the Assistant solves it. "
        "The assistant first thinks about the reasoning process in the mind and then provides the user with the answer. "
        "The reasoning process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, "
        "i.e., <think> reasoning process here </think><answer> answer here </answer>"
    )
    
    # 启动 vLLM 服务
    serve_process = None
    if not args.skip_vllm_launch:
        log_dir = Path(args.log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)
        log_file = log_dir / f"{args.model_name}_vllm.log"
        
        print(f"Starting vLLM server for {args.model_name}...")
        print(f"Server logs will be saved to: {log_file}")
        
        with open(log_file, "w") as f:
            serve_process = subprocess.Popen(
                ["bash", "scripts/vllm_serve.sh", args.model_path, args.model_name],
                stdout=f,
                stderr=subprocess.STDOUT,
                preexec_fn=os.setsid  # 创建新的进程组
            )
        
        try:
            # 等待服务启动
            wait_for_server(port=args.model_port, timeout=600)
            
            # 额外等待确保模型加载完成
            print("Waiting additional 30 seconds for model to fully load...")
            time.sleep(30)
        except TimeoutError as e:
            print(f"❌ Timeout error: {e}")
            print(f"   Check vLLM logs at: {log_file}")
            if serve_process:
                kill_process_group(serve_process)
            return
        except Exception as e:
            print(f"❌ Error starting vLLM server: {e}")
            if serve_process:
                kill_process_group(serve_process)
            return
    else:
        print("⚠️  Skipping vLLM launch, assuming service is already running...")
    
    try:
        # 加载数据集
        print("\nLoading dataset from HuggingFace...")
        dataset = load_dataset("Coobiw/merged_agiqa5k_prompt_1022", split="train")
        print(f"Dataset loaded: {len(dataset)} samples")
        
        # 准备数据
        print("Preparing data...")
        prepared_data = prepare_dataset(dataset, sys_prompt)
        del dataset  # 释放内存
        
        # 限制数据量（用于测试）
        if args.max_data is not None:
            prepared_data = prepared_data[:args.max_data]
            print(f"Limited to {len(prepared_data)} samples for testing")
        
        # 设置 OpenAI 客户端
        openai_api_base = f"http://localhost:{args.model_port}/v1"
        client = AsyncOpenAI(api_key="EMPTY", base_url=openai_api_base)
        
        # 创建输出目录
        output_dir = Path(args.output_dir) / args.model_name
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # 进行 reject sampling
        print(f"\nStarting reject sampling with {args.num_rollout} rollouts per data point...")
        print(f"Sample-level Concurrency: {args.concurrency} samples in parallel")
        print(f"Rollout-level Concurrency: {args.num_rollout} rollouts per sample")
        print(f"Total parallel requests: up to {args.concurrency * args.num_rollout}")
        print(f"Temperature: {args.temperature}")
        print(f"Max Retries: {args.max_retries}")
        print()
        print(f"⏳ Note: First batch may take longer (model warmup)...")
        print(f"📊 Progress bar updates every time a sample completes (not just per batch)")
        print()
        
        results = []
        total_rollouts = 0
        successful_rollouts = 0
        
        # 分批处理，每批并发处理 concurrency 个样本
        batch_size = args.concurrency
        num_batches = (len(prepared_data) + batch_size - 1) // batch_size
        
        # 使用 tqdm 显示样本级别的进度
        with tqdm(total=len(prepared_data), desc="Processing Samples", unit="sample", 
                  smoothing=0.1, mininterval=0.5) as pbar:
            for batch_idx in range(num_batches):
                start_idx = batch_idx * batch_size
                end_idx = min(start_idx + batch_size, len(prepared_data))
                batch = prepared_data[start_idx:end_idx]
                
                # 并发处理一个 batch 内的所有样本
                # 创建任务字典，用于追踪 task 和 item 的对应关系
                task_to_item = {}
                for item in batch:
                    task = asyncio.create_task(
                        async_reject_sample_item(
                            client=client,
                            item=item,
                            model_name=args.model_name,
                            num_rollout=args.num_rollout,
                            temperature=args.temperature,
                            max_retries=args.max_retries,
                        )
                    )
                    task_to_item[task] = item
                
                # 使用 wait 来实时监控任务完成情况
                pending = set(task_to_item.keys())
                while pending:
                    # 等待至少一个任务完成
                    done, pending = await asyncio.wait(pending, return_when=asyncio.FIRST_COMPLETED)
                    
                    # 处理所有已完成的任务
                    for task in done:
                        result = await task
                        item = task_to_item[task]
                        result["problem"] = item["problem"]
                        results.append(result)
                        
                        # 更新统计
                        total_rollouts += result["num_total_rollouts"]
                        successful_rollouts += result["num_valid_rollouts"]
                        success_rate = (successful_rollouts / total_rollouts * 100) if total_rollouts > 0 else 0
                        avg_valid = successful_rollouts / len(results) if results else 0
                        
                        # 计算最近的平均误差（避免重复计算所有结果）
                        recent_errors = [r["min_error"] for r in results[-100:] if r["min_error"] != float('inf')]
                        avg_min_err = np.mean(recent_errors) if recent_errors else 0
                        
                        # 实时更新进度条
                        pbar.set_postfix({
                            'batch': f'{batch_idx+1}/{num_batches}',
                            'valid': f'{result["num_valid_rollouts"]}/{args.num_rollout}',
                            'err': f'{result["min_error"]:.3f}' if result["min_error"] != float('inf') else 'inf',
                            'avg': f'{avg_min_err:.3f}',
                            'succ': f'{success_rate:.1f}%'
                        })
                        pbar.update(1)
    
        # 打印 rollout 统计
        print(f"\n{'='*80}")
        print("ROLLOUT STATISTICS")
        print(f"{'='*80}")
        print(f"Total samples processed: {len(results)}")
        print(f"Total rollouts attempted: {total_rollouts}")
        print(f"Successful rollouts: {successful_rollouts}")
        print(f"Failed rollouts: {total_rollouts - successful_rollouts}")
        print(f"Overall success rate: {success_rate:.2f}%")
        
        # 保存详细结果
        output_file = output_dir / "reject_sampling_results.json"
        with open(output_file, "w") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        print(f"\n✅ Detailed results saved to: {output_file}")
        
        # 统计分析
        print("\n" + "="*80)
        print("STATISTICS")
        print("="*80)
        
        min_errors = [r["min_error"] for r in results if r["min_error"] != float('inf')]
        valid_count = len(min_errors)
        total_count = len(results)
        
        print(f"\nTotal samples: {total_count}")
        print(f"Valid samples: {valid_count} ({valid_count/total_count*100:.2f}%)")
        print(f"Failed samples: {total_count - valid_count}")
        
        if len(min_errors) > 0:
            print(f"\nError Statistics:")
            print(f"  Mean error: {np.mean(min_errors):.4f}")
            print(f"  Median error: {np.median(min_errors):.4f}")
            print(f"  Std error: {np.std(min_errors):.4f}")
            print(f"  Min error: {np.min(min_errors):.4f}")
            print(f"  Max error: {np.max(min_errors):.4f}")
            
            # 误差分布统计
            print(f"\nError Distribution:")
            error_bins = [0, 0.1, 0.2, 0.3, 0.5, 0.8, 1.0, 1.5, 2.0, float('inf')]
            error_counts = defaultdict(int)
            for err in min_errors:
                for i in range(len(error_bins) - 1):
                    if error_bins[i] <= err < error_bins[i+1]:
                        bin_name = f"[{error_bins[i]}, {error_bins[i+1]})"
                        error_counts[bin_name] += 1
                        break
            
            for bin_name, count in sorted(error_counts.items()):
                percentage = count / len(min_errors) * 100
                print(f"  {bin_name}: {count} ({percentage:.2f}%)")
            
            # 画直方图
            print(f"\nGenerating histogram...")
            plt.figure(figsize=(12, 6))
            
            # 子图1：所有误差
            plt.subplot(1, 2, 1)
            plt.hist(min_errors, bins=50, edgecolor='black', alpha=0.7)
            plt.xlabel('Minimum Error')
            plt.ylabel('Frequency')
            plt.title(f'Distribution of Minimum Errors\n(n={len(min_errors)})')
            plt.grid(True, alpha=0.3)
            
            # 子图2：误差 < 2.0 的部分（放大看细节）
            filtered_errors = [e for e in min_errors if e < 2.0]
            plt.subplot(1, 2, 2)
            plt.hist(filtered_errors, bins=50, edgecolor='black', alpha=0.7, color='orange')
            plt.xlabel('Minimum Error')
            plt.ylabel('Frequency')
            plt.title(f'Distribution of Minimum Errors (< 2.0)\n(n={len(filtered_errors)})')
            plt.grid(True, alpha=0.3)
            
            plt.tight_layout()
            hist_file = output_dir / "error_distribution.png"
            plt.savefig(hist_file, dpi=300, bbox_inches='tight')
            print(f"✅ Histogram saved to: {hist_file}")
            
            # 保存统计摘要
            stats_file = output_dir / "statistics.txt"
            with open(stats_file, "w") as f:
                f.write(f"Reject Sampling Statistics\n")
                f.write(f"="*80 + "\n\n")
                f.write(f"Model: {args.model_name}\n")
                f.write(f"Rollouts per data point: {args.num_rollout}\n")
                f.write(f"Temperature: {args.temperature}\n")
                f.write(f"Max Retries: {args.max_retries}\n\n")
                f.write(f"Total samples: {total_count}\n")
                f.write(f"Valid samples: {valid_count} ({valid_count/total_count*100:.2f}%)\n")
                f.write(f"Failed samples: {total_count - valid_count}\n\n")
                f.write(f"Error Statistics:\n")
                f.write(f"  Mean: {np.mean(min_errors):.4f}\n")
                f.write(f"  Median: {np.median(min_errors):.4f}\n")
                f.write(f"  Std: {np.std(min_errors):.4f}\n")
                f.write(f"  Min: {np.min(min_errors):.4f}\n")
                f.write(f"  Max: {np.max(min_errors):.4f}\n\n")
                f.write(f"Error Distribution:\n")
                for bin_name, count in sorted(error_counts.items()):
                    percentage = count / len(min_errors) * 100
                    f.write(f"  {bin_name}: {count} ({percentage:.2f}%)\n")
            print(f"✅ Statistics saved to: {stats_file}")
        else:
            print("\n❌ No valid samples to analyze!")
        
        print("\n" + "="*80)
        print("COMPLETED")
        print("="*80)
        
    finally:
        # 停止 vLLM 服务
        if serve_process is not None:
            print("\nStopping vLLM server...")
            kill_process_group(serve_process)
            print("vLLM server stopped")
            
            # 等待端口释放
            print("Waiting for port to be released...")
            time.sleep(10)

if __name__ == "__main__":
    asyncio.run(main())


