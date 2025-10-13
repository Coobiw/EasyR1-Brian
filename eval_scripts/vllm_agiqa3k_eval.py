#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
优化点：
1. 新增 --concurrency 参数，可直接控制并发数
2. 统一 base64 编码函数，避免多次定义
3. 把进程池换成 asyncio + AsyncOpenAI（I/O 密集 → 线程 / 协程更划算）
4. 支持批量推理：一个 batch 内所有请求并发发送，batch 间顺序执行，显著提高吞吐
5. 关键超参（batch_size、concurrency 等）可命令行调整
"""

import os
import io
import argparse
import json
from pathlib import Path
from tqdm import tqdm

from openai import OpenAI, AsyncOpenAI

from datasets import load_dataset

import torch
from torch.utils.data import Dataset

import random
import numpy as np
import base64
import asyncio   # + 新增

from scipy import stats
from scipy.optimize import curve_fit

# ----------- 随机种子 ---------------------------------------------------------
def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)

set_seed(42)


def logistic_func(X, bayta1, bayta2, bayta3, bayta4):
    logisticPart = 1 + np.exp(np.negative(np.divide(X - bayta3, np.abs(bayta4))))
    yhat = bayta2 + np.divide(bayta1 - bayta2, logisticPart)
    return yhat

def fit_function(y_label, y_output):
    beta = [np.max(y_label), np.min(y_label), np.mean(y_output), 0.5]
    popt, _ = curve_fit(logistic_func, y_output, \
        y_label, p0=beta, maxfev=100000000)
    y_output_logistic = logistic_func(y_output, *popt)
    
    return y_output_logistic


def performance_fit(y_label, y_output, func_fit=True):
    if func_fit:
        y_output_logistic = fit_function(y_label, y_output)
    else:
        y_output_logistic = y_output
    PLCC = stats.pearsonr(y_output_logistic, y_label)[0]
    SRCC = stats.spearmanr(y_output, y_label)[0]

    return PLCC, SRCC, (PLCC+SRCC) / 2

# ----------- 数据生成器 -------------------------------------------------------
def agiqa3k_generator(dataset: Dataset, sys_prompt: str):
    new_dataset = []
    for data_item in dataset:
        image = data_item['images'][0] # PIL.Image
        query = f"{data_item['problem'].replace('<image>', '').strip()} {sys_prompt}"
        mos_gt = data_item['answer']
        new_dataset.append({
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": f"data:image;base64,{pil2base64(image)}"}}, 
                        {"type": "text", "text": query}
                    ],
                }
            ],
            "mos_perception": mos_gt,
        })
    return new_dataset

# ----------- 工具函数 ---------------------------------------------------------
def pil2base64(image):
    with io.BytesIO() as output:
        image.save(output, format="PNG")
        return base64.b64encode(output.getvalue()).decode("utf-8")

# ----------- Async 推理 -------------------------------------------------------
async def async_single_item_infer(
    client: AsyncOpenAI,
    item: dict,
    model_name: str,
):
    """单条异步推理"""
    resp = await client.chat.completions.create(
        model=model_name,
        messages=item["messages"],
        temperature=0.0,
        max_tokens=4096,
    )
    return resp.choices[0].message.content

async def async_infer_batch(
    client: AsyncOpenAI,
    batch_items: list,
    model_name: str,
    semaphore: asyncio.Semaphore,
):
    """一个 batch 内部并发推理；使用 semaphore 控制全局并发上限"""
    async with semaphore:
        tasks = [
            asyncio.create_task(
                async_single_item_infer(client, item, model_name)
            )
            for item in batch_items
        ]
        return await asyncio.gather(*tasks)

# ----------- 主入口 -----------------------------------------------------------
async def main():
    parser = argparse.ArgumentParser(description="vLLM Qwen2.5-VL 推理脚本")
    parser.add_argument("--model_port", default="8000", help="vLLM Server 端口")
    parser.add_argument("--model_name", required=True, help="模型名称")
    parser.add_argument("--rl_prompt", type=int, default=1, choices=[0, 1])
    
    parser.add_argument("--concurrency", type=int, default=64, help="并发请求数")
    args = parser.parse_args()

    annos = "/code/All-In-One/qbw/EasyR1-20250410/cache/data/AGIQA-3k/annos/test.jsonl"
    # ...（下方 prompt 构造逻辑保持不变，略）...

    sys_prompt = "A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The assistant first thinks about the reasoning process in the mind and then provides the user with the answer. The reasoning process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., <think> reasoning process here </think><answer> answer here </answer>"

    dataset = load_dataset("Coobiw/agiqa3k_finale_1013")['test']
    new_dataset = agiqa3k_generator(dataset, sys_prompt)
    del dataset
    
    eval_bs = 512  # 一次读多少条进入内存
    indices = list(range(0, len(new_dataset), eval_bs))
    openai_api_base = f"http://localhost:{args.model_port}/v1"

    # + 使用 AsyncOpenAI
    client = AsyncOpenAI(api_key="EMPTY", base_url=openai_api_base)
    semaphore = asyncio.Semaphore(args.concurrency)

    output, output_fname = [], Path(
        f"/code/All-In-One/qbw/EasyR1-20250410/eval_results/agiqa-3k_vllm/{args.model_name}/rollout_results.json"
    )
    output_fname.parent.mkdir(parents=True, exist_ok=True)

    for start_idx in tqdm(indices, desc="Batches"):
        batch = new_dataset[start_idx : start_idx + eval_bs]
        # + 真正的并发推理
        responses = await async_infer_batch(
            client, batch, args.model_name, semaphore
        )
        for item, model_resp in zip(batch, responses):
            item["model_response"] = model_resp
            output.append(item)

    with open(output_fname, "w") as fo:
        json.dump(output, fo, ensure_ascii=False, indent=4)
    print(f"✅ 推理完毕，结果已保存到 {output_fname}")
    
    y_label, y_out = [], []
    error_count = 0
    for i, item in enumerate(output):
        model_response = item['model_response']
        try:
            answer_start = model_response.find("<answer>")
            answer_end = model_response.find("</answer>", answer_start + len("<answer>"))
            if answer_end == -1:
                if answer_start == -1:
                    model_response = model_response
                else:
                    model_response = model_response[answer_start+len("<answer>") : ]
            else:
                model_response = model_response[answer_start+len("<answer>") : answer_end]
                
            out = float(model_response.strip())
            y_out.append(out)
            y_label.append(float(item['mos_perception']))
        except Exception as e:
            error_count += 1
            print(f"{i}th error:\t", e)
            
    print(error_count)
    output1 = performance_fit(y_label, y_out, func_fit=True)
    output2 = performance_fit(y_label, y_out, func_fit=False)

    print(output1)
    print(output2)
    
    out_score = os.path.join(output_fname.parent, "score.txt")
    with open(out_score, 'w') as fo:
        fo.write(f"PLCC: {output1[0]}\n")
        fo.write(f"SRCC: {output1[1]}\n")
        fo.write(f"MainScore: {output1[2]}\n")
        fo.write(f"PLCC: {output2[0]}\n")
        fo.write(f"SRCC: {output2[1]}\n")
        fo.write(f"MainScore: {output2[2]}\n")

if __name__ == "__main__":
    # + 将整个流程丢给 asyncio 运行
    asyncio.run(main())
