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
import argparse
import json
from pathlib import Path
from tqdm import tqdm

# - from openai import OpenAI
# + from openai import OpenAI, AsyncOpenAI
from openai import OpenAI, AsyncOpenAI

from transformers import Qwen2_5_VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info

import torch
from torch.utils.data import Dataset

import random
import numpy as np
import base64
import asyncio   # + 新增

# ----------- 随机种子 ---------------------------------------------------------
def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)

set_seed(42)

# ----------- 工具函数 ---------------------------------------------------------
def img_path2url(image_path):
    """本地图片 -> base64 data url（统一放到外层，避免 Dataset 里多次定义）"""
    with open(image_path, "rb") as f:
        encoded_image = base64.b64encode(f.read()).decode("utf-8")
    return f"data:image;base64,{encoded_image}"

# ----------- 数据集 -----------------------------------------------------------
class AGIQA3k(Dataset):
    def __init__(self, annos, sys_prompt, query_format):
        super().__init__()
        self.annos = []
        with open(annos) as f:
            for line in f:
                item = json.loads(line.strip())
                self.annos.append(item)
        self.sys_prompt = sys_prompt
        self.user_query_format = query_format

    def __len__(self):
        return len(self.annos)

    def __getitem__(self, idx):
        # 兼容 list / slice
        if isinstance(idx, slice):
            return [self.__getitem__(i) for i in range(*idx.indices(len(self)))]
        item = self.annos[idx]
        return {
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {"url": img_path2url(item['image'])},
                        },
                        {"type": "text", "text": self.user_query_format + self.sys_prompt},
                    ],
                },
            ],
            "mos_perception": item['mos_perception'],
            "mos_align": item['mos_align'],
            "image_path": item['image'],
        }

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
        temperature=1.0,
        max_tokens=4096,
    )
    return "<think>" + resp.choices[0].message.content

async def async_infer_batch(
    client: AsyncOpenAI,
    batch_items: list,
    model_name: str,
    semaphore: asyncio.Semaphore,
    pbar=None,
):
    """一个 batch 内部并发推理；使用 semaphore 控制全局并发上限"""
    async def single_infer_with_progress(item):
        async with semaphore:  # 将 semaphore 移到单个请求级别
            result = await async_single_item_infer(client, item, model_name)
            if pbar:
                pbar.update(1)
            return result
            
    tasks = [
        asyncio.create_task(
            single_infer_with_progress(item)
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
    parser.add_argument("--rollout_times", type=int, default=4, help="每条数据的rollout次数")
    args = parser.parse_args()

    annos = "/code/All-In-One/qbw/EasyR1-20250410/cache/data/AGIQA-3k/annos/train.jsonl"
    # ...（下方 prompt 构造逻辑保持不变，略）...

    if args.rl_prompt:
        sys_prompt = (
            "A conversation between User and Assistant. "
            "The user asks a question, and the Assistant solves it. "
            "The assistant first thinks about the reasoning process in the mind and then provides the user with the answer. "
            "The reasoning process and answer are enclosed within <think> </think> and <answer> </answer> tags, "
            "respectively, i.e., <think> reasoning process here </think><answer> answer here </answer>"
        )
        return_dtype, lower_bound, upper_bound = "float", "1", "5"
        mid_prompt = " rounded to two decimal places,"
        query_format = (
            "What is your overall rating on the quality of this AI-generated picture?"
            f" The rating should be a {return_dtype} between {lower_bound} and {upper_bound},{mid_prompt}"
            f" with {lower_bound} representing very poor quality and {upper_bound} representing excellent quality."
            " Return the final answer like: <answer> the score </answer>\n\n"
        )
    else:
        sys_prompt = ""
        return_dtype, lower_bound, upper_bound = "float", "1", "5"
        mid_prompt = " rounded to two decimal places,"
        query_format = (
            "What is your overall rating on the quality of this AI-generated picture?"
            f" The rating should be a {return_dtype} between {lower_bound} and {upper_bound},{mid_prompt}"
            f" with {lower_bound} representing very poor quality and {upper_bound} representing excellent quality."
            " Return the final answer directly.\n\n"
        )

    dataset = AGIQA3k(annos, sys_prompt, query_format)
    openai_api_base = f"http://localhost:{args.model_port}/v1"

    # + 使用 AsyncOpenAI
    client = AsyncOpenAI(api_key="EMPTY", base_url=openai_api_base)
    semaphore = asyncio.Semaphore(args.concurrency)

    output, output_fname = [], Path(
        f"/code/All-In-One/qbw/EasyR1-20250410/rollout_results/agiqa3k_rollout_temp1/"
        f"{args.model_name}_roll{args.rollout_times}_{return_dtype}_{lower_bound}_{upper_bound}.json"
    )
    output_fname.parent.mkdir(parents=True, exist_ok=True)

    # 获取所有训练数据
    all_data = []
    for i in range(len(dataset)):
        for _ in range(args.rollout_times):
            all_data.append(dataset[i])
    
    print(f"开始并发推理 {len(all_data)} 条数据（{len(dataset)} 条原始数据 × {args.rollout_times} 次rollout），并发数: {args.concurrency}")
    
    # 并发发送所有请求，添加进度条
    with tqdm(total=len(all_data), desc="推理进度") as pbar:
        responses = await async_infer_batch(
            client, all_data, args.model_name, semaphore, pbar
        )
    
    # 处理结果
    for item, model_resp in zip(all_data, responses):
        item["model_response"] = model_resp
        output.append(item)

    with open(output_fname, "w") as fo:
        json.dump(output, fo, ensure_ascii=False, indent=4)
    print(f"✅ 推理完毕，结果已保存到 {output_fname}")

if __name__ == "__main__":
    # + 将整个流程丢给 asyncio 运行
    asyncio.run(main())
