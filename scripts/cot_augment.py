from openai import OpenAI
import os

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
import re
import time  # + 新增

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

# ----------- Async 推理 -------------------------------------------------------
async def async_single_item_infer(
    client: AsyncOpenAI,
    item: dict,
):
    """单条异步推理"""
    max_retries = 3
    retry_count = 0
    
    while retry_count < max_retries:
        try:
            resp = await client.chat.completions.create(
                model="qwen2.5-vl-72b-instruct",
                messages=item["messages"],
                temperature=1.0,
                max_tokens=8*1024,
            )
            return resp.choices[0].message.content
            
        except Exception as e:
            if "429" in str(e) or "rate limit" in str(e).lower() or "limit_requests" in str(e):
                print(f"⚠️  遇到限流错误，等待60秒后重试... (重试次数: {retry_count + 1}/{max_retries})")
                await asyncio.sleep(60)  # 等待1分钟
                retry_count += 1
                if retry_count >= max_retries:
                    print(f"❌ 重试{max_retries}次后仍然失败，跳过该请求")
                    return f"ERROR: Rate limit exceeded after {max_retries} retries"
            else:
                print(f"❌ 其他错误: {e}")
                return f"ERROR: {str(e)}"
    
    return "ERROR: Max retries exceeded"

async def async_infer_batch(
    client: AsyncOpenAI,
    batch_items: list,
    semaphore: asyncio.Semaphore,
    pbar=None,
):
    """一个 batch 内部并发推理；使用 semaphore 控制全局并发上限"""
    async def single_infer_with_progress(item):
        async with semaphore:  # 将 semaphore 移到单个请求级别
            result = await async_single_item_infer(client, item)
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
    parser = argparse.ArgumentParser(description="vLLM Qwen2.5-VL 72B API 推理脚本")
    
    parser.add_argument("--concurrency", type=int, default=32, help="并发请求数")
    args = parser.parse_args()


    # + 使用 AsyncOpenAI
    client = AsyncOpenAI(
        api_key=os.getenv("DASHSCOPE_API_KEY"),
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
    )

    semaphore = asyncio.Semaphore(args.concurrency)

    output = []
    input_fp = "/code/All-In-One/qbw/EasyR1-20250410/rollout_results/agiqa3k_rollout_temp1/agiqa3k_qual_n16_continuous-thres0p75_format0p1_on-policy_newcode_20250616_step144_roll4_float_1_5.json"
    output_fp = "/code/All-In-One/qbw/EasyR1-20250410/rollout_results/agiqa3k_rollout_temp1/agiqa3k_qual_n16_continuous-thres0p75_format0p1_on-policy_newcode_20250616_step144_roll4_float_1_5_sft-qwenvl-cot-augmented.json"

#     EVOLVE_PROMPT = """
# The provided image is an AI-generated picture, and the goal is to assess its visual quality. You are given a brief text evaluating both its overall and detailed visual quality, along with a final score (from 1 to 5, where 1 represents very poor quality and 5 represents excellent quality). Your task is to enrich this evaluation text based on the image’s content. You may elaborate on the quality-related aspects already mentioned, or introduce new details related to image quality that the original text did not cover. Throughout, your additions must remain faithful to the image’s actual details and content. Please directly output a more detailed comment, the comment should be a continuous paragraph (not formatted as a document), and the overall length should not be too long (fewer than 1000 words).

# Final Score: {final_score}

# Brief Comment: {brief_comment}

# Overall Comment:
#     """.strip()
    EVOLVE_PROMPT = """
    You are a meticulous visual-quality reviewer. Firstly, I've given you an image to be analysed.
    Then, I will give you:
    1. Final Score → quality score of the given image (integer 1-5, where 1 = very poor and 5 = excellent)
    2. Brief Comment → the brief reasoning progress for the given score (one-sentence summary written by another reviewer)
    
    Your task is to write one continuous paragraph (fewer than 1000 words) that greatly expands on the brief comment while strictly describing things that are visibly present in the image.
    
    While writing, follow this internal reasoning loop:
    1. Observe – Mentally list the main objects, colours, lighting, composition, perspective, textures, realism, coherence, artefacts, etc.
    2. Judge – For every observation, decide how it affects quality.
    3. Reflect – Pause and question yourself in-line using short self-checks such as "Wait… does the shadow direction contradict the light source?" and "Hmm, are the edges over-sharpened?" Immediately correct or refine earlier thoughts if needed.
    4. Synthesize – Merge the insights into a smooth narrative that mentions both strengths and weaknesses, justifies the given final score, and adds any new quality-related details not covered in the brief comment.
    
    Writing rules:
    1. Keep everything in one paragraph (no bullet points or line breaks). 
    2. Use natural language; self-reflection phrases may begin with "Wait…", "Hmm…", "On second thought…", etc.
    3. Do not reveal numbered steps—perform them mentally and embed the reflective phrases organically.
    4. Stay faithful to the image; do not hallucinate unseen details.
    5. End with a concise justification of why the image deserves the final_score.
    
    Final Score: {final_score}

    Brief Comment: {brief_comment}
    
    Overall Comment (Return only the enriched comment paragraph please.):
    """.strip()
    all_data = []
    with open(input_fp, 'r') as f_in, open(output_fp, 'w') as f_out:
        items = json.load(f_in)
        for data in items:
            response = data['model_response']
            gt = data['mos_perception']

            try:
                pattern = r'<think>(.*?)</think>(.*)<answer>(.*?)</answer>'
                match = re.search(pattern, response, re.DOTALL)
                thinking = match.groups(1)[0]
                summary = match.groups(1)[1]
                score = match.groups(1)[2]
    
                score = float(score.strip())
                
                if abs(score - gt) < 1:
                    data['messages'][0]['content'][1]['text'] = EVOLVE_PROMPT.format(
                        final_score = gt,
                        brief_comment = thinking
                    )
                    all_data.append(data)
            except Exception as e:
                print(e)
                
        print(f"开始并发推理 {len(all_data)} 条数据，并发数: {args.concurrency}")
        
        # 并发发送所有请求，添加进度条
        with tqdm(total=len(all_data), desc="推理进度") as pbar:
            responses = await async_infer_batch(
                client, all_data, semaphore, pbar
            )
        
        # 处理结果
        for item, model_resp in zip(all_data, responses):
            item["model_response"] = model_resp
            output.append(item)
    
        json.dump(output, f_out, ensure_ascii=False, indent=4)
        print(f"✅ 推理完毕，结果已保存到 {output_fp}")

if __name__ == "__main__":
    # + 将整个流程丢给 asyncio 运行
    print(os.getenv("DASHSCOPE_API_KEY"))
    asyncio.run(main())
