import os
import json
from pathlib import Path
from tqdm import tqdm

from transformers import Qwen2_5_VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info

import torch
from torch.utils.data import Dataset

import random
import numpy as np

random.seed(42)
np.random.seed(42)

class AGIQA3k(Dataset):

    def __init__(self, annos, sys_prompt, query_format):
        super().__init__()
        self.annos = []
        with open(annos) as f:
            lines = f.readlines()
            for line in lines:
                line = line.strip()
                item = json.loads(line)
                self.annos.append(item)
            self.sys_prompt = sys_prompt
            self.user_query_format = query_format

    def __len__(self):
        return len(self.annos)

    def __getitem__(self,idx):
        item = self.annos[idx]
        return {
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "image": item['image'],
                        },
                        {"type": "text", "text": self.user_query_format.format(item['prompt']) + self.sys_prompt},
                    ],
                }
            ],
            "mos_perception": item['mos_perception'],
            "mos_align": item['mos_align']
        }

def model_gen(model, processor, messages):
    # Preparation for inference
    texts = [
        processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=True)
        for msg in messages
    ]
    # import pdb;pdb.set_trace()
    image_inputs, _ = process_vision_info(messages)
    inputs = processor(
        text=texts,
        images=image_inputs,
        padding=True,
        return_tensors="pt",
    )
    inputs = inputs.to("cuda")
    
    # Batch Inference: Generation of the output
    generated_ids = model.generate(**inputs, max_new_tokens=2048, do_sample=False)

    generated_ids_trimmed = [
        out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_texts = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )
    return output_texts


if __name__ == "__main__":
    model_path = "/code/All-In-One/qbw/EasyR1-20250410/cache/ckpt/Qwen2.5-VL-7B-Instruct"
    model_name = str(Path(model_path).name)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_path, 
        torch_dtype=torch.bfloat16, 
        device_map="cuda",
        attn_implementation="flash_attention_2",
    ).eval()
    
    
    max_pixels = 1048576 # 1024 x 1024
    min_pixels = 262144 # 512 x 512
    processor = AutoProcessor.from_pretrained("/code/All-In-One/qbw/EasyR1-20250410/cache/ckpt/Qwen2.5-VL-7B-Instruct", min_pixels=min_pixels, max_pixels=max_pixels)
    processor.tokenizer.padding_side  = 'left'

    annos = "/code/All-In-One/qbw/EasyR1-20250410/cache/data/AGIQA-3k/annos/data_jsonl.jsonl"
    sys_prompt = 'A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The assistant first thinks about the reasoning process in the mind and then provides the user with the answer. The reasoning process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., <think> reasoning process here </think><answer> answer here </answer>'

    # return_dtype, lower_bound, upper_bound  = "float", "1", "5"
    return_dtype, lower_bound, upper_bound  = "int", "1", "100"
    mid_prompt = " rounded to two decimal places," if return_dtype == "float" else ""
    query_format = 'What is your overall rating on the quality of this AI-generated picture with the textual prompt: "{}"?' + f' The rating should be a {return_dtype} between {lower_bound} and {upper_bound},{mid_prompt} with {lower_bound} representing very poor quality and {upper_bound} representing excellent quality. Return the final answer like: <answer> the score </answer>\n\n'

    agiqa_3k = AGIQA3k(annos, sys_prompt, query_format)
    output = []
    output_fname = f"/code/All-In-One/qbw/EasyR1-20250410/eval_results/agiqa-3k/{model_name}_{return_dtype}_{lower_bound}_{upper_bound}.json"

    eval_bs = 128
    indices = list(range(len(agiqa_3k)))[::eval_bs]
    l = len(agiqa_3k)
    for start_idx in tqdm(indices):
        if start_idx + eval_bs > l:
            items = [agiqa_3k[idx] for idx in range(start_idx, l)]
        else:
            items = [agiqa_3k[idx] for idx in range(start_idx, start_idx + eval_bs)]

        batch_messages = [item['messages'] for item in items]
        model_responses = model_gen(model, processor, batch_messages)

        for response_idx, model_response in enumerate(model_responses):
            item = items[response_idx]
            mos_perception = item['mos_perception']
            mos_align = item['mos_align']
            item['model_response'] = model_response
            output.append(item)

    with open(output_fname, 'w') as fo:
        json.dump(output, fo, ensure_ascii=False, indent=4)