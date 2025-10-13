import json
import re

jsonl_fp = "/code/All-In-One/qbw/EasyR1-20250410/rollout_results/agiqa3k_rollout_temp1/agiqa3k_qual_n16_continuous-thres0p75_format0p1_on-policy_newcode_20250616_step144_roll4_float_1_5.json"
output_fp = "/code/All-In-One/qbw/EasyR1-20250410/rollout_results/agiqa3k_rollout_temp1/agiqa3k_qual_n16_continuous-thres0p75_format0p1_on-policy_newcode_20250616_step144_roll4_float_1_5_sft-rule-processed.json"

total = 0
processed = 0
with open(jsonl_fp, 'r') as f_in, open(output_fp, 'w') as f_out:
    items = json.load(f_in)
    for data in items:

        response = data['model_response']
        gt = data['mos_perception']

        total += 1
        try:
            pattern = r'<think>(.*?)</think>(.*)<answer>(.*?)</answer>'
            match = re.search(pattern, response, re.DOTALL)
            thinking = match.groups(1)[0]
            summary = match.groups(1)[1]
            score = match.groups(1)[2]

            score = float(score.strip())

            if abs(score - gt) < 1:
                out_item = {
                    "images": [data['image_path']],
                    "messages": [
                        {"role": "user", "content": "<image>" + data['messages'][0]['content'][1]['text']},
                        {"role": "assistant", "content": f"<think>{thinking}</think>{summary}<answer>{gt}</answer>"},
                    ]
                }
                f_out.write(json.dumps(out_item, ensure_ascii=False) + "\n")

                processed += 1
        except Exception as e:
            print(e)
            print(response)

print(total, processed)