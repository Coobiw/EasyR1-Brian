import os
import json

with open("./cache/data/AGIQA-3k/annos/data.csv") as f, \
    open("./cache/data/AGIQA-3k/annos/data_jsonl.jsonl", 'w') as fo:
    lines = f.readlines()
    first_line = True
    for line in lines:
        if first_line:
            first_line = False
            continue
        line = line.strip()
        item = line.split(",")
        img_name, prompt, mos_perception, mos_align = item[0], item[1], float(item[-4]), float(item[-2])

        img_path = os.path.join("/code/All-In-One/qbw/EasyR1-20250410/cache/data/AGIQA-3k/images", img_name)
        assert os.path.exists(img_path)
        new_item = {
            "image": img_path,
            "prompt": prompt,
            "mos_perception": mos_perception,
            "mos_align": mos_align,
        }

        fo.write(f"{json.dumps(new_item, ensure_ascii=False)}\n")