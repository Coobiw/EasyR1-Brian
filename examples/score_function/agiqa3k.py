import re
from typing import Dict
import math

def grade_answer(pred, gt, threshold):
    # print("threshold:\t", threshold)
    return (abs(float(pred) - float(gt)) < threshold)

def grade_answer_continuous(pred, gt, threshold):
    diff = abs(float(pred) - float(gt))

    if diff < threshold:
        return 1 - diff
    else:
        return 0.

def grade_answer_gaussian(
    pred,
    gt,
    r_min: float = 0.05,     # 在 diff=1 处的目标最小奖励
    diff_at_rmin: float = 1.0,  # “相差多少分”时衰减到 r_min（原始分值尺度）
    use_floor: bool = True,
) -> float:
    # 归一误差到[0,1]，gt/pred在[1,5]
    d = abs(float(pred) - float(gt)) / 4.0
    d0 = diff_at_rmin / 4.0

    # 数值保护
    r_min_clamped = max(min(float(r_min), 0.999999), 1e-6)

    # 由 r(d0)=r_min 反推 sigma（高斯）
    sigma = d0 / math.sqrt(2.0 * math.log(1.0 / r_min_clamped))

    # 高斯衰减（标量）
    r = math.exp(- (d ** 2) / (2.0 * sigma ** 2))

    # 地板（避免稀疏）
    if use_floor:
        r = max(r, r_min_clamped)

    return float(r)
            


def format_reward(predict_str: str) -> float:
    pattern = re.compile(r"<think>.*?</think>\s*<answer>.*?</answer>", re.DOTALL)
    format_match = re.fullmatch(pattern, predict_str)
    return 1.0 if format_match else 0.0


def accuracy_reward(predict_str: str, ground_truth: str, threshold: float) -> float:
    try:
        content_match = re.search(r"<answer>(.*?)</answer>", predict_str)
        given_answer = content_match.group(1).strip() if content_match else predict_str.strip()
        if isinstance(ground_truth, float) or isinstance(ground_truth, int):
            ground_truth = str(ground_truth)
        if grade_answer(given_answer, ground_truth.strip(), threshold):
            return 1.0

    except Exception as e:
        # print(e)
        pass

    return 0.0

def accuracy_reward_continuous(predict_str: str, ground_truth: str, threshold: float) -> float:
    try:
        content_match = re.search(r"<answer>(.*?)</answer>", predict_str)
        given_answer = content_match.group(1).strip() if content_match else predict_str.strip()
        if isinstance(ground_truth, float) or isinstance(ground_truth, int):
            ground_truth = str(ground_truth)
        return grade_answer_continuous(given_answer, ground_truth.strip(), threshold)

    except Exception as e:
        # print(e)
        pass

    return 0.0

def accuracy_reward_gaussian(
    predict_str: str,
    ground_truth: str,
    r_min=0.05,  # 奖励地板；“diff=1”处的目标值也就是 r_min
    diff_at_rmin=1.0,   # 在“相差多少分”时把奖励压到 r_min；默认 1 分
    use_floor=True,     # 是否启用地板裁剪
) -> float:
    try:
        content_match = re.search(r"<answer>(.*?)</answer>", predict_str)
        given_answer = content_match.group(1).strip() if content_match else predict_str.strip()
        if isinstance(ground_truth, float) or isinstance(ground_truth, int):
            ground_truth = str(ground_truth)
        return grade_answer_gaussian(given_answer, ground_truth.strip(), r_min, diff_at_rmin, use_floor)

    except Exception as e:
        # print(e)
        pass

    return 0.0

def compute_score(predict_str: str, ground_truth: str, format_weight: float = 0.5, threshold: float = 0.35) -> Dict[str, float]:
    format_score = format_reward(predict_str)
    accuracy_score = accuracy_reward(predict_str, ground_truth, threshold)
    return {
        "overall": (1 - format_weight) * accuracy_score + format_weight * format_score,
        "format": format_score,
        "accuracy": accuracy_score,
    }

def compute_score_continuous(predict_str: str, ground_truth: str, format_weight: float = 0.5, threshold: float = 0.35) -> Dict[str, float]:
    format_score = format_reward(predict_str)
    accuracy_score = accuracy_reward_continuous(predict_str, ground_truth, threshold)
    return {
        "overall": (1 - format_weight) * accuracy_score + format_weight * format_score,
        "format": format_score,
        "accuracy": accuracy_score,
    }

def compute_score_gaussian(
    predict_str: str,
    ground_truth: str,
    format_weight: float = 0.5,
    r_min=0.05,  # 奖励地板；“diff=1”处的目标值也就是 r_min
    diff_at_rmin=1.0,   # 在“相差多少分”时把奖励压到 r_min；默认 1 分
    use_floor=True,     # 是否启用地板裁剪
)-> Dict[str, float]:
    format_score = format_reward(predict_str)
    accuracy_score = accuracy_reward_gaussian(predict_str, ground_truth, r_min, diff_at_rmin, use_floor)
    return {
        "overall": (1 - format_weight) * accuracy_score + format_weight * format_score,
        "format": format_score,
        "accuracy": accuracy_score,
    }