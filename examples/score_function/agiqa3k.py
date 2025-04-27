import re
from typing import Dict

def grade_answer(pred, gt, threshold):
    # print("threshold:\t", threshold)
    return (abs(float(pred) - float(gt)) < threshold)


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


def compute_score(predict_str: str, ground_truth: str, format_weight: float = 0.5, threshold: float = 0.35) -> Dict[str, float]:
    format_score = format_reward(predict_str)
    accuracy_score = accuracy_reward(predict_str, ground_truth, threshold)
    return {
        "overall": (1 - format_weight) * accuracy_score + format_weight * format_score,
        "format": format_score,
        "accuracy": accuracy_score,
    }
