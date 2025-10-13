from scipy import stats
import numpy as np
from scipy.optimize import curve_fit

import json
import re
import argparse

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

def main():
    parser = argparse.ArgumentParser(description="模型路径与名称配置")
    parser.add_argument(
        "--out_name",
        default="",
        help="模型输出文件主要名称",
    )
    parser.add_argument(
        "--vllm",
        action="store_true",
        help="是否使用vllm目录",
    )
    args = parser.parse_args()
    
    out_name = args.out_name
    dir_name = "agiqa-3k_vllm" if args.vllm else "agiqa-3k"
    suffix = "think-chat-template" if args.vllm else "newcode-0623"
    output_file = f"/code/All-In-One/qbw/EasyR1-20250410/eval_results/{dir_name}/{out_name}_float_1_5_{suffix}.json"

    y_label, y_out = [], []
    items = json.load(open(output_file))

    error_count = 0
    for i, item in enumerate(items):
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

    return output1, output2
    

if __name__ == "__main__":
    main()