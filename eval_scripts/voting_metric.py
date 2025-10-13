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
    args = parser.parse_args()
    
    out_name = args.out_name
    dir_name = "agiqa-3k_vllm_voting"
    output_file = f"/code/All-In-One/qbw/EasyR1-20250410/eval_results/{dir_name}/{out_name}_float_1_5_think-chat-template.json"

    y_label, y_out = [], []
    items = json.load(open(output_file))

    error_count = 0
    for i, item in enumerate(items):
        voting_result = item['voting_result']
        if voting_result is not None:
            out = float(voting_result)
            y_out.append(out)
            y_label.append(float(item['mos_perception']))
        else:
            error_count += 1
            print(f"{i}th error")
            
    print(error_count)
    output1 = performance_fit(y_label, y_out, func_fit=True)
    output2 = performance_fit(y_label, y_out, func_fit=False)

    print(output1)
    print(output2)

    return output1, output2
    

if __name__ == "__main__":
    main()