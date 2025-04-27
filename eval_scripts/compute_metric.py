from scipy import stats
import numpy as np
from scipy.optimize import curve_fit

import json
import re

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


def performance_fit(y_label, y_output):
    y_output_logistic = fit_function(y_label, y_output)
    PLCC = stats.pearsonr(y_output_logistic, y_label)[0]
    SRCC = stats.spearmanr(y_output, y_label)[0]

    return PLCC, SRCC, (PLCC+SRCC) / 2

if __name__ == "__main__":
    # output_file = "/code/All-In-One/qbw/EasyR1-20250410/eval_results/agiqa-3k/Qwen2.5-VL-7B-Instruct_float_1_5.json"
    output_file = "/code/All-In-One/qbw/EasyR1-20250410/eval_results/agiqa-3k/Qwen2.5-VL-7B-Instruct_int_1_100.json"

    y_label, y_out = [], []
    items = json.load(open(output_file))

    error_count = 0
    for i, item in enumerate(items):
        model_response = item['model_response']
        try:
            answer_start = model_response.find("<answer>")
            answer_end = model_response.find("</answer>", answer_start + len("<answer>"))
            if answer_end == -1:
                model_response = model_response[answer_start+len("<answer>") : ]
            else:
                model_response = model_response[answer_start+len("<answer>") : answer_end]
                
            out = float(model_response.strip())
            y_out.append(out)
            y_label.append(float(item['mos_perception']))
        except Exception as e:
            error_count += 1
            print(f"{i}th error:\t", e)

    print(performance_fit(y_label, y_out))
    print(error_count)