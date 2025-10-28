# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Any, Dict, List

import numpy as np
import torch
from scipy import stats
from scipy.optimize import curve_fit

from ..protocol import DataProto

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

def quality_assessment_metrics(y_out: List[str], y_label: List[str]):
    plcc, srcc, main_score = performance_fit(y_label, y_out, func_fit=True)
    return plcc, srcc, main_score
    
def reduce_metrics(metrics: Dict[str, List[Any]]) -> Dict[str, Any]:
    return {key: np.mean(value) for key, value in metrics.items()}


def compute_non_cutoff_response_length_metrics(
    response_length: torch.Tensor, max_response_length: int
) -> Dict[str, Any]:
    """
    Compute response length metrics excluding samples that hit the cutoff (max_response_length).
    This is useful for tracking the evolution of response length without being skewed by cutoff samples.
    
    Args:
        response_length: Tensor of response lengths, shape (batch_size,)
        max_response_length: Maximum allowed response length
    
    Returns:
        Dictionary with metrics for non-cutoff samples
    """
    # Find samples that are NOT cutoff (response_length < max_response_length)
    non_cutoff_mask = response_length < max_response_length
    non_cutoff_lengths = response_length[non_cutoff_mask]
    
    # If no non-cutoff samples, return zeros
    if non_cutoff_lengths.numel() == 0:
        return {
            "response_length_no_cutoff/mean": 0.0,
            "response_length_no_cutoff/max": 0.0,
            "response_length_no_cutoff/min": 0.0,
            "response_length_no_cutoff/count": 0,
        }
    
    return {
        "response_length_no_cutoff/mean": torch.mean(non_cutoff_lengths.float()).detach().item(),
        "response_length_no_cutoff/max": torch.max(non_cutoff_lengths).detach().item(),
        "response_length_no_cutoff/min": torch.min(non_cutoff_lengths).detach().item(),
        "response_length_no_cutoff/count": non_cutoff_lengths.numel(),
    }


def compute_data_metrics(batch: DataProto, use_critic: bool = False) -> Dict[str, Any]:
    sequence_score = batch.batch["token_level_scores"].sum(-1)
    sequence_reward = batch.batch["token_level_rewards"].sum(-1)

    advantages = batch.batch["advantages"]
    returns = batch.batch["returns"]
    
    # For GRPO & BW-GRPO (with keep_neg_ratio < 1.0), use original advantages for metrics logging (to compare with other jobs)
    # The actual training uses filtered advantages stored in batch["advantages"]
    advantages_for_metrics = batch.batch.get("advantages_original", advantages)

    max_response_length = batch.batch["responses"].size(-1)

    prompt_mask = batch.batch["attention_mask"][:, :-max_response_length].bool()
    response_mask = batch.batch["attention_mask"][:, -max_response_length:].bool()

    max_prompt_length = prompt_mask.size(-1)
    prompt_length = prompt_mask.sum(-1).float()
    response_length = response_mask.sum(-1).float()

    valid_adv = torch.masked_select(advantages_for_metrics, response_mask)
    valid_returns = torch.masked_select(returns, response_mask)
    
    # For GRPO & BW-GRPO (with keep_neg_ratio < 1.0), also compute metrics for their filtered advantages
    additional_metrics = {}
    if "advantages_original" in batch.batch:
        # This means we're using GRPO or BW-GRPO with keep_neg_ratio < 1.0
        # batch["advantages"] contains algorithm-specific advantages (used for training, with samples filtered)
        valid_adv_specific = torch.masked_select(advantages, response_mask)
        
        # Compute metrics for filtered advantages
        # Note: metric name keeps "bw-grpo" for backward compatibility, but applies to both GRPO and BW-GRPO
        additional_metrics = {
            "critic/advantages_processed/mean": torch.mean(valid_adv_specific).detach().item(),
            "critic/advantages_processed/max": torch.max(valid_adv_specific).detach().item(),
            "critic/advantages_processed/min": torch.min(valid_adv_specific).detach().item(),
        }

    if use_critic:
        values = batch.batch["values"]
        valid_values = torch.masked_select(values, response_mask)
        return_diff_var = torch.var(valid_returns - valid_values)
        return_var = torch.var(valid_returns)

    metrics = {
        # score
        "critic/score/mean": torch.mean(sequence_score).detach().item(),
        "critic/score/max": torch.max(sequence_score).detach().item(),
        "critic/score/min": torch.min(sequence_score).detach().item(),
        # reward
        "critic/rewards/mean": torch.mean(sequence_reward).detach().item(),
        "critic/rewards/max": torch.max(sequence_reward).detach().item(),
        "critic/rewards/min": torch.min(sequence_reward).detach().item(),
        # adv
        "critic/advantages/mean": torch.mean(valid_adv).detach().item(),
        "critic/advantages/max": torch.max(valid_adv).detach().item(),
        "critic/advantages/min": torch.min(valid_adv).detach().item(),
        # returns
        "critic/returns/mean": torch.mean(valid_returns).detach().item(),
        "critic/returns/max": torch.max(valid_returns).detach().item(),
        "critic/returns/min": torch.min(valid_returns).detach().item(),
        **(
            {
                # values
                "critic/values/mean": torch.mean(valid_values).detach().item(),
                "critic/values/max": torch.max(valid_values).detach().item(),
                "critic/values/min": torch.min(valid_values).detach().item(),
                # vf explained var
                "critic/vf_explained_var": (1.0 - return_diff_var / (return_var + 1e-5)).detach().item(),
            }
            if use_critic
            else {}
        ),
        # response length (all samples)
        "response_length/mean": torch.mean(response_length).detach().item(),
        "response_length/max": torch.max(response_length).detach().item(),
        "response_length/min": torch.min(response_length).detach().item(),
        "response_length/clip_ratio": torch.mean(torch.eq(response_length, max_response_length).float())
        .detach()
        .item(),
        # response length (excluding cutoff samples) - for plotting evolution curve
        **compute_non_cutoff_response_length_metrics(response_length, max_response_length),
        # prompt length
        "prompt_length/mean": torch.mean(prompt_length).detach().item(),
        "prompt_length/max": torch.max(prompt_length).detach().item(),
        "prompt_length/min": torch.min(prompt_length).detach().item(),
        "prompt_length/clip_ratio": torch.mean(torch.eq(prompt_length, max_prompt_length).float()).detach().item(),
        # GRPO & BW-GRPO filtered advantages metrics (when keep_neg_ratio < 1.0)
        **additional_metrics,
    }
    return metrics


def compute_timing_metrics(batch: DataProto, timing_raw: Dict[str, float]) -> Dict[str, Any]:
    num_response_tokens = torch.sum(batch.batch["response_mask"]).item()
    num_overall_tokens = sum(batch.meta_info["global_token_num"])
    num_tokens_of_section = {
        **dict.fromkeys(["gen", "reward"], num_response_tokens),
        **dict.fromkeys(["ref", "old", "values", "adv", "update_critic", "update_actor"], num_overall_tokens),
    }
    return {
        **{f"timing_s/{name}": value for name, value in timing_raw.items()},
        **{
            f"timing_per_token_ms/{name}": timing_raw[name] * 1000 / num_tokens_of_section[name]
            for name in set(num_tokens_of_section.keys()) & set(timing_raw.keys())
        },
    }


def compute_throughout_metrics(batch: DataProto, timing_raw: Dict[str, float], num_gpus: int) -> Dict[str, Any]:
    total_num_tokens = sum(batch.meta_info["global_token_num"])
    time = timing_raw["step"]
    return {
        "perf/total_num_tokens": total_num_tokens,
        "perf/time_per_step": time,
        "perf/throughput": total_num_tokens / (time * num_gpus),
    }
