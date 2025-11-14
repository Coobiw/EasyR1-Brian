set -x

# MODEL_PATH="/code/All-In-One/qbw/EasyR1-20250410/cache/ckpt/Qwen2.5-VL-7B-Instruct"  # replace it with your local file path
MODEL_PATH="/code/All-In-One/qbw/LLaMA-Factory-20250504/saves/qwen2p5_vl-7b/full/ie4k_merge/checkpoint-279"


# DAPO Clip-Higher setting
# clip_ratio_low=0.2
# clip_ratio_high=0.28
# worker.actor.clip_ratio_low=${clip_ratio_low} \
# worker.actor.clip_ratio_high=${clip_ratio_high} \
# worker.actor.clip_ratio_dual=10.0 \

# DAPO Loss Aggregation
# worker.actor.loss_agg_mode="seq-mean-token-mean" \

# data.train_files="Coobiw/merged_agiqa5k_prompt_1022@train" \
# data.val_files="Coobiw/agiqa3k_prompt_1013@test" \

python3 -m verl.trainer.main \
    config=examples/config_agiqa3k_gaussian.yaml \
    data.train_files="Coobiw/IE-R1-4K@train" \
    data.val_files="Coobiw/IE-R1-4K@test" \
    worker.actor.global_batch_size=128 \
    worker.actor.model.model_path=${MODEL_PATH} \
    worker.actor.optim.lr=1e-6 \
    worker.actor.offload.offload_optimizer=True \
    worker.rollout.limit_images=2 \
    worker.rollout.n=8 \
    worker.rollout.val_override_config.top_p=0.9 \
    worker.rollout.val_override_config.temperature=1.0 \
    worker.reward.score_function_kwargs.format_weight=1.0 \
    worker.reward.score_function_kwargs.strict_format=False \
    worker.reward.score_function="./examples/score_function/agiqa3k.py:compute_score_gaussian" \
    algorithm.adv_estimator="grpo" \
    algorithm.keep_neg_ratio=1.0 \
    algorithm.keep_pos_ratio=1.0 \
    algorithm.compute_new_adv=False \
    trainer.save_freq=-1 \
    trainer.save_limit=60 \
    trainer.experiment_name=ie4k_n8_grpo-keepneg1_keeppos1_temp1_gaussian-default_format1_bs128-mbs128_onpolicy_kl0_chat-template_20251108_val-temp1-topp9_fix-nan2_mergesft_mergetrain \
    trainer.save_checkpoint_path="./cache/output/ie4k_n8_grpo-keepneg1_keeppos1_temp1_gaussian-default_format1_bs128-mbs128_onpolicy_kl0_chat-template_20251108_val-temp1-topp9_fix-nan2_mergesft_mergetrain" \
    trainer.total_episodes=10 \
    trainer.logger=['console','wandb'] \
    trainer.n_gpus_per_node=8