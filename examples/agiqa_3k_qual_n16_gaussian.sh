set -x

MODEL_PATH="/code/All-In-One/qbw/EasyR1-20250410/cache/ckpt/Qwen2.5-VL-7B-Instruct"  # replace it with your local file path


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
    worker.actor.global_batch_size=128 \
    worker.actor.model.model_path=${MODEL_PATH} \
    worker.rollout.val_override_config.temperature=1.0 \
    worker.reward.score_function_kwargs.format_weight=0.5 \
    worker.reward.score_function_kwargs.strict_format=False \
    worker.reward.score_function_kwargs.diff_at_rmin=4.0 \
    trainer.save_freq=18 \
    trainer.save_limit=60 \
    trainer.experiment_name=agiqa3k_qual_n16_temp1_gaussian-rmin-diff4_format0p5_bs128-mbs64_kl0_chat-template_20251021_on-policy_val-temp1_promptdataset \
    trainer.save_checkpoint_path="./cache/output/agiqa3k_qual_n16_temp1_gaussian-rmin-diff4_format0p5_bs128-mbs64_kl0_chat-template_20251021_on-policy_val-temp1_promptdataset/" \
    trainer.total_episodes=10 \
    trainer.logger=['console','wandb'] \
    trainer.n_gpus_per_node=8
