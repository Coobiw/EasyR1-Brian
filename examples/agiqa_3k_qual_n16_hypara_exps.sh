set -x

MODEL_PATH="/code/All-In-One/qbw/EasyR1-20250410/cache/ckpt/Qwen2.5-VL-7B-Instruct"  # replace it with your local file path

# HIGH_TEMP_CKPT="/code/All-In-One/qbw/EasyR1-20250410/cache/output/agiqa3k_qual_n16_continuous-thres0p75_format0p1_on-policy_newcode_20250616/global_step_144/actor/huggingface"


# DAPO Clip-Higher setting
# clip_ratio_low=0.2
# clip_ratio_high=0.28
# worker.actor.clip_ratio_low=${clip_ratio_low} \
# worker.actor.clip_ratio_high=${clip_ratio_high} \
# worker.actor.clip_ratio_dual=10.0 \

# DAPO Loss Aggregation
# worker.actor.loss_agg_mode="seq-mean-token-mean" \

python3 -m verl.trainer.main \
    config=examples/config_agiqa3k.yaml \
    worker.actor.model.model_path=${MODEL_PATH} \
    trainer.save_freq=9 \
    trainer.save_limit=50 \
    trainer.experiment_name=agiqa3k_qual_n16_temp1_continuous-thres0p75_format0p1_on-policy_bs128_kl1e-2_newcode_20250620 \
    trainer.save_checkpoint_path="./cache/output/agiqa3k_qual_n16_temp1_continuous-thres0p75_format0p1_on-policy_bs128_kl1e-2_newcode_20250620/" \
    trainer.total_episodes=10 \
    trainer.logger=['console','swanlab'] \
    trainer.n_gpus_per_node=8
