set -x

MODEL_PATH="/code/All-In-One/qbw/EasyR1-20250410/cache/ckpt/Qwen2.5-VL-7B-Instruct"  # replace it with your local file path

python3 -m verl.trainer.main \
    config=examples/config_agiqa3k.yaml \
    data.train_files=Coobiw/agiqa3k_joint@train \
    data.val_files=Coobiw/agiqa3k_joint@test \
    worker.actor.model.model_path=${MODEL_PATH} \
    worker.reward.score_function_kwargs.threshold=0.35 \
    worker.rollout.n=8 \
    trainer.experiment_name=qwen2_5_vl_7b_agiqa3k_joint_n8_20250429 \
    trainer.save_checkpoint_path="./cache/output/agiqa3k_joint_n8_20250429/" \
    trainer.n_gpus_per_node=8
