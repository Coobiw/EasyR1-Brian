THIS_CUDA=$1
MODEL_PATH=$2
MODEL_NAME=$3

CUDA_VISIBLE_DEVICES=$THIS_CUDA python eval_scripts/agiqa3k_eval.py --model_path $MODEL_PATH --model_name $MODEL_NAME