#!/usr/bin/env bash
set -euo pipefail
cd /home/jaeung/activation-patching-unlearning
export CUDA_VISIBLE_DEVICES=0

echo '[mem] rmu_lr1e5_l15_s10_ep5'
python3 -m patchscope.memorization_eval --hf_dataset locuslab/TOFU --hf_config forget10_perturbed --model rmu_lr1e5_l15_s10_ep5 --batch_size 8 --max_length 512 --use_chat_template --out_dir runs/memorization_eval/alpha5_fix/rmu_lr1e5_l15_s10_ep5
echo '[privacy] rmu_lr1e5_l15_s10_ep5'
python3 -m patchscope.privacy_eval --model rmu_lr1e5_l15_s10_ep5 --reference_model retain --full_model full --attack all --k 0.4 --batch_size 8 --max_length 512 --use_chat_template --out_dir runs/privacy_eval/alpha5_fix/rmu_lr1e5_l15_s10_ep5
 
echo '[utility] rmu_lr1e5_l15_s10_ep5'
python3 -m patchscope.utility_eval --model rmu_lr1e5_l15_s10_ep5 --batch_size 8 --max_length 512 --max_new_tokens 200 --use_chat_template --out_dir runs/utility_eval/alpha5_fix/rmu_lr1e5_l15_s10_ep5

echo '[mem] rmu_lr1e5_l5_s10_ep5'
python3 -m patchscope.memorization_eval --hf_dataset locuslab/TOFU --hf_config forget10_perturbed --model rmu_lr1e5_l5_s10_ep5 --batch_size 8 --max_length 512 --use_chat_template --out_dir runs/memorization_eval/alpha5_fix/rmu_lr1e5_l5_s10_ep5
echo '[privacy] rmu_lr1e5_l5_s10_ep5'
python3 -m patchscope.privacy_eval --model rmu_lr1e5_l5_s10_ep5 --reference_model retain --full_model full --attack all --k 0.4 --batch_size 8 --max_length 512 --use_chat_template --out_dir runs/privacy_eval/alpha5_fix/rmu_lr1e5_l5_s10_ep5
echo '[utility] rmu_lr1e5_l5_s10_ep5'
python3 -m patchscope.utility_eval --model rmu_lr1e5_l5_s10_ep5 --batch_size 8 --max_length 512 --max_new_tokens 200 --use_chat_template --out_dir runs/utility_eval/alpha5_fix/rmu_lr1e5_l5_s10_ep5

echo '[mem] rmu_lr2e5_l10_s10_ep5'
python3 -m patchscope.memorization_eval --hf_dataset locuslab/TOFU --hf_config forget10_perturbed --model rmu_lr2e5_l10_s10_ep5 --batch_size 8 --max_length 512 --use_chat_template --out_dir runs/memorization_eval/alpha5_fix/rmu_lr2e5_l10_s10_ep5
echo '[privacy] rmu_lr2e5_l10_s10_ep5'
python3 -m patchscope.privacy_eval --model rmu_lr2e5_l10_s10_ep5 --reference_model retain --full_model full --attack all --k 0.4 --batch_size 8 --max_length 512 --use_chat_template --out_dir runs/privacy_eval/alpha5_fix/rmu_lr2e5_l10_s10_ep5
echo '[utility] rmu_lr2e5_l10_s10_ep5'
python3 -m patchscope.utility_eval --model rmu_lr2e5_l10_s10_ep5 --batch_size 8 --max_length 512 --max_new_tokens 200 --use_chat_template --out_dir runs/utility_eval/alpha5_fix/rmu_lr2e5_l10_s10_ep5

echo '[mem] rmu_lr2e5_l15_s10_ep5'
python3 -m patchscope.memorization_eval --hf_dataset locuslab/TOFU --hf_config forget10_perturbed --model rmu_lr2e5_l15_s10_ep5 --batch_size 8 --max_length 512 --use_chat_template --out_dir runs/memorization_eval/alpha5_fix/rmu_lr2e5_l15_s10_ep5
echo '[privacy] rmu_lr2e5_l15_s10_ep5'
python3 -m patchscope.privacy_eval --model rmu_lr2e5_l15_s10_ep5 --reference_model retain --full_model full --attack all --k 0.4 --batch_size 8 --max_length 512 --use_chat_template --out_dir runs/privacy_eval/alpha5_fix/rmu_lr2e5_l15_s10_ep5
echo '[utility] rmu_lr2e5_l15_s10_ep5'
python3 -m patchscope.utility_eval --model rmu_lr2e5_l15_s10_ep5 --batch_size 8 --max_length 512 --max_new_tokens 200 --use_chat_template --out_dir runs/utility_eval/alpha5_fix/rmu_lr2e5_l15_s10_ep5

echo '[mem] rmu_lr2e5_l5_s10_ep5'
python3 -m patchscope.memorization_eval --hf_dataset locuslab/TOFU --hf_config forget10_perturbed --model rmu_lr2e5_l5_s10_ep5 --batch_size 8 --max_length 512 --use_chat_template --out_dir runs/memorization_eval/alpha5_fix/rmu_lr2e5_l5_s10_ep5
echo '[privacy] rmu_lr2e5_l5_s10_ep5'
python3 -m patchscope.privacy_eval --model rmu_lr2e5_l5_s10_ep5 --reference_model retain --full_model full --attack all --k 0.4 --batch_size 8 --max_length 512 --use_chat_template --out_dir runs/privacy_eval/alpha5_fix/rmu_lr2e5_l5_s10_ep5
echo '[utility] rmu_lr2e5_l5_s10_ep5'
python3 -m patchscope.utility_eval --model rmu_lr2e5_l5_s10_ep5 --batch_size 8 --max_length 512 --max_new_tokens 200 --use_chat_template --out_dir runs/utility_eval/alpha5_fix/rmu_lr2e5_l5_s10_ep5

echo '[mem] rmu_lr5e5_l10_s10_ep5'
python3 -m patchscope.memorization_eval --hf_dataset locuslab/TOFU --hf_config forget10_perturbed --model rmu_lr5e5_l10_s10_ep5 --batch_size 8 --max_length 512 --use_chat_template --out_dir runs/memorization_eval/alpha5_fix/rmu_lr5e5_l10_s10_ep5
echo '[privacy] rmu_lr5e5_l10_s10_ep5'
python3 -m patchscope.privacy_eval --model rmu_lr5e5_l10_s10_ep5 --reference_model retain --full_model full --attack all --k 0.4 --batch_size 8 --max_length 512 --use_chat_template --out_dir runs/privacy_eval/alpha5_fix/rmu_lr5e5_l10_s10_ep5
echo '[utility] rmu_lr5e5_l10_s10_ep5'
python3 -m patchscope.utility_eval --model rmu_lr5e5_l10_s10_ep5 --batch_size 8 --max_length 512 --max_new_tokens 200 --use_chat_template --out_dir runs/utility_eval/alpha5_fix/rmu_lr5e5_l10_s10_ep5

echo '[mem] rmu_lr5e5_l15_s10_ep5'
python3 -m patchscope.memorization_eval --hf_dataset locuslab/TOFU --hf_config forget10_perturbed --model rmu_lr5e5_l15_s10_ep5 --batch_size 8 --max_length 512 --use_chat_template --out_dir runs/memorization_eval/alpha5_fix/rmu_lr5e5_l15_s10_ep5
echo '[privacy] rmu_lr5e5_l15_s10_ep5'
python3 -m patchscope.privacy_eval --model rmu_lr5e5_l15_s10_ep5 --reference_model retain --full_model full --attack all --k 0.4 --batch_size 8 --max_length 512 --use_chat_template --out_dir runs/privacy_eval/alpha5_fix/rmu_lr5e5_l15_s10_ep5
echo '[utility] rmu_lr5e5_l15_s10_ep5'
python3 -m patchscope.utility_eval --model rmu_lr5e5_l15_s10_ep5 --batch_size 8 --max_length 512 --max_new_tokens 200 --use_chat_template --out_dir runs/utility_eval/alpha5_fix/rmu_lr5e5_l15_s10_ep5
