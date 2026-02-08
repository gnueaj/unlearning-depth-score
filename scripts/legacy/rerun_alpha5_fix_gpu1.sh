#!/usr/bin/env bash
set -euo pipefail
cd /home/jaeung/activation-patching-unlearning
export CUDA_VISIBLE_DEVICES=1

echo '[mem] rmu_lr5e5_l5_s10_ep5'
python3 -m patchscope.memorization_eval --hf_dataset locuslab/TOFU --hf_config forget10_perturbed --model rmu_lr5e5_l5_s10_ep5 --batch_size 8 --max_length 512 --use_chat_template --out_dir runs/memorization_eval/alpha5_fix/rmu_lr5e5_l5_s10_ep5
echo '[privacy] rmu_lr5e5_l5_s10_ep5'
python3 -m patchscope.privacy_eval --model rmu_lr5e5_l5_s10_ep5 --reference_model retain --full_model full --attack all --k 0.4 --batch_size 8 --max_length 512 --use_chat_template --out_dir runs/privacy_eval/alpha5_fix/rmu_lr5e5_l5_s10_ep5
echo '[utility] rmu_lr5e5_l5_s10_ep5'
python3 -m patchscope.utility_eval --model rmu_lr5e5_l5_s10_ep5 --batch_size 8 --max_length 512 --max_new_tokens 200 --use_chat_template --out_dir runs/utility_eval/alpha5_fix/rmu_lr5e5_l5_s10_ep5

echo '[mem] simnpo_lr1e5_b35_a1_d1_g0125_ep5'
python3 -m patchscope.memorization_eval --hf_dataset locuslab/TOFU --hf_config forget10_perturbed --model simnpo_lr1e5_b35_a1_d1_g0125_ep5 --batch_size 8 --max_length 512 --use_chat_template --out_dir runs/memorization_eval/alpha5_fix/simnpo_lr1e5_b35_a1_d1_g0125_ep5
echo '[privacy] simnpo_lr1e5_b35_a1_d1_g0125_ep5'
python3 -m patchscope.privacy_eval --model simnpo_lr1e5_b35_a1_d1_g0125_ep5 --reference_model retain --full_model full --attack all --k 0.4 --batch_size 8 --max_length 512 --use_chat_template --out_dir runs/privacy_eval/alpha5_fix/simnpo_lr1e5_b35_a1_d1_g0125_ep5
echo '[utility] simnpo_lr1e5_b35_a1_d1_g0125_ep5'
python3 -m patchscope.utility_eval --model simnpo_lr1e5_b35_a1_d1_g0125_ep5 --batch_size 8 --max_length 512 --max_new_tokens 200 --use_chat_template --out_dir runs/utility_eval/alpha5_fix/simnpo_lr1e5_b35_a1_d1_g0125_ep5

echo '[mem] simnpo_lr2e5_b35_a1_d1_g0125_ep5'
python3 -m patchscope.memorization_eval --hf_dataset locuslab/TOFU --hf_config forget10_perturbed --model simnpo_lr2e5_b35_a1_d1_g0125_ep5 --batch_size 8 --max_length 512 --use_chat_template --out_dir runs/memorization_eval/alpha5_fix/simnpo_lr2e5_b35_a1_d1_g0125_ep5
echo '[privacy] simnpo_lr2e5_b35_a1_d1_g0125_ep5'
python3 -m patchscope.privacy_eval --model simnpo_lr2e5_b35_a1_d1_g0125_ep5 --reference_model retain --full_model full --attack all --k 0.4 --batch_size 8 --max_length 512 --use_chat_template --out_dir runs/privacy_eval/alpha5_fix/simnpo_lr2e5_b35_a1_d1_g0125_ep5
echo '[utility] simnpo_lr2e5_b35_a1_d1_g0125_ep5'
python3 -m patchscope.utility_eval --model simnpo_lr2e5_b35_a1_d1_g0125_ep5 --batch_size 8 --max_length 512 --max_new_tokens 200 --use_chat_template --out_dir runs/utility_eval/alpha5_fix/simnpo_lr2e5_b35_a1_d1_g0125_ep5

echo '[mem] simnpo_lr5e5_b35_a1_d1_g0125_ep5'
python3 -m patchscope.memorization_eval --hf_dataset locuslab/TOFU --hf_config forget10_perturbed --model simnpo_lr5e5_b35_a1_d1_g0125_ep5 --batch_size 8 --max_length 512 --use_chat_template --out_dir runs/memorization_eval/alpha5_fix/simnpo_lr5e5_b35_a1_d1_g0125_ep5
echo '[privacy] simnpo_lr5e5_b35_a1_d1_g0125_ep5'
python3 -m patchscope.privacy_eval --model simnpo_lr5e5_b35_a1_d1_g0125_ep5 --reference_model retain --full_model full --attack all --k 0.4 --batch_size 8 --max_length 512 --use_chat_template --out_dir runs/privacy_eval/alpha5_fix/simnpo_lr5e5_b35_a1_d1_g0125_ep5
echo '[utility] simnpo_lr5e5_b35_a1_d1_g0125_ep5'
python3 -m patchscope.utility_eval --model simnpo_lr5e5_b35_a1_d1_g0125_ep5 --batch_size 8 --max_length 512 --max_new_tokens 200 --use_chat_template --out_dir runs/utility_eval/alpha5_fix/simnpo_lr5e5_b35_a1_d1_g0125_ep5

echo '[mem] undial_lr1e4_b10_a5_ep5'
python3 -m patchscope.memorization_eval --hf_dataset locuslab/TOFU --hf_config forget10_perturbed --model undial_lr1e4_b10_a5_ep5 --batch_size 8 --max_length 512 --use_chat_template --out_dir runs/memorization_eval/alpha5_fix/undial_lr1e4_b10_a5_ep5
echo '[privacy] undial_lr1e4_b10_a5_ep5'
python3 -m patchscope.privacy_eval --model undial_lr1e4_b10_a5_ep5 --reference_model retain --full_model full --attack all --k 0.4 --batch_size 8 --max_length 512 --use_chat_template --out_dir runs/privacy_eval/alpha5_fix/undial_lr1e4_b10_a5_ep5
echo '[utility] undial_lr1e4_b10_a5_ep5'
python3 -m patchscope.utility_eval --model undial_lr1e4_b10_a5_ep5 --batch_size 8 --max_length 512 --max_new_tokens 200 --use_chat_template --out_dir runs/utility_eval/alpha5_fix/undial_lr1e4_b10_a5_ep5

echo '[mem] undial_lr1e5_b10_a5_ep5'
python3 -m patchscope.memorization_eval --hf_dataset locuslab/TOFU --hf_config forget10_perturbed --model undial_lr1e5_b10_a5_ep5 --batch_size 8 --max_length 512 --use_chat_template --out_dir runs/memorization_eval/alpha5_fix/undial_lr1e5_b10_a5_ep5
echo '[privacy] undial_lr1e5_b10_a5_ep5'
python3 -m patchscope.privacy_eval --model undial_lr1e5_b10_a5_ep5 --reference_model retain --full_model full --attack all --k 0.4 --batch_size 8 --max_length 512 --use_chat_template --out_dir runs/privacy_eval/alpha5_fix/undial_lr1e5_b10_a5_ep5
echo '[utility] undial_lr1e5_b10_a5_ep5'
python3 -m patchscope.utility_eval --model undial_lr1e5_b10_a5_ep5 --batch_size 8 --max_length 512 --max_new_tokens 200 --use_chat_template --out_dir runs/utility_eval/alpha5_fix/undial_lr1e5_b10_a5_ep5

echo '[mem] undial_lr3e4_b10_a5_ep5'
python3 -m patchscope.memorization_eval --hf_dataset locuslab/TOFU --hf_config forget10_perturbed --model undial_lr3e4_b10_a5_ep5 --batch_size 8 --max_length 512 --use_chat_template --out_dir runs/memorization_eval/alpha5_fix/undial_lr3e4_b10_a5_ep5
echo '[privacy] undial_lr3e4_b10_a5_ep5'
python3 -m patchscope.privacy_eval --model undial_lr3e4_b10_a5_ep5 --reference_model retain --full_model full --attack all --k 0.4 --batch_size 8 --max_length 512 --use_chat_template --out_dir runs/privacy_eval/alpha5_fix/undial_lr3e4_b10_a5_ep5
echo '[utility] undial_lr3e4_b10_a5_ep5'
python3 -m patchscope.utility_eval --model undial_lr3e4_b10_a5_ep5 --batch_size 8 --max_length 512 --max_new_tokens 200 --use_chat_template --out_dir runs/utility_eval/alpha5_fix/undial_lr3e4_b10_a5_ep5
