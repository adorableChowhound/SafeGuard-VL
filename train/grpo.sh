#!/bin/bash
# Stage-2: Policy-Aware Reinforcement Learning (GRPO)
# Based on ms-swift (https://github.com/modelscope/ms-swift)
# Tested with ms-swift version from 2025.9.16

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
NPROC_PER_NODE=8 \
swift rlhf \
    --rlhf_type grpo \
    --model Qwen/Qwen2.5-VL-7B-Instruct \
    --external_plugins ./plugin.py \
    --reward_funcs external_r1v_acc format \
    --train_type full \
    --torch_dtype bfloat16 \
    --dataset /path/to/grpo_data.jsonl \
    --load_from_cache_file true \
    --max_completion_length 512 \
    --num_train_epochs 1 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --learning_rate 1e-6 \
    --gradient_accumulation_steps 2 \
    --save_strategy steps \
    --eval_strategy steps \
    --eval_steps 300 \
    --save_steps 300 \
    --save_total_limit 5 \
    --logging_steps 1 \
    --output_dir ./output/grpo \
    --warmup_ratio 0.01 \
    --dataloader_num_workers 4 \
    --num_generations 8 \
    --temperature 1.0 \
    --system "A conversation between User and Assistant. The user asks a question, and the Assistant solves it." \
    --deepspeed zero3 \
    --log_completions true \
    --num_iterations 1 \
    --async_generate false \
    --beta 0.01
