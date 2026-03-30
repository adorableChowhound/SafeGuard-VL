#!/bin/bash
# Stage-1: Supervised Fine-Tuning (SFT) for Unsafe Semantics Learning
# Based on ms-swift (https://github.com/modelscope/ms-swift)
# Tested with ms-swift version from 2025.9.16

NPROC_PER_NODE=8 \
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
swift sft \
    --model Qwen/Qwen2.5-VL-7B-Instruct \
    --dataset /path/to/recaptioned_data.json \
    --torch_dtype bfloat16 \
    --num_train_epochs 1 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --learning_rate 1e-4 \
    --weight_decay 1e-3 \
    --gradient_accumulation_steps 8 \
    --eval_steps 400 \
    --save_steps 800 \
    --save_total_limit 5 \
    --logging_steps 5 \
    --output_dir ./output/sft \
    --warmup_ratio 0.05 \
    --dataloader_num_workers 8 \
    --dataset_num_proc 8 \
    --train_type lora \
    --split_dataset_ratio 0.01 \
    --lora_rank 64 \
    --lora_alpha 128 \
    --gradient_checkpointing_kwargs '{"use_reentrant": false}' \
    --gradient_checkpointing True \
    --max_length 20480
