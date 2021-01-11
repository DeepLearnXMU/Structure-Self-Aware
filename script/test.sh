#!/usr/bin/env bash
train_file="./Molweni/DP(500)/train.json"
eval_file="./Molweni/DP(500)/dev.json"
test_file="./Molweni/DP(500)/test.json"
glove_file="./glove.6B.200d.txt"
dataset_dir="./dataset"
model_dir="./model"

GPU=0
model_name=model
task=student
CUDA_VISIBLE_DEVICES=${GPU} nohup python -u main.py --train_file=$train_file --test_file=$test_file \
                                               --dataset_dir=$dataset_dir \
                                               --eval_pool_size 10 --glove_embedding_size 200 \
                                               --model_path "${model_dir}/${model_name}.pt" \
                                               --teacher_model_path "${model_dir}/teacher_model.pt" \
                                               --task ${task} > ${model_name}.log 2>&1 &
