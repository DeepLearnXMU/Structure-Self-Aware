#!/usr/bin/env bash
train_file="./Molweni/DP(500)/train.json"
eval_file="./Molweni/DP(500)/dev.json"
test_file="./Molweni/DP(500)/test.json"
glove_file="./glove.6B.200d.txt"
dataset_dir="./dataset"
model_dir="./model"

if [ ! -d "${model_dir}" ]; then mkdir -p "${model_dir}"; fi

GPU=0
model_name=model
task=student  # student, teacher or distill
CUDA_VISIBLE_DEVICES=${GPU} nohup python -u main.py --train_file=$train_file --eval_file=$eval_file --test_file=$test_file \
                                    --dataset_dir=$dataset_dir --glove_vocab_path $glove_file \
                                    --epoches 30 --batch_size 100 --pool_size 100 \
                                    --eval_pool_size 10 --report_step 30 \
                                    --save_model --overwrite --do_train \
                                    --model_path "${model_dir}/${model_name}.pt" \
                                    --teacher_model_path "${model_dir}/teacher_model.pt" \
                                    --learning_rate 0.1 --glove_embedding_size 200 \
                                    --remake_dataset --remake_tokenizer \
                                    --task ${task} > ${model_name}.log 2>&1 &