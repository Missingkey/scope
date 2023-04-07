#!/usr/bin/env bash
# -*- coding: utf-8 -*-

REPO_PATH=/home/mdh19/004_csc_sl/SCOPE
BERT_PATH=/home/mdh19/004_csc_sl/SCOPE/FPT
DATA_DIR=$REPO_PATH/data
# TEXT='我想吃香交'
export PYTHONPATH="$PYTHONPATH:$REPO_PATH"

ckpt_path=outputs/bs32epoch30/checkpoint/epoch=29-df=79.2825-cf=77.6682.ckpt
# ckpt_path=/home/mdh19/004_csc_sl/ECSpell/Checkpoint/ecspell/results/checkpoint/rng_state_7.pth


OUTPUT_DIR=outputs/predict
mkdir -p $OUTPUT_DIR
# echo "请输入要改错的文本："
# read TEXT
# python /home/mdh19/test_projects/SCOPE/data_process/get_test_data_mdh.py \
#   --text $TEXT


CUDA_VISIBLE_DEVICES=5  python -u finetune/predict.py \
  --bert_path $BERT_PATH \
  --ckpt_path $ckpt_path \
  --data_dir $DATA_DIR \
  --save_path $OUTPUT_DIR \
  --label_file $DATA_DIR/test.sighan14.lbl.tsv \
  --gpus=0,
