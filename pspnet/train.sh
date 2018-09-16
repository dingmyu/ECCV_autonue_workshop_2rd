#!/bin/sh
EXP_DIR=exp/voc2012/psp50
mkdir -p ${EXP_DIR}/model
now=$(date +"%Y%m%d_%H%M%S")
cp train.sh train.py ${EXP_DIR}

python train.py \
  --data_root=/mnt/sda1/hszhao/dataset/VOC2012 \
  --train_list=/mnt/sda1/hszhao/dataset/VOC2012/list/train.txt \
  --val_list=/mnt/sda1/hszhao/dataset/VOC2012/list/val.txt \
  --layers=50 \
  --syncbn=1 \
  --classes=21 \
  --crop_h=473 \
  --crop_w=473 \
  --zoom_factor=1 \
  --gpu 0 1 2 3 \
  --base_lr=1e-2 \
  --epochs=100 \
  --start_epoch=1 \
  --batch_size=16 \
  --save_path=${EXP_DIR}/model \
  --evaluate=0 \
  2>&1 | tee ${EXP_DIR}/model/train-$now.log
