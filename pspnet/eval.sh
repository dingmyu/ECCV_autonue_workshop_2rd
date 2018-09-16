#!/bin/sh
EXP_DIR=exp/voc2012/psp50
mkdir -p ${EXP_DIR}/result
now=$(date +"%Y%m%d_%H%M%S")
cp eval.sh eval.py ${EXP_DIR}

python eval.py \
  --data_root=/mnt/sda1/hszhao/dataset/VOC2012 \
  --val_list=/mnt/sda1/hszhao/dataset/VOC2012/list/val.txt \
  --split=val \
  --layers=50 \
  --classes=21 \
  --base_size=512 \
  --crop_h=473 \
  --crop_w=473 \
  --zoom_factor=1 \
  --ignore_label=255 \
  --scales 1.0 \
  --has_prediction=0 \
  --gpu 0 \
  --model_path=${EXP_DIR}/model/train_epoch_100.pth \
  --save_folder=${EXP_DIR}/result/epoch_100/val/ss \
  --colors_path=data/voc2012/voc2012colors.mat \
  --names_path=data/voc2012/voc2012names.mat \
  2>&1 | tee ${EXP_DIR}/result/epoch_100-val-ss-$now.log

# --scales 0.5 0.75 1.0 1.25 1.5 1.75 \
