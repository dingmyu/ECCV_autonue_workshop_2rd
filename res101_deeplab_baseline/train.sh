#!/bin/sh
exp=res50_psp
EXP_DIR=exp/drivable/$exp
mkdir -p ${EXP_DIR}/model
now=$(date +"%Y%m%d_%H%M%S")
cp train.sh train.py ${EXP_DIR}
#part=Segmentation
part=Segmentation
numGPU=4
nodeGPU=2
GLOG_vmodule=MemcachedClient=-1 srun --mpi=pmi2 -p $part --gres=gpu:$nodeGPU -n$numGPU --ntasks-per-node=$nodeGPU --job-name=${exp} \
python -u train.py \
  --data_root= \
  --train_list=/mnt/lustre/sunpeng/Research/image-base-workshop/anue/lists/train_list.txt \
  --val_list=/mnt/lustre/sunpeng/Research/image-base-workshop/anue/lists/val_list.txt \
  --layers=101 \
  --backbone=resnet \
  --net_type=0 \
  --port=12345 \
  --syncbn=1 \
  --classes=26 \
  --crop_h=881 \
  --crop_w=881 \
  --zoom_factor=2 \
  --base_lr=1e-2 \
  --epochs=200 \
  --start_epoch=1 \
  --batch_size=1 \
  --bn_group=4 \
  --save_step=5 \
  --save_path=${EXP_DIR}/model \
  --evaluate=0 \
  --ignore_label 255 \
  --workers 2 \
  2>&1 | tee ${EXP_DIR}/model/train-$now.log
