#!/bin/sh
exp=res-psp50
EXP_DIR=exp/drivable/$exp
mkdir -p ${EXP_DIR}/result
now=$(date +"%Y%m%d_%H%M%S")
cp eval.sh eval.py ${EXP_DIR}

part=Segmentation1080
#part=Test
numGPU=1
MV2_USE_CUDA=1 MV2_ENABLE_AFFINITY=0 MV2_SMP_USE_CMA=0 srun --mpi=pmi2 --gres=gpu:$numGPU -n1 --ntasks-per-node=$numGPU --partition=$part --job-name=${exp} --kill-on-bad-exit=1 \
python eval.py \
  --data_root=/mnt/lustre/share/sunpeng/bdd/image/bdd100k \
  --val_list1=/mnt/lustre/share/sunpeng/bdd/image/bdd-data/bdd_data/lists/100k/drivable/freespace_bdd_val.txt \
  --split=val \
  --layers=50 \
  --classes=3 \
  --backbone=resnet \
  --net_type=0 \
  --crop_h=481 \
  --crop_w=481 \
  --zoom_factor=4 \
  --ignore_label=255 \
  --scales 0.5 0.75 1.0 1.25 1.5 1.75 \
  --has_prediction=0 \
  --gpu 0 \
  --model_path=exp/drivable/ibn101_b_psp/model/train_epoch_50.pth \
  --save_folder=${EXP_DIR}/result/epoch_50/val/ms \
  --colors_path=data/bdd/bdddrivablecolors.txt \
  --names_path=data/bdd/bdddrivablenames.txt \
  2>&1 | tee ${EXP_DIR}/result/epoch_50-val-ms-$now.log

# --scales 0.5 0.75 1.0 1.25 1.5 1.75 \
# --scales 1.0 \
