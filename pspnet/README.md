# PSPNet



## Setup Environment

- Pytorch0.4
- torchE (default path in '/mnt/lustre/share/miniconda3/envs/r0.1.2/lib/python3.6/site-packages/torchE')

## Setup Repo
- git clone ＆＆ git checkout new_syncbn

## Train && Test
#### For cityscapes training
```shell
sh train_dist.sh bj11part 4 4
```
You can modify some parameters in bash script to satisify your setting.

#### For cityscapes test
```shell
sh eval_dist.sh bj11part 1 1 400
# use single GPU to evaluate the 400th epoch checkpoint.

sh eval_multiscale.sh bj11part 1 1 400
# for multiscale evaluation
```



## Performance

model | batch_size | epoch | mIoU | model
:----: | :---: |  :---: | :---: | :---:
Resnet-50 | 4x4 | 100 | 73.4 | [download](ftp://10.10.11.22/share/zhuxinge/psp_checkpoints/train_epoch_100.pth)
Resnet-50 | 4x4 | 200 | 74.6 | [download](ftp://10.10.11.22/share/zhuxinge/psp_checkpoints/train_epoch_200.pth)
Resnet-50 | 4x4 | 400 | 76.2 | [download](ftp://10.10.11.22/share/zhuxinge/psp_checkpoints/train_epoch_400.pth)

