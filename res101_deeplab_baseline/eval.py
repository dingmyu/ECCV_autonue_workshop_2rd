import os
import cv2
import time
import logging
from argparse import ArgumentParser
import numpy as np

import torch
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torch.nn.functional as F

import segdata as datasets
import segtransforms as transforms
# from pspnet import PSPNet
from utils import AverageMeter, intersectionAndUnion, check_makedirs, colorize
cv2.ocl.setUseOpenCL(False)


# Setup
def get_parser():
    parser = ArgumentParser(description='PyTorch Semantic Segmentation Evaluation')
    parser.add_argument('--data_root', type=str, default='/mnt/sda1/hszhao/dataset/VOC2012', help='data root')
    parser.add_argument('--val_list1', type=str, default='/mnt/sda1/hszhao/dataset/VOC2012/list/val.txt', help='val list')
    parser.add_argument('--split', type=str, default='val', help='split in [train, val and test]')
    parser.add_argument('--backbone', type=str, default='resnet', help='backbone network type')
    parser.add_argument('--net_type', type=int, default=0, help='0-single branch, 1-div4 branch')
    parser.add_argument('--layers', type=int, default=50, help='layers number of based resnet')
    parser.add_argument('--classes', type=int, default=21, help='number of classes')
    parser.add_argument('--base_size1', type=int, default=512, help='based size for scaling')
    parser.add_argument('--crop_h', type=int, default=473, help='validation crop size h')
    parser.add_argument('--crop_w', type=int, default=473, help='validation crop size w')
    parser.add_argument('--zoom_factor', type=int, default=1, help='zoom factor in final prediction map')
    parser.add_argument('--ignore_label', type=int, default=255, help='ignore label in ground truth')
    parser.add_argument('--scales', type=float, default=[1.0], nargs='+', help='evaluation scales')
    parser.add_argument('--has_prediction', type=int, default=0, help='has prediction already or not')

    parser.add_argument('--gpu', type=int, default=[0], nargs='+', help='used gpu')
    parser.add_argument('--workers', type=int, default=1, help='data loader workers')
    parser.add_argument('--model_path', type=str, default='exp/voc2012/psp50/model/train_epoch_100.pth', help='evaluation model path')
    parser.add_argument('--save_folder', type=str, default='exp/voc2012/psp50/result/epoch_100/val/ss', help='results save folder')
    parser.add_argument('--colors_path', type=str, default='data/voc2012/voc2012colors.txt', help='path of dataset colors')
    parser.add_argument('--names_path', type=str, default='data/voc2012/voc2012names.txt', help='path of dataset category names')
    return parser


# logger
def get_logger():
    logger_name = "main-logger"
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler()
    fmt = "[%(asctime)s %(levelname)s %(filename)s line %(lineno)d %(process)d] %(message)s"
    handler.setFormatter(logging.Formatter(fmt))
    logger.addHandler(handler)
    return logger


def main():
    global args, logger
    args = get_parser().parse_args()
    logger = get_logger()
    # os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(str(x) for x in args.gpu)
    logger.info(args)
    assert args.classes > 1
    assert args.zoom_factor in [1, 2, 4, 8]
    assert (args.crop_h - 1) % 8 == 0 and (args.crop_w - 1) % 8 == 0
    assert args.split in ['train', 'val', 'test']
    logger.info("=> creating model ...")
    logger.info("Classes: {}".format(args.classes))

    value_scale = 255
    mean = [0.485, 0.456, 0.406]
    mean = [item * value_scale for item in mean]
    std = [0.229, 0.224, 0.225]
    std = [item * value_scale for item in std]

    gray_folder = os.path.join(args.save_folder, 'gray')
    color_folder = os.path.join(args.save_folder, 'color')

    val_transform = transforms.Compose([transforms.ToTensor()])
    val_data1 = datasets.SegData(split=args.split, data_root=args.data_root, data_list=args.val_list1, transform=val_transform)
    val_loader1 = torch.utils.data.DataLoader(val_data1, batch_size=1, shuffle=False, num_workers=args.workers, pin_memory=True)
    colors = np.loadtxt(args.colors_path).astype('uint8')
    names = [line.rstrip('\n') for line in open(args.names_path)]

    if not args.has_prediction:
        if args.net_type == 0:
            from pspnet import PSPNet
            model = PSPNet(backbone = args.backbone, layers=args.layers, classes=args.classes, zoom_factor=args.zoom_factor, use_softmax=True, use_aux=False, pretrained=False, syncbn=False).cuda()
        elif  args.net_type in [1, 2, 3]:
            from pspnet_div4 import PSPNet
            model = PSPNet(backbone = args.backbone, layers=args.layers, classes=args.classes, zoom_factor=args.zoom_factor, use_softmax=True, use_aux=False, pretrained=False, syncbn=False, net_type=args.net_type).cuda()
        logger.info(model)
        model = torch.nn.DataParallel(model).cuda()
        cudnn.enabled = True
        cudnn.benchmark = True
        if os.path.isfile(args.model_path):
            logger.info("=> loading checkpoint '{}'".format(args.model_path))
            checkpoint = torch.load(args.model_path)
            model.load_state_dict(checkpoint['state_dict'], strict=False)
            logger.info("=> loaded checkpoint '{}'".format(args.model_path))
        else:
            raise RuntimeError("=> no checkpoint found at '{}'".format(args.model_path))
        cv2.setNumThreads(0)
        mIoUs = []
        mAccs = []
        allAccs = []
        validate(val_loader1, val_data1.data_list, model, args.classes, mean, std, args.base_size1, args.crop_h, args.crop_w, args.scales, gray_folder, color_folder, colors)
        if args.split != 'test':
            mIoU, mAcc, allAcc = cal_acc(val_data1.data_list, gray_folder, args.classes, names)
            mIoUs.append(mIoU)
            mAccs.append(mAcc)
            allAccs.append(allAcc)

def net_process(model, image, mean, std=None):
    input = torch.from_numpy(image.transpose((2, 0, 1))).float()
    if std is None:
        for t, m in zip(input, mean):
            t.sub_(m)
    else:
        for t, m, s in zip(input, mean, std):
            t.sub_(m).div_(s)
    input = input.contiguous()
    input = input.unsqueeze(0).cuda(async=True)
    input_var = torch.autograd.Variable(input)
    output = model(input_var)
    h, w, _ = image.shape
    output = F.upsample(output, (h, w), mode='bilinear')
    output = output.squeeze(0).data.cpu().numpy()
    output = output.transpose(1, 2, 0)
    return output


def scale_process(model, image, classes, crop_h, crop_w, h, w, mean, std=None, stride_rate=2/3):
    ori_h, ori_w, _ = image.shape
    pad_h = max(crop_h - ori_h, 0)
    pad_w = max(crop_w - ori_w, 0)
    pad_h_half = int(pad_h / 2)
    pad_w_half = int(pad_w / 2)
    if pad_h > 0 or pad_w > 0:
        image = cv2.copyMakeBorder(image, pad_h_half, pad_h - pad_h_half, pad_w_half, pad_w - pad_w_half, cv2.BORDER_CONSTANT, value=mean)
    new_h, new_w, _ = image.shape
    stride_h = int(np.ceil(crop_h*stride_rate))
    stride_w = int(np.ceil(crop_w*stride_rate))
    grid_h = int(np.ceil(float(new_h-crop_h)/stride_h) + 1)
    grid_w = int(np.ceil(float(new_w-crop_w)/stride_w) + 1)
    prediction_crop = np.zeros((new_h, new_w, classes), dtype=float)
    count_crop = np.zeros((new_h, new_w), dtype=float)
    for index_h in range(0, grid_h):
        for index_w in range(0, grid_w):
            s_h = index_h * stride_h
            e_h = min(s_h + crop_h, new_h)
            s_h = e_h - crop_h
            s_w = index_w * stride_w
            e_w = min(s_w + crop_w, new_w)
            s_w = e_w - crop_w
            image_crop = image[s_h:e_h, s_w:e_w].copy()
            count_crop[s_h:e_h, s_w:e_w] += 1
            prediction_crop[s_h:e_h, s_w:e_w, :] += net_process(model, image_crop, mean, std)
    prediction_crop /= np.expand_dims(count_crop, 2)
    prediction_crop = prediction_crop[pad_h_half:pad_h_half+ori_h, pad_w_half:pad_w_half+ori_w]
    prediction = cv2.resize(prediction_crop, (w, h), interpolation=cv2.INTER_LINEAR)
    return prediction


def validate(val_loader, data_list, model, classes, mean, std, base_size, crop_h, crop_w, scales, gray_folder, color_folder, colors):
    logger.info('>>>>>>>>>>>>>>>> Start Evaluation >>>>>>>>>>>>>>>>')
    data_time = AverageMeter()
    batch_time = AverageMeter()
    model.eval()
    end = time.time()
    for i, (input, _) in enumerate(val_loader):
        data_time.update(time.time() - end)
        input = np.squeeze(input.numpy(), axis=0)
        image = np.transpose(input, (1, 2, 0))
        h, w, _ = image.shape
        prediction = np.zeros((h, w, classes), dtype=float)
        for scale in scales:
            new_w = round(w * scale)
            new_h = round(h * scale)
            #long_size = round(scale * base_size)
            #new_h = long_size
            #new_w = long_size
            #if h > w:
            #    new_w = round(long_size/float(h)*w)
            #else:
            #    new_h = round(long_size/float(w)*h)
            image_scale = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
            prediction += scale_process(model, image_scale, classes, crop_h, crop_w, h, w, mean, std)
        prediction /= len(scales)
        prediction = np.argmax(prediction, axis=2)
        batch_time.update(time.time() - end)
        end = time.time()
        if (i + 1) % 10 == 0:
            logger.info('Test: [{}/{}] '
                        'Data {data_time.val:.3f} ({data_time.avg:.3f}) '
                        'Batch {batch_time.val:.3f} ({batch_time.avg:.3f}).'.format(i + 1, len(val_loader),
                                                                                    data_time=data_time,
                                                                                    batch_time=batch_time))
        check_makedirs(gray_folder)
        check_makedirs(color_folder)
        gray = np.uint8(prediction)
        color = colorize(gray, colors)
        image_path, _ = data_list[i]
        image_name = image_path.split('/')[-1].split('.')[0]
        gray_path = os.path.join(gray_folder, image_name + '.png')
        color_path = os.path.join(color_folder, image_name + '.png')
        cv2.imwrite(gray_path, gray)
        color.save(color_path)
    logger.info('<<<<<<<<<<<<<<<<< End Evaluation <<<<<<<<<<<<<<<<<')


def cal_acc(data_list, pred_folder, classes, names):
    intersection_meter = AverageMeter()
    union_meter = AverageMeter()
    target_meter = AverageMeter()

    for i, (image_path, target_path) in enumerate(data_list):
        image_name = image_path.split('/')[-1].split('.')[0]
        pred = cv2.imread(os.path.join(pred_folder, image_name+'.png'), cv2.IMREAD_GRAYSCALE)
        target = cv2.imread(target_path, cv2.IMREAD_GRAYSCALE)
        intersection, union, target = intersectionAndUnion(pred, target, classes, ignore_index=255)
        intersection = intersection[1:]
        union = union[1:]
        target = target[1:]
        intersection_meter.update(intersection)
        union_meter.update(union)
        target_meter.update(target)
        accuracy = sum(intersection_meter.val) / (sum(target_meter.val) + 1e-10)
        logger.info('Evaluating {0}/{1} on image {2}, accuracy {3:.4f}.'.format(i + 1, len(data_list), image_name+'.png', accuracy))

    iou_class = intersection_meter.sum / (union_meter.sum + 1e-10)
    accuracy_class = intersection_meter.sum / (target_meter.sum + 1e-10)
    mIoU = np.mean(iou_class)
    mAcc = np.mean(accuracy_class)
    allAcc = sum(intersection_meter.sum) / (sum(target_meter.sum) + 1e-10)

    logger.info('Eval result: mIoU/mAcc/allAcc {:.4f}/{:.4f}/{:.4f}.'.format(mIoU, mAcc, allAcc))
    for i in range(classes):
        logger.info('Class_{} result: iou/accuracy {:.4f}/{:.4f}, name: {}.'.format(i, iou_class[i], accuracy_class[i], names[i]))
    return mIoU, mAcc, allAcc


if __name__ == '__main__':
    main()
