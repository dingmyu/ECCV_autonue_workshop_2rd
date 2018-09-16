#encoding:utf-8

from functions.anchor_target import compute_anchor_targets
from functions.proposal_target import compute_proposal_targets
from functions.rpn_proposal import compute_rpn_proposals
from functions.predict_bbox import compute_predicted_bboxes
import functools
import torch
import torch.nn.functional as F
import torch.nn as nn
import logging
import numpy as np

logger = logging.getLogger('global')


class FasterRCNN(nn.Module):
    def __init__(self):
        super(FasterRCNN, self).__init__()

    def feature_extractor(self, x):
        raise NotImplementedError

    def rpn(self, x):
        raise NotImplementedError

    def rcnn(self, x, rois):
        raise NotImplementedError

    def _add_rpn_loss(self, compute_anchor_targets_fn, rpn_pred_cls,
                      rpn_pred_loc):
        '''
        :param compute_anchor_targets_fn: functions to produce anchors' learning targets.
        :param rpn_pred_cls: [B, num_anchors * 2, h, w], output of rpn for classification.
        :param rpn_pred_loc: [B, num_anchors * 4, h, w], output of rpn for localization.
        :return: loss of classification and localization, respectively.
        '''
        # [B, num_anchors * 2, h, w], [B, num_anchors * 4, h, w]
        cls_targets, loc_targets, loc_masks, loc_normalizer = \
                compute_anchor_targets_fn(rpn_pred_loc.size())
        # tranpose to the input format of softmax_loss function
        rpn_pred_cls = rpn_pred_cls.permute(0,2,3,1).contiguous().view(-1, 2)
        cls_targets = cls_targets.permute(0,2,3,1).contiguous().view(-1)
        rpn_loss_cls = F.cross_entropy(
            rpn_pred_cls, cls_targets, ignore_index=-1)
        # mask out negative anchors
        rpn_loss_loc = smooth_l1_loss_with_sigma(rpn_pred_loc * loc_masks,
                                                 loc_targets, normalizer=loc_normalizer)

        # classification accuracy, top1
        acc = accuracy(rpn_pred_cls.data, cls_targets.data)[0]
        return rpn_loss_cls, rpn_loss_loc, acc

    def _add_rcnn_loss(self, rcnn_pred_cls, rcnn_pred_loc, cls_targets,
                       loc_targets, loc_weights):
        rcnn_loss_cls = F.cross_entropy(rcnn_pred_cls, cls_targets)
        loc_normalizer = cls_targets.shape[0]
        rcnn_loss_loc = smooth_l1_loss_with_sigma(rcnn_pred_loc * loc_weights,
                                                  loc_targets, normalizer=loc_normalizer)
        acc = accuracy(rcnn_pred_cls, cls_targets)[0]
        return rcnn_loss_cls, rcnn_loss_loc, acc

    def _add_rcnn_loss_ohem(self, batch_size, rcnn_pred_cls, rcnn_pred_loc, cls_targets, loc_targets,
                            loc_weights):
        ohem_loss_cls = F.cross_entropy(rcnn_pred_cls, cls_targets, reduce=False)
        ohem_loss_loc = smooth_l1_loss_with_sigma(rcnn_pred_loc * loc_weights,
                                                  loc_targets, reduce=False)
        ohem_loss = ohem_loss_cls + ohem_loss_loc

        # This could be replaced by tensor implement
        #ohem_loss = ohem_loss.data.cpu().numpy()
        #logger.info("the size of ohem_loss is {0}".format(ohem_loss.shape))
        #loss_argsort = np.argsort(ohem_loss)[::-1]

        #keep_num = min(loss_argsort.size, batch_size)

        #drop_idx = loss_argsort[keep_num:]
        # without copy(), it will crash for sth like "longtensor doesn't support negative stride"
        #drop_idx_cuda = torch.cuda.LongTensor(drop_idx.copy())
        ##############

        sorted_ohem_loss, idx = torch.sort(ohem_loss, descending=True)

        # logger.info("the size of sorted_ohem_loss is {0}".format(sorted_ohem_loss.size()))
        keep_num = min(sorted_ohem_loss.size()[0], batch_size)
        if keep_num < sorted_ohem_loss.size()[0]:
            drop_idx_cuda = idx[keep_num:]

            ohem_loss_cls[drop_idx_cuda] = 0
            ohem_loss_loc[drop_idx_cuda] = 0

        rcnn_loss_cls = ohem_loss_cls.sum() / keep_num
        rcnn_loss_loc = ohem_loss_loc.sum() / keep_num

        acc = accuracy(rcnn_pred_cls, cls_targets)[0]
        return rcnn_loss_cls, rcnn_loss_loc, acc

    def _pin_args_to_fn(self, cfg, ground_truth_bboxes, image_info, ignore_regions):
        partial_fn = {}
        if self.training:
            partial_fn['anchor_target_fn'] = functools.partial(
                compute_anchor_targets,
                cfg=cfg['train_anchor_target_cfg'],
                ground_truth_bboxes=ground_truth_bboxes,
                ignore_regions=ignore_regions,
                image_info=image_info)
            partial_fn['proposal_target_fn'] = functools.partial(
                compute_proposal_targets,
                cfg=cfg['train_proposal_target_cfg'],
                ground_truth_bboxes=ground_truth_bboxes,
                ignore_regions=ignore_regions,
                image_info=image_info,
		use_ohem=cfg['shared']['use_ohem'])
            partial_fn['rpn_proposal_fn'] = functools.partial(
                compute_rpn_proposals,
                cfg=cfg['train_rpn_proposal_cfg'],
                image_info=image_info)
        else:
            partial_fn['rpn_proposal_fn'] = functools.partial(
                compute_rpn_proposals,
                cfg=cfg['test_rpn_proposal_cfg'],
                image_info=image_info)
            partial_fn['predict_bbox_fn'] = functools.partial(
                compute_predicted_bboxes,
                cfg=cfg['test_predict_bbox_cfg'],
                image_info=image_info)
        return partial_fn

    def forward(self, input):
        '''
        Args:
            input: dict of input with keys of:
                'cfg': hyperparamters of faster-rcnn.
                'image': [b, 3, h, w], input data.
                'ground_truth_bboxes':[b, max_num_gts, 5] or None(self.training==False),
                                     each gt contains x1,y1,x2,y2,class.
                'image_info':[b, 3], resized_image_h, resized_image_w, resize_scale.
                'ignore_regions':[b,max_num_gts,4] or None.
        Return: dict of loss, predict, accuracy
        '''
        cfg = input['cfg']
        x = input['image']
        # for calculating batch_size for _add_rcnn_loss_ohem
        image_per_gpu = x.size()[0]
        ground_truth_bboxes = input['ground_truth_bboxes']
        image_info = input['image_info']
        ignore_regions = input['ignore_regions']
        partial_fn = self._pin_args_to_fn(
                cfg,
                ground_truth_bboxes,
                image_info,
                ignore_regions)

        outputs = {'losses': [], 'predict': [], 'accuracy': []}
        logger = logging.getLogger('global')
        x = self.feature_extractor(x)
        rpn_pred_cls, rpn_pred_loc = self.rpn(x)

        # rpn train function

        if self.training:
            # train rpn
            rpn_loss_cls, rpn_loss_loc, rpn_acc = \
                    self._add_rpn_loss(partial_fn['anchor_target_fn'],
                            rpn_pred_cls,
                            rpn_pred_loc)
            # get rpn proposals
            compute_rpn_proposals_fn = partial_fn['rpn_proposal_fn']
            rpn_pred_cls = rpn_pred_cls.permute(0, 2, 3, 1).contiguous()
            rpn_pred_cls = F.softmax(rpn_pred_cls.view(-1, 2), dim=1).view_as(rpn_pred_cls)
            rpn_pred_cls = rpn_pred_cls.permute(0, 3, 1, 2)
            proposals = compute_rpn_proposals_fn(rpn_pred_cls.data, rpn_pred_loc.data)

            # train fast-rcnn
            compute_proposal_target_fn = partial_fn['proposal_target_fn']
            rois, cls_targets, loc_targets, loc_weights = \
                    compute_proposal_target_fn(proposals)
            assert (rois.shape[1] == 5)
            rcnn_pred_cls, rcnn_pred_loc = self.rcnn(x, rois)

            # ohem
            if cfg['shared']['use_ohem']:
                rcnn_loss_cls, rcnn_loss_loc, rcnn_acc = self._add_rcnn_loss_ohem(
                    cfg['train_proposal_target_cfg']['batch_size'] * image_per_gpu,
                    rcnn_pred_cls, rcnn_pred_loc, cls_targets, loc_targets,
                    loc_weights)
            else:
                rcnn_loss_cls, rcnn_loss_loc, rcnn_acc = self._add_rcnn_loss(
                    rcnn_pred_cls, rcnn_pred_loc, cls_targets, loc_targets,
                    loc_weights)

            outputs['losses'] = [rpn_loss_cls, rpn_loss_loc,
                                 rcnn_loss_cls, rcnn_loss_loc]
            outputs['accuracy'] = [rpn_acc, rcnn_acc]
            outputs['predict'] = [proposals]
        else:
            # rpn test
            compute_rpn_proposals_fn = partial_fn['rpn_proposal_fn']
            rpn_pred_cls = rpn_pred_cls.permute(0, 2, 3, 1).contiguous()
            rpn_pred_cls = F.softmax(rpn_pred_cls.view(-1, 2), dim=1).view_as(rpn_pred_cls)
            rpn_pred_cls = rpn_pred_cls.permute(0, 3, 1, 2)
            proposals = compute_rpn_proposals_fn(rpn_pred_cls.data, rpn_pred_loc.data)

            # fast-rcnn test
            assert ('proposal_target_fn' not in partial_fn)
            predict_bboxes_fn = partial_fn['predict_bbox_fn']
            proposals = proposals[:, :5].cuda().contiguous()
            assert (proposals.shape[1] == 5)
            rcnn_pred_cls, rcnn_pred_loc = self.rcnn(x, proposals)
            rcnn_pred_cls = F.softmax(rcnn_pred_cls, dim=1)
            bboxes = predict_bboxes_fn(proposals, rcnn_pred_cls,
                                       rcnn_pred_loc)
            outputs['predict'] = [proposals, bboxes]
        return outputs

def smooth_l1_loss_with_sigma(pred, targets, sigma=3.0, reduce=True, normalizer=1.0):
    sigma_2 = sigma**2
    diff = pred - targets
    abs_diff = torch.abs(diff)
    smoothL1_sign = (abs_diff < 1. / sigma_2).detach().float()
    loss = torch.pow(diff, 2) * sigma_2 / 2. * smoothL1_sign \
            + abs_diff - 0.5 / sigma_2 * (1. - smoothL1_sign)
    if reduce:
        loss = torch.sum(loss)
    else:
        loss = torch.sum(loss, dim=1)
    return loss / normalizer

def accuracy(output, target, topk=(1, ), ignore_index=-1):
    """Computes the precision@k for the specified values of k"""
    keep = torch.nonzero(target != ignore_index).squeeze()
    #logger.info('target.shape:{0}, keep.shape:{1}'.format(target.shape, keep.shape))
    assert (keep.dim() == 1)
    target = target[keep]
    output = output[keep]
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res
