from extensions import PSRoIPool
from .rfcn import FasterRCNN
from models.head import NaiveRpnHead
import torch.nn as nn
import torch.utils.model_zoo as model_zoo
import math
import torch

def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes,out_planes,kernel_size=3,stride=stride,padding=1,bias=False)

class BasicBlock(nn.Module):
    def __init__(self, in_planes, planes, stride=1):

        super(BasicBlock, self).__init__()

        self.conv1 = conv3x3(in_planes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.stride = stride
    
    def forward(self, x):

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        return out

class ConcatBlock(nn.Module):
    expansion = [16,8,4,4]

    def __init__(self, in_planes, mul, stride = 1):
        super(ConcatBlock, self).__init__()
        self.expansion  = [tmp * mul for tmp in self.expansion]
        self.block1 = BasicBlock(in_planes,         self.expansion[0], stride = 1)
        self.block2 = BasicBlock(self.expansion[0], self.expansion[1], stride = 1)
        self.block3 = BasicBlock(self.expansion[1], self.expansion[2], stride = 1)
        self.block4 = BasicBlock(self.expansion[2], self.expansion[3], stride = 1)

    def forward(self,x):

        out1 = self.block1(x)
        out2 = self.block2(out1)
        out3 = self.block3(out2)
        out4 = self.block4(out3)

        out = torch.cat([out1,out2,out3,out4] , 1)

        return out

class ConcatBlockRes(nn.Module):
    expansion = [16,8,4,4]

    def __init__(self, in_planes, mul, stride = 1):
        super(ConcatBlockRes, self).__init__()
        self.expansion  = [tmp * mul for tmp in self.expansion]
        self.block1 = BasicBlock(in_planes,         self.expansion[0], stride = 1)
        self.block2 = BasicBlock(self.expansion[0], self.expansion[1], stride = 1)

    def forward(self,x):

        out1 = self.block1(x)
        out2 = self.block2(out1)

        out = torch.cat([out1,out2] , 1)

        return out



class V11RFCN(FasterRCNN):

    def __init__(self, cfg):

        super(V11RFCN, self).__init__()
        self.n_classes = len(cfg['class_names'])

        #feature;
        #self.features = features
        self.layer1 = BasicBlock (3 ,  16, stride = 2)
        self.layer2 = BasicBlock (16,  16, stride = 2)
        self.layer3 = ConcatBlock(16,  1 , stride = 1)
        self.layer4 = BasicBlock (32,  32, stride = 2)
        self.layer5 = ConcatBlock(32,  2 , stride = 1)
        self.layer6 = ConcatBlock(64,  2 , stride = 1)
        self.layer7 = BasicBlock (64,  64, stride = 2)
        self.layer8 = ConcatBlock(64,  4 , stride = 1)
        self.layer9 = ConcatBlock(128, 4 , stride = 1)
        self.layer10 = ConcatBlock(128,4 , stride = 1)
        
        #rpn head;
        num_anchors = len(cfg['anchor_scales']) * len(cfg['anchor_ratios'])
        self.rpn_head = NaiveRpnHead(128, num_classes = 2, num_anchors=num_anchors)

        #rcnn head;
        self.thin_conv = BasicBlock(128, 32, stride = 1)
        self.cls_feat  = conv3x3(32, 5*5*self.n_classes, stride = 1)
        self.bbox_feat = conv3x3(32, 5*5*self.n_classes*4, stride = 1)

        self.psroi_pool_cls = PSRoIPool(group_size = 5, spatial_scale = (1.0 / cfg['anchor_stride']), output_dim = self.n_classes)
        self.psroi_pool_loc = PSRoIPool(group_size = 5, spatial_scale = (1.0 / cfg['anchor_stride']), output_dim = 4 * self.n_classes)

        self.cls_pred  = nn.AvgPool2d((5,5), stride = (5,5))
        self.bbox_pred = nn.AvgPool2d((5,5), stride = (5,5))

        self._initialize_weights()

    def feature_extractor(self, x):
        #return self.features(x)
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.layer5(out)
        out = self.layer6(out)
        out = self.layer7(out)
        out = self.layer8(out)
        out = self.layer9(out)
        out = self.layer10(out)

        return out

    def rpn(self, x):
        return self.rpn_head(x)

    def rcnn(self, x, rois):
        thin_conv = self.thin_conv(x)

        r_score_map = self.cls_feat(thin_conv)
        r_bbox_map  = self.bbox_feat(thin_conv)

        psroi_pooled_cls = self.psroi_pool_cls(r_score_map, rois)
        psroi_pooled_loc = self.psroi_pool_loc(r_bbox_map , rois)

        cls_pred  = self.cls_pred(psroi_pooled_cls)
        bbox_pred = self.bbox_pred(psroi_pooled_loc)

        cls_pred  = torch.squeeze(cls_pred)
        bbox_pred = torch.squeeze(bbox_pred)

        return cls_pred, bbox_pred

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()
    
def make_layers():
    layers = []
    layers.append(BasicBlock (3 ,  16, stride = 2))
    layers.append(BasicBlock (16,  16, stride = 2))
    layers.append(ConcatBlock(16,  1 , stride = 1))
    layers.append(BasicBlock (32,  32, stride = 2))
    layers.append(ConcatBlock(64,  2 , stride = 1))
    layers.append(ConcatBlock(64,  2 , stride = 1))
    layers.append(BasicBlock (64,  64, stride = 2))
    layers.append(ConcatBlock(64,  4 , stride = 1))
    layers.append(ConcatBlock(128, 4 , stride = 1))
    layers.append(ConcatBlockRes(128, 4 , stride = 1))

    return nn.Sequential(*layers)

def V11Net_RFCN(pretrained=False, **kwargs):

    model = V11RFCN(**kwargs)
    if pretrained:
        print("no pretrain model!")
    return model

