import torch
from torch import nn
import torch.nn.functional as F
# from torchvision import models
# import resnet as models
from torchE.nn import SyncBatchNorm2d

from utils import init_weights
class ChannelAttentionLayer(nn.Module):
        def __init__(self, channel, reduction=1, multiply=True,BatchNorm=nn.BatchNorm2d):
            super(ChannelAttentionLayer, self).__init__()
            self.avg_pool = nn.AdaptiveAvgPool2d(1)
            self.fc = nn.Sequential(
                    nn.Linear(channel, channel // reduction),
                    nn.ReLU(inplace=True),
                    nn.Linear(channel // reduction, channel),
                    nn.Sigmoid()
                    )
            self.multiply = multiply
        def forward(self, x):
            b, c, _, _ = x.size()
            y = self.avg_pool(x).view(b, c)
            y = self.fc(y).view(b, c, 1, 1)
            if self.multiply == True:
                return x * y
            else:
                return y

class SpatialAttentionLayer(nn.Module):
        def __init__(self, channel, reduction=1, multiply=True,BatchNorm=nn.BatchNorm2d):
            super(SpatialAttentionLayer, self).__init__()
            self.fc = nn.Sequential(
                    nn.Conv2d(channel, channel // 32, kernel_size=3, stride=1, padding=1, bias=False),
                    BatchNorm(channel // 32),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(channel // 32, 1,kernel_size=3, stride=1, padding=1, bias=False),
                    BatchNorm(1),
                    nn.Sigmoid()
                    )
            self.multiply = multiply
        def forward(self, x):
            b, c, w, h = x.size()
            y = self.fc(x).view(b, 1, w, h)
            if self.multiply == True:
                return x * y
            else:
                return y
            
class SpatialFCAttentionLayer(nn.Module):
        def __init__(self, channel, reduction=1, multiply=True,BatchNorm=nn.BatchNorm2d):
            super(SpatialFCAttentionLayer, self).__init__()
            self.fc = nn.Sequential(
                    nn.Conv2d(channel, channel // reduction, kernel_size=3, stride=1, padding=1, bias=False),
                    BatchNorm(channel // reduction),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(channel // reduction, 1,kernel_size=3, stride=1, padding=1, bias=False),
                    BatchNorm(1),
                    nn.Sigmoid()
                    )
            self.multiply = multiply
        def forward(self, x):
            b, c, w, h = x.size()
            y = self.fc(x).view(b, 1, w, h)
            if self.multiply == True:
                return x * y
            else:
                return y


class PPM(nn.Module):
    def __init__(self, in_dim, reduction_dim, bins, BatchNorm=nn.BatchNorm2d):
        super(PPM, self).__init__()
        self.features = []
        self.features.append(nn.Sequential(
            nn.Conv2d(in_dim, 256, kernel_size=3, stride=1, padding=6, dilation =6, bias=True),
            nn.Conv2d(256, 256, kernel_size=1, stride=1, bias=False),			
            BatchNorm(256),
            nn.ReLU(inplace=True)
        ))
        self.features.append(nn.Sequential(
            nn.Conv2d(in_dim, 256, kernel_size=3, stride=1, padding=12, dilation =12, bias=True),
            nn.Conv2d(256, 256, kernel_size=1, stride=1, bias=False),			
            BatchNorm(256),
            nn.ReLU(inplace=True)
        ))
        self.features.append(nn.Sequential(
            nn.Conv2d(in_dim, 256, kernel_size=3, stride=1, padding=18, dilation =18, bias=True),
            nn.Conv2d(256, 256, kernel_size=1, stride=1, bias=False),			
            BatchNorm(256),
            nn.ReLU(inplace=True)
        ))
        self.features.append(nn.Sequential(
            nn.Conv2d(in_dim, 256, kernel_size=1, stride=1, bias=False),			
            BatchNorm(256),
            nn.ReLU(inplace=True)
        ))
        
        bins = (1, 2, 3, 6)
        for bin in bins:
            self.features.append(nn.Sequential(
                nn.AdaptiveAvgPool2d(bin),
                #nn.Upsample(size=(101, 101), mode='bilinear'),
                nn.Conv2d(in_dim, 256, kernel_size=1, stride=1, bias=False),			
                BatchNorm(256),
                nn.ReLU(inplace=True),
		nn.Upsample(size=(101, 101), mode='bilinear')
            ))

        self.features = nn.ModuleList(self.features)

    def forward(self, x):
        x_size = x.size()
        out = []
        for f in self.features:
            out.append(f(x))
            #out.append(F.upsample(f(x), x_size[2:], mode='bilinear'))
        out = torch.cat(out, 1)
        return out


class PSPNet(nn.Module):
    def __init__(self, backbone='resnet', layers=50, bins=(1, 2, 3, 6), dropout=0.1, classes=2, zoom_factor=8, use_ppm=True, use_softmax=True, use_aux=True, pretrained=True, syncbn=True, group_size=8, group=None):
        super(PSPNet, self).__init__()
        assert layers in [50, 101, 152]
        assert 2048 % len(bins) == 0
        assert classes > 1
        assert zoom_factor in [1, 2, 4, 8]
        self.zoom_factor = zoom_factor
        self.use_ppm = use_ppm
        self.use_softmax = use_softmax
        self.use_aux = use_aux

        if backbone == 'resnet':
            import resnet as models
        elif backbone == 'ibnnet_a':
            import ibnnet_a as models
        elif backbone == 'ibnnet_b':
            import ibnnet_b as models
        else:
            raise NameError('Backbone type not defined!')

        if syncbn:
            # from lib.syncbn import SynchronizedBatchNorm2d as BatchNorm
            def BNFunc(*args, **kwargs):
                return SyncBatchNorm2d(*args, **kwargs, group_size=group_size, group=group, sync_stats=True)
            BatchNorm = BNFunc
        else:
            from torch.nn import BatchNorm2d as BatchNorm
        models.BatchNorm = BatchNorm

        if layers == 50:
            resnet = models.resnet50(pretrained=pretrained)
        elif layers == 101:
            resnet = models.resnet101(pretrained=pretrained)
        else:
            resnet = models.resnet152(pretrained=pretrained)
        if backbone == 'ibnnet_b':
            self.layer0 = nn.Sequential(resnet.conv1, resnet.INCat0, resnet.relu, resnet.maxpool)
        else:
            self.layer0 = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool)
        self.layer1, self.layer2, self.layer3, self.layer4 = resnet.layer1, resnet.layer2, resnet.layer3, resnet.layer4

        for n, m in self.layer3.named_modules():
            if 'conv2' in n and not 'convbnin.conv2' in n:
                m.dilation, m.padding, m.stride = (2, 2), (2, 2), (1, 1)
            elif 'downsample.0' in n:
                m.stride = (1, 1)
        for n, m in self.layer4.named_modules():
            if 'conv2' in n and not 'convbnin.conv2' in n:
                m.dilation, m.padding, m.stride = (4, 4), (4, 4), (1, 1)
            elif 'downsample.0' in n:
                m.stride = (1, 1)
                
        channel_4x = 256
        # channel attention layer and spatial attention layer.
        self.cam_4x = ChannelAttentionLayer(channel_4x, reduction=1, multiply=True)
        self.sam_4x = SpatialAttentionLayer(channel_4x, reduction=1, multiply=True)

        channel_8x = 512
        # channel attention layer and spatial attention layer.
        self.cam_8x = ChannelAttentionLayer(channel_8x, reduction=1, multiply=True)
        self.sam_8x = SpatialAttentionLayer(channel_8x, reduction=1, multiply=True)
        
        channel_1x = classes # final predict
        # channel attention layer and spatial attention layer.
        self.cam_1x = ChannelAttentionLayer(channel_1x, reduction=1, multiply=True)
        self.sam_1x = SpatialFCAttentionLayer(channel_1x, reduction=1, multiply=True)

        fea_dim = 2048
        if use_ppm:
            self.ppm = PPM(fea_dim + 128, int(fea_dim/len(bins)), bins, BatchNorm)
            fea_dim *= 2
            
        self.cls = nn.Sequential(
            nn.Conv2d(768, 256, kernel_size=3, padding=1, bias=False),
            BatchNorm(256),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=dropout),
            nn.Conv2d(256, classes, kernel_size=1)
        )
        
        self.cls_2 = nn.Sequential(
            nn.Conv2d(classes * 2, classes, kernel_size=1)
        )
        
        self.conv6 = nn.Sequential(
            nn.Conv2d(256 * 8, 512, kernel_size=1, padding=0, bias=True),
            BatchNorm(512),
            nn.ReLU(inplace=True)
        )
        self.conv1_1x1 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=1, padding=0, bias=True),
            BatchNorm(256),
            nn.ReLU(inplace=True)
        )
        self.conv2_1x1 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=1, padding=0, bias=True),
            BatchNorm(256),
            nn.ReLU(inplace=True)
        )
        if use_aux:
            self.aux = nn.Sequential(
                nn.Conv2d(1024, 256, kernel_size=3, padding=1, bias=False),
                BatchNorm(256),
                nn.ReLU(inplace=True),
                nn.Dropout2d(p=dropout),
                nn.Conv2d(256, classes, kernel_size=1)
            )
            # init_weights(self.aux)
        # comment to use default initialization
        init_weights(self.ppm)
        # init_weights(self.cls)

    def forward(self, x, fcw_input):
        x_size = x.size()
        assert (x_size[2]-1) % 8 == 0 and (x_size[3]-1) % 8 == 0
        h = int((x_size[2] - 1) / 8 * self.zoom_factor + 1)
        w = int((x_size[3] - 1) / 8 * self.zoom_factor + 1)

        x = self.layer0(x)
        x1 = self.layer1(x)
        
        x = self.layer2(x1)
        x = self.layer3(x)
        if self.training and self.use_aux:
            aux = self.aux(x)
            if self.zoom_factor != 1:
                aux = F.upsample(aux, size=(h, w), mode='bilinear')
        x = self.layer4(x)
        x = torch.cat([x, fcw_input], dim = 1)
        if self.use_ppm:
            x = self.ppm(x)
        
        x = self.conv6(x)
        # 8x attention
        x = self.cam_8x(x)
        x = self.sam_8x(x)

        if self.zoom_factor != 1:
            x = F.upsample(x, size=(h, w), mode='bilinear')
        x1 = self.conv1_1x1(x1)
        x1 = self.conv2_1x1(x1)
        x1 = self.cam_4x(x1)
        x1 = self.sam_4x(x1)
        
        x = torch.cat([x,x1], 1)
        x = self.cls(x)
        
        x_a = self.cam_1x(x)
        x_a = self.sam_1x(x_a)

        x = torch.cat([x, x_a], dim=1) # concat 19 and 19.
        x = self.cls_2(x)

        if self.use_softmax:
            x = F.log_softmax(x, dim=1)
            if self.use_aux:
                aux = F.log_softmax(aux, dim=1)
                return x, aux
            else:
                return x
        else:
            if self.use_aux:
                return x, aux
            else:
                return x


if __name__ == '__main__':
    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = '0'
    sim_data = torch.autograd.Variable(torch.rand(2, 3, 473, 473)).cuda(async=True)
    model = PSPNet(layers=50, bins=(1, 2, 3, 6), dropout=0.1, classes=21, zoom_factor=1, use_ppm=True, use_softmax=True, use_aux=True, pretrained=True, syncbn=True).cuda()
    print(model)
    output, _ = model(sim_data)
    print('PSPNet', output.size())
