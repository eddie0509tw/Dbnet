import math
import torch
import torch.nn as nn
from torch.hub import load_state_dict_from_url

__all__ = ['MobileNetV3_Large']

model_urls = {
    #currently hadn't found a pretrained weight
    'MobileNetV3_Large': None,
}
def _make_divisible(v, divisor, min_value=None):
        """
        This function is taken from the original tf repo.
        It ensures that all layers have a channel number that is divisible by 8
        It can be seen here:
        https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
        :param v:
        :param divisor:
        :param min_value:
        :return:
        """
        if min_value is None:
            min_value = divisor
        new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
        # Make sure that round down does not go down by more than 10%.
        if new_v < 0.9 * v:
            new_v += divisor
        return new_v

class SELayer(nn.Module):
    def __init__(self, channel, reduction=4):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
                nn.Linear(channel, _make_divisible(channel // reduction, 8)),
                nn.ReLU(inplace=True),
                nn.Linear(_make_divisible(channel // reduction, 8), channel),
                h_sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y
    
class h_sigmoid(nn.Module):
    def __init__(self, inplace=True):
        super(h_sigmoid, self).__init__()
        self.relu = nn.ReLU6(inplace=inplace)

    def forward(self, x):
        return self.relu(x + 3) / 6


class h_swish(nn.Module):
    def __init__(self, inplace=True):
        super(h_swish, self).__init__()
        self.sigmoid = h_sigmoid(inplace=inplace)

    def forward(self, x):
        return x * self.sigmoid(x) 

def conv_3x3_bn(inp, oup, stride):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
        nn.BatchNorm2d(oup),
        h_swish()
    )


def conv_1x1_bn(inp, oup):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
        nn.BatchNorm2d(oup),
        h_swish()
    )   

class InvertedResidual(nn.Module):
    def __init__(self, inp,hidden_dim, oup, kernel_size, stride, use_se, use_hs):
        super(InvertedResidual, self).__init__()

        assert stride in [1, 2]

        if inp == hidden_dim:
            self.branch = nn.Sequential(
                # dw conv
                nn.Conv2d(hidden_dim, hidden_dim, kernel_size, stride, (kernel_size - 1) // 2, groups=hidden_dim, bias=False),
                nn.BatchNorm2d(hidden_dim),
                h_swish() if use_hs else nn.ReLU(inplace=True),
                # Squeeze-and-Excite
                SELayer(hidden_dim) if use_se else nn.Identity(),
                # pw-linear
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
            )

        else:
            self.branch = nn.Sequential(
                # pw
                nn.Conv2d(inp, hidden_dim, 1, 1, 0, bias=False),
                nn.BatchNorm2d(hidden_dim),
                h_swish() if use_hs else nn.ReLU(inplace=True),
                # dw
                nn.Conv2d(hidden_dim, hidden_dim, kernel_size, stride, (kernel_size - 1) // 2, groups=hidden_dim, bias=False),
                nn.BatchNorm2d(hidden_dim),
                # Squeeze-and-Excite
                SELayer(hidden_dim) if use_se else nn.Identity(),
                h_swish() if use_hs else nn.ReLU(inplace=True),
                # pw-linear
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
            )

        self.downsample = nn.Sequential(
            nn.Conv2d(inp, oup, kernel_size = 1, stride = stride , bias=False),
            nn.BatchNorm2d(oup),
        )

        self.identity = stride == 1 


    def forward(self, x):
        if self.identity:
            downsampx = self.downsample(x)
            return downsampx + self.branch(x)
        else:
            return self.branch(x)


class MobileNetV3(nn.Module):
    def __init__(self,  stages_out_channels, num_classes=1000, width_mult=1.):
        super(MobileNetV3, self).__init__()

        if len(stages_out_channels) != 16:
            raise ValueError('expected stages_out_channels as list of 15 positive ints')
        self._stage_out_channels = stages_out_channels

        self.klist = [3,3,3,5,5,5,3,3,3, 3,3,3,5,5,5]
        self.tlist = [1,4,3,3,3,3,6,2.5, 2.3,2.3,6,6,6,6,6]
        self.slist = [1,2,1,2,1,1,2, 1, 1, 1, 1,1,2,1,1]
        self.selist = [0,0,0,1,1,1,0,0, 0, 0, 1,1,1,1,1]
        self.hslist = [0,0,0,0,0, 0,1,1,1, 1, 1,1,1,1,1]
    

        input_channels = 3
        output_channels = self._stage_out_channels[0] # 16
        #output_channels = _make_divisible(output_channels * width_mult, 4 if width_mult == 0.1 else 8)

        self.conv1 = conv_3x3_bn(input_channels, output_channels, 2)
        input_channels = output_channels

        
        stage_names = ['stage{}'.format(i) for i in range(2,17)]
        for name, output_channels,k, t, s ,use_se,use_hs in zip(
            stage_names,  self._stage_out_channels[1:], self.klist, self.tlist, self.slist, self.selist, self.hslist):
            seq = []
            output_channels = _make_divisible(output_channels * width_mult, 8)
            exp_size = _make_divisible(input_channels * t, 8)
            seq.append(InvertedResidual(input_channels, exp_size, output_channels, k, s, use_se, use_hs))
            setattr(self, name, nn.Sequential(*seq))
            input_channels = output_channels
            



    def forward(self, x):
        x = self.conv1(x)
        c2 = self.stage2(x)
        c3 = self.stage3(c2)
        c4 = self.stage4(c3)
        c5 = self.stage5(c4)
        c6 = self.stage6(c5)   
        c7 = self.stage7(c6)
        c8 = self.stage8(c7)
        c10 = self.stage9(c8)

        c11 = self.stage10(c10)
        c12 = self.stage11(c11)
        c13 = self.stage12(c12)
        c14 = self.stage13(c13)
        c15 = self.stage14(c14)        
        c16 = self.stage15(c15)
        c17 = self.stage16(c16)

        return c3, c5, c16, c17
    

    
def _mobilenetv3(arch, pretrained, progress, *args, **kwargs):
    model = MobileNetV3(*args, **kwargs)

    if pretrained:
        model_url = model_urls[arch]
        if model_url is None:
            raise NotImplementedError('pretrained {} is not supported as of now'.format(arch))
        else:
            state_dict = load_state_dict_from_url(model_url, progress=progress)
            model.load_state_dict(state_dict,strict=False)

    return model


def MobileNetV3_Large(pretrained=False, progress=True, **kwargs):
    """
    Constructs a ShuffleNetV2 with 0.5x output channels, as described in
    `"ShuffleNet V2: Practical Guidelines for Efficient CNN Architecture Design"
    <https://arxiv.org/abs/1807.11164>`_.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bsar of the download to stderr
    """
    return _mobilenetv3('MobileNetV3_Large', pretrained, progress,
                         [16,16,24,24,40,40,40,80, 80, 80, 80, 112,112,160,160,160], **kwargs)