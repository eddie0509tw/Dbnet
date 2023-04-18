import math
import torch
import torch.nn as nn
from torch.hub import load_state_dict_from_url

__all__ = ['mobilenet_v2_x1_0']

model_urls = {
    #currently hadn't found a pretrained weight
    'mobilenet_v2_x1_0': None,
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

class InvertedResidual(nn.Module):
    def __init__(self, inp, oup, stride, expand_ratio):
        super(InvertedResidual, self).__init__()

        if not (1 <= stride <= 2):
            raise ValueError('illegal stride value')
        self.stride = stride

        self.exp_r = expand_ratio;
        hidden_dim = round(inp * self.exp_r);

        if self.exp_r == 1:
            self.branch = nn.Sequential(
                # dw conv
                self.depthwise_conv(hidden_dim, hidden_dim, 3, stride, padding = 1, bias=False),               
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True),
                # pw-linear
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
            )

        else:
            self.branch = nn.Sequential(
                # pw
                nn.Conv2d(inp, hidden_dim, 1, 1, 0, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True),
                # dw
                self.depthwise_conv(hidden_dim, hidden_dim, 3, stride, padding = 1, bias=False),               
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True),
                # pw-linear
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
            )

        self.downsample = nn.Sequential(
            nn.Conv2d(inp, oup, kernel_size = 1, stride = stride , bias=False),
            nn.BatchNorm2d(oup),
        )

        self.identity = stride == 1 
    @staticmethod
    def depthwise_conv(i, o, kernel_size, stride, padding, bias=False):
        return nn.Conv2d(i, o, kernel_size, stride, padding, bias=bias, groups=i)

    def forward(self, x):
        if self.identity:
            downsampx = self.downsample(x)
            return downsampx + self.branch(x)
        else:
            return self.branch(x)


class MobileNetV2(nn.Module):
    def __init__(self,  stages_repeats, stages_out_channels, num_classes=1000, width_mult=1.):
        super(MobileNetV2, self).__init__()

        if len(stages_repeats) != 7:
            raise ValueError('expected stages_repeats as list of 4 positive ints')
        if len(stages_out_channels) != 9:
            raise ValueError('expected stages_out_channels as list of 5 positive ints')
        self._stage_out_channels = stages_out_channels

        self.tlist = [1,6,6,6,6,6,6,6]
        self.slist = [1,2,2,2,1,2,1,1]

        input_channels = 3
        output_channels = self._stage_out_channels[0] # 32
        #output_channels = _make_divisible(output_channels * width_mult, 4 if width_mult == 0.1 else 8)

        self.conv1 = nn.Sequential(
            nn.Conv2d(input_channels, output_channels, 3, 2, 1, bias=False),
            nn.BatchNorm2d(output_channels),
            nn.ReLU6(inplace=True),
        )
        input_channels = output_channels


        stage_names = ['stage{}'.format(i) for i in [2, 3, 4, 5, 6,7,8,9]]
        for name, repeats, output_channels, t, s in zip(
                stage_names, stages_repeats, self._stage_out_channels[1:], self.tlist, self.slist):
            
            #output_channels = _make_divisible(output_channels * width_mult, 4 if width_mult == 0.1 else 8)

            seq = [InvertedResidual(input_channels, output_channels, s, t)]
            for i in range(repeats - 1):
                
                seq.append(InvertedResidual(output_channels, output_channels, 1, t))
                


            setattr(self, name, nn.Sequential(*seq))
            input_channels = output_channels
            

        output_channels = self._stage_out_channels[-1]

        self.conv9 = nn.Sequential(
            nn.Conv2d(input_channels, output_channels, 1, 1, 0, bias=False),
            nn.BatchNorm2d(output_channels),
            nn.ReLU6(inplace=True),
        )



    def forward(self, x):
        x = self.conv1(x)
        c2 = self.stage2(x)
        c3 = self.stage3(c2)
        c4 = self.stage4(c3)
        c5 = self.stage5(c4)
        c6 = self.stage6(c5)
        
        #c7 = self.stage7(c6)
        #c8 = self.stage8(c7)
        #c9 = self.conv9(c8)

        return c3, c4, c5, c6
    

    
def _mobilenetv2(arch, pretrained, progress, *args, **kwargs):
    model = MobileNetV2(*args, **kwargs)

    if pretrained:
        model_url = model_urls[arch]
        if model_url is None:
            raise NotImplementedError('pretrained {} is not supported as of now'.format(arch))
        else:
            state_dict = load_state_dict_from_url(model_url, progress=progress)
            model.load_state_dict(state_dict,strict=False)

    return model


def mobilenet_v2_x1_0(pretrained=False, progress=True, **kwargs):
    """
    Constructs a ShuffleNetV2 with 0.5x output channels, as described in
    `"ShuffleNet V2: Practical Guidelines for Efficient CNN Architecture Design"
    <https://arxiv.org/abs/1807.11164>`_.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bsar of the download to stderr
    """
    return _mobilenetv2('mobilenet_v2_x1_0', pretrained, progress,
                         [1, 2, 3, 4, 3,3,1], [32, 16, 24, 32, 64 , 96, 160, 320,1280], **kwargs)