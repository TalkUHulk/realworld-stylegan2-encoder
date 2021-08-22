from enum import Enum
import math
import numpy as np
import torch
from torch import nn
from torch.nn import Conv2d, BatchNorm2d, PReLU, Sequential, Module
from e4e.helpers import get_blocks, bottleneck_IR, bottleneck_IR_SE, _upsample_add
from stylegan2.model import EqualLinear
from torchvision.models import mobilenet_v3_small


class ProgressiveStage(Enum):
    WTraining = 0
    Delta1Training = 1
    Delta2Training = 2
    Delta3Training = 3
    Delta4Training = 4
    Delta5Training = 5
    Delta6Training = 6
    Delta7Training = 7
    Delta8Training = 8
    Delta9Training = 9
    Delta10Training = 10
    Delta11Training = 11
    Delta12Training = 12
    Delta13Training = 13
    Delta14Training = 14
    Delta15Training = 15
    Delta16Training = 16
    Delta17Training = 17
    Inference = 18


class GradualStyleBlock(Module):
    def __init__(self, in_c, out_c, spatial):
        super(GradualStyleBlock, self).__init__()
        self.out_c = out_c
        self.spatial = spatial
        num_pools = int(np.log2(spatial))
        modules = []
        modules += [Conv2d(in_c, out_c, kernel_size=3, stride=2, padding=1),
                    nn.LeakyReLU()]
        for i in range(num_pools - 1):
            modules += [
                Conv2d(out_c, out_c, kernel_size=3, stride=2, padding=1),
                nn.LeakyReLU()
            ]
        self.convs = nn.Sequential(*modules)
        self.linear = EqualLinear(out_c, out_c, lr_mul=1)

    def forward(self, x):
        x = self.convs(x)
        x = x.view(-1, self.out_c)
        x = self.linear(x)
        return x


class Encoder4EditingMobileNet(Module):
    def __init__(self, stylegan_size=1024):
        super(Encoder4EditingMobileNet, self).__init__()

        backbone = mobilenet_v3_small(pretrained=False)
        self.input_layer = Sequential(*backbone.features[0])
        self.body = backbone.features[1:-1]

        self.styles = nn.ModuleList()
        log_size = int(math.log(stylegan_size, 2))
        self.style_count = 2 * log_size - 2
        self.coarse_ind = 3
        self.middle_ind = 7

        for i in range(self.style_count):

            if i < self.coarse_ind:
                style = GradualStyleBlock(512, 512, 8)
            elif i < self.middle_ind:
                style = GradualStyleBlock(512, 512, 16)
            else:
                style = GradualStyleBlock(512, 512, 32)
            self.styles.append(style)

        self.latlayer0 = nn.Conv2d(96, 512, kernel_size=1, stride=1, padding=0)
        self.latlayer1 = nn.Conv2d(48, 512, kernel_size=1, stride=1, padding=0)
        self.latlayer2 = nn.Conv2d(24, 512, kernel_size=1, stride=1, padding=0)

        self.progressive_stage = ProgressiveStage.Inference

        self.latent_avg = nn.Parameter(torch.zeros(18, 512))

    def get_deltas_starting_dimensions(self):
        ''' Get a list of the initial dimension of every delta from which it is applied '''
        return list(range(self.style_count))  # Each dimension has a delta applied to it

    def set_progressive_stage(self, new_stage: ProgressiveStage):
        self.progressive_stage = new_stage
        print('Changed progressive stage to: ', new_stage)

    def forward(self, x):
        x = self.input_layer(x)

        modulelist = list(self.body._modules.values())
        for i, l in enumerate(modulelist):
            x = l(x)
            if i == 2:
                c1 = x  # 1, 24, 32, 32
            elif i == 7:
                c2 = x  # 1, 48, 16, 16
            elif i == 10:
                c3 = x  # 1, 96, 8, 8

        # Infer main W and duplicate it
        c3 = self.latlayer0(c3)
        w0 = self.styles[0](c3)
        w = w0.repeat(self.style_count, 1, 1).permute(1, 0, 2)
        stage = self.progressive_stage.value
        features = c3
        for i in range(1, min(stage + 1, self.style_count)):  # Infer additional deltas
            if i == self.coarse_ind:
                p2 = _upsample_add(c3, self.latlayer1(c2))  # FPN's middle features
                features = p2
            elif i == self.middle_ind:
                p1 = _upsample_add(p2, self.latlayer2(c1))  # FPN's fine features
                features = p1
            delta_i = self.styles[i](features)
            w[:, i] += delta_i

        if w.ndim == 2:
            w = w + self.latent_avg.repeat(w.shape[0], 1, 1)[:, 0, :]
        else:
            w = w + self.latent_avg.repeat(w.shape[0], 1, 1)

        return w
