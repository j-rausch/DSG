import numpy as np
import fvcore.nn.weight_init as weight_init
import torch
import torch.nn.functional as F
from torch import nn
from detectron2.modeling.backbone.build import BACKBONE_REGISTRY
from detectron2.layers import (
    CNNBlockBase,
    Conv2d,
    DeformConv,
    ModulatedDeformConv,
    ShapeSpec,
    get_norm,
)
from torchvision.models.vgg import vgg16
from detectron2.modeling.backbone import Backbone

def load_vgg(use_dropout=True, use_relu=True, use_linear=True, pretrained=True):
    model = vgg16(pretrained=pretrained)
    del model.features._modules['30']  # Get rid of the maxpool
    del model.classifier._modules['6']  # Get rid of class layer
    if not use_dropout:
        del model.classifier._modules['5']  # Get rid of dropout
        if not use_relu:
            del model.classifier._modules['4']  # Get rid of relu activation
            if not use_linear:
                del model.classifier._modules['3']  # Get rid of linear layer
    convs = model.features
    fc = model.classifier
    return convs, fc

def get_conv_scale(convs):
    """
    Determines the downscaling performed by a sequence of convolutional and pooling layers
    """
    scale = 1.
    channels = 3
    for c in convs:
        stride = getattr(c, 'stride', 1.)
        scale *= stride if isinstance(stride, (int, float)) else stride[0]
        channels = getattr(c, 'out_channels') if isinstance(c, nn.Conv2d) else channels
    return scale, channels

@BACKBONE_REGISTRY.register()
class VGG(Backbone):
    def __init__(self, cfg, input_shape):
        super().__init__()
        convs, _ = load_vgg(pretrained=True)
        self.convs = convs
        self.scale, self.channels = get_conv_scale(convs)
        self._out_features = ['vgg_conv']

    def output_shape(self):
        return {
            name: ShapeSpec(
                channels=self.channels, stride=self.scale
            )
            for name in self._out_features
        }
    def forward(self, x):
        output = self.convs(x)
        return {self._out_features[0]: output}