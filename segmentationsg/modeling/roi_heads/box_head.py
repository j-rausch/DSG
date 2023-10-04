import numpy as np
from typing import List
import fvcore.nn.weight_init as weight_init
import torch
from torch import nn
import logging
from detectron2.config import configurable
from detectron2.layers import Conv2d, Linear, ShapeSpec, get_norm
from detectron2.utils.registry import Registry
from detectron2.modeling.roi_heads.box_head import ROI_BOX_HEAD_REGISTRY

from ..backbone import load_vgg
from torchvision import models as M, ops
from detectron2.modeling.backbone.resnet import BottleneckBlock, ResNet

@ROI_BOX_HEAD_REGISTRY.register()
class VGGConvFCHead(nn.Module):
    def __init__(self, cfg, input_shape):
        super().__init__()
        _, fc = load_vgg(pretrained=True)
        _output_size = input_shape.channels
        for c in fc:
            _output_size = getattr(c, 'out_features') if isinstance(c, nn.Linear) else _output_size
        self.fc = fc
        self._output_size = _output_size

    def forward(self, x):
        x = x.flatten(1)
        return self.fc(x)

    @property
    @torch.jit.unused
    def output_shape(self):
        """
        Returns:
            ShapeSpec: the output feature shape
        """
        o = self._output_size
        if isinstance(o, int):
            return ShapeSpec(channels=o)
        else:
            return ShapeSpec(channels=o[0], height=o[1], width=o[2])