#!/usr/bin/env python

from setuptools import setup
import torch

torch_ver = [int(x) for x in torch.__version__.split(".")[:2]]
assert torch_ver >= [1, 6], "Requires PyTorch >= 1.6"


setup(name='segmentationsg',
      version='0.1',
      description='segmentationsg',
      packages=['segmentationsg'],
      zip_safe=False)

