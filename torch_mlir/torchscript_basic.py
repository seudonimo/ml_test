# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
# Also available under a BSD-style license. See LICENSE.

import sys

from PIL import Image
import requests


import torchvision.models as models
from torchvision import transforms

import torch
import torch.nn as nn
import torch.nn.functional as F

import torch_mlir
from torch_mlir_e2e_test.linalg_on_tensors_backends import refbackend

class CNN(torch.nn.Module):
  def __init__(self):
    super(CNN, self).__init__()
    self.conv1 = nn.Conv2d(in_channels=3, out_channels=3, kernel_size=5, stride=1)
    #self.conv2 = nn.Conv2d(in_channels=3, out_channels=5, kernel_size=5, stride=1)
    #self.flatten = nn.Flatten()
    #self.softmax = nn.Softmax()

  def forward(self, x):
    print("연산 전", x.size())
    #x = F.relu(self.conv1(x))
    x = self.conv1(x)
    return
    print("conv1 연산 후", x.size())
    #x = F.relu(self.conv2(x))
    #print("conv2 연산 후",x.size())
    x = self.flatten(x)
    x = self.softmax(x)
    return x

cnn = CNN()
x = torch.randn(1, 3, 10, 10)
torch.onnx.export(cnn, x, './torch_basic.onnx')
cnn.train(False)
import pdb;pdb.set_trace()

module = torch_mlir.compile(cnn, torch.randn(1, 3, 10, 10), output_type="torch")
try:
    backend = refbackend.RefBackendLinalgOnTensorsBackend()
    compiled = backend.compile(module)
    jit_module = backend.load(compiled)
except:
   print("compile failed!")
import pdb;pdb.set_trace()
print(cnn)