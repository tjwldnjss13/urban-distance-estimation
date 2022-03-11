import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F
from scipy.io import loadmat

if __name__ == '__main__':
   a = torch.Tensor([1]).to('cuda:0')
   b = torch.Tensor([2]).to('cpu')
   if a.device == b.device:
      print(1)
   else:
      print(2)















