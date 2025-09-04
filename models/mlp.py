# -*- coding: utf-8 -*-
"""
Created on 2025/09/04 17:19:11
@author: Whenxuan Wang
@email: wwhenxuan@gmail.com
@url: https://github.com/wwhenxuan
"""
import numpy as np
import torch

from torch import nn

from layers import MLPEncoder


class MLP(nn.Module):
    """The MLP models for time series clustering."""

    def __init__(
        self,
    ) -> None:
        super(MLP, self).__init__()
        pass
