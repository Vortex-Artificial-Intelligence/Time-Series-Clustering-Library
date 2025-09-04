# -*- coding: utf-8 -*-
"""
Created on 2025/08/26 21:55:50
@author: Whenxuan Wang
@email: wwhenxuan@gmail.com
@url: https://github.com/wwhenxuan
"""
import torch
from torch import nn

from typing import Optional, Union, Tuple, List, Any


class MLPEncoder(nn.Module):
    def __init__(
        self,
        context_len: int,
        n_channels: int,
        hidden_dim: Union[Tuple[int], List[int]] = (256, 72, 32),
        bias: Optional[bool] = True,
    ) -> None:
        super(MLPEncoder, self).__init__()

        self.context_len = context_len
        self.n_channels = n_channels

        self.hidden_dim = (
            hidden_dim if isinstance(hidden_dim, list) else list(hidden_dim)
        )

        self.dim_list = [self.context_len] + self.hidden_dim
        self.features_dim = [
            (in_features, out_features)
            for in_features, out_features in zip(self.dim_list[:-1], self.dim_list[1:])
        ]

        self.bias = bias

        self.encoder = self._built_encoder()
        self.decoder = self._built_decoder()

    def _built_encoder(self) -> nn.ModuleList:
        """构建MLP的编码器部分"""
        return nn.ModuleList(
            modules=[
                nn.Sequential(
                    nn.Linear(
                        in_features=in_features,
                        out_features=out_features,
                        bias=self.bias,
                    ),
                    nn.ReLU(inplace=True),
                )
                for in_features, out_features in self.features_dim
            ]
        )

    def _built_decoder(self) -> nn.ModuleList:
        """构建MLP的解码器部分"""
        features_dim = list(reversed(self.features_dim))
        print(features_dim)
        modules = [
            nn.Sequential(
                nn.Linear(
                    in_features=in_features,
                    out_features=out_features,
                    bias=self.bias,
                ),
                nn.ReLU(inplace=True),
            )
            for out_features, in_features in features_dim[:-1]
        ]
        modules.append(
            nn.Sequential(
                nn.Linear(
                    in_features=self.features_dim[0][1],
                    out_features=self.features_dim[0][0],
                    bias=self.bias,
                )
            )
        )

        return nn.ModuleList(modules=modules)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """"""
        for layer in self.encoder:
            x = layer(x)
            # print("encoder:", x.shape)
        return x

    def decode(self, x: torch.Tensor) -> torch.Tensor:
        """"""
        for layer in self.decoder:
            x = layer(x)
            # print("decoder:", x.shape)
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """全连接深度神经网络中的正向传播部分"""
        x_enc = self.encode(x=x)
        return self.decode(x=x_enc)


class CNNEncoder(nn.Module):
    def __init__(
        self,
        context_len: int,
        n_channels: int,
        channels_list: Union[Tuple[int], List[int]] = (16, 32, 16, 8),
        kernel_list: Union[Tuple[int], List[int]] = (5, 5, 3, 3),
        bias: Optional[bool] = True,
    ) -> None:
        super(CNNEncoder, self).__init__()

        self.context_len = context_len
        self.n_channels = n_channels

        self.channels_list = (
            channels_list if isinstance(channels_list, list) else list(channels_list)
        )
        self.kernel_list = (
            kernel_list if isinstance(kernel_list, list) else list(kernel_list)
        )

        assert len(self.channels_list) == len(self.kernel_list)

        self.channels_list = [n_channels] + self.channels_list
        self.params_list = [
            (in_channels, out_channels, kernel_size, kernel_size // 2)
            for in_channels, out_channels, kernel_size in zip(
                self.channels_list[:-1],
                self.channels_list[1:],
                self.kernel_list,
            )
        ]
        print("params_list:", len(self.params_list))

        self.bias = bias

        self.encoder = self._built_encoder()
        self.decoder = self._built_decoder()

    def _built_encoder(self) -> nn.ModuleList:
        """构建CNN模块的编码器"""
        return nn.ModuleList(
            modules=[
                nn.Sequential(
                    nn.Conv1d(
                        in_channels=in_channels,
                        out_channels=out_channels,
                        kernel_size=kernel_size,
                        stride=1,
                        padding=padding,
                        bias=True,
                    ),
                    nn.ReLU(inplace=True),
                    nn.MaxPool1d(kernel_size=2),
                )
                for (
                    in_channels,
                    out_channels,
                    kernel_size,
                    padding,
                ) in self.params_list
            ]
        )

    def _built_decoder(self) -> nn.Module:
        """构建CNN模块的编码器"""
        in_features = self.context_len
        for _ in range(0, len(self.params_list)):
            in_features = int(in_features / 2)

        return nn.Sequential(
            nn.Conv1d(
                in_channels=self.channels_list[-1],
                out_channels=self.n_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=self.bias,
            ),
            nn.ReLU(inplace=True),
            nn.Linear(
                in_features=in_features,
                out_features=self.context_len // 2,
                bias=True,
            ),
            nn.ReLU(inplace=True),
            nn.Linear(
                in_features=self.context_len // 2,
                out_features=self.context_len,
                bias=True,
            ),
        )

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """"""
        for layer in self.encoder:
            x = layer(x)
            print("encode:", x.shape)
        return x

    def decode(self, x: torch.Tensor) -> torch.Tensor:
        """"""
        return self.decoder(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """CNN编码器架构的正向传播部分"""
        x_enc = self.encode(x=x)
        return self.decode(x=x_enc)


class RNNEncoder(nn.Module):
    def __init__(self):
        super(RNNEncoder, self).__init__()


class TransformerEncoder(nn.Module):
    def __init__(self):
        super(TransformerEncoder, self).__init__()


if __name__ == "__main__":
    # mlp = MLPEncoder(context_len=512, n_channels=3)
    # print(mlp)
    x = torch.randn(size=(1, 3, 512))
    # print(mlp(x).shape)

    cnn = CNNEncoder(
        context_len=512,
        n_channels=3,
        channels_list=[10, 10, 10, 10, 10],
        kernel_list=[3, 3, 3, 3, 3],
    )
    print(cnn)
    print(cnn(x).shape)
