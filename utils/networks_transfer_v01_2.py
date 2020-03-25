'''
The transfered vectors would be quantized by AppMem
add Resblock
'''
import torch
from torch import nn
from torch.nn import functional as F

import numpy as np

# Copyright 2018 The Sonnet Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or  implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================


# Borrowed from https://github.com/deepmind/sonnet and ported it to PyTorch


class ResBlock(nn.Module):
    def __init__(self, in_channel, channel):
        super().__init__()

        self.conv = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channel, channel, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel, in_channel, 1),
        )

    def forward(self, input):
        out = self.conv(input)
        out += input

        return out


class Encoder(nn.Module):
    def __init__(self, in_channel, channel, n_res_block, n_res_channel, stride):
        super().__init__()

        if stride == 4:
            blocks = [
                nn.Conv2d(in_channel, channel // 2, 4, stride=2, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(channel // 2, channel, 4, stride=2, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(channel, channel, 3, padding=1),
            ]

        elif stride == 2:
            blocks = [
                nn.Conv2d(in_channel, channel // 2, 4, stride=2, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(channel // 2, channel, 3, padding=1),
            ]

        for i in range(n_res_block):
            blocks.append(ResBlock(channel, n_res_channel))

        blocks.append(nn.ReLU(inplace=True))

        self.blocks = nn.Sequential(*blocks)

    def forward(self, input):
        return self.blocks(input)


class Decoder(nn.Module):
    def __init__(
        self, in_channel, out_channel, channel, n_res_block, n_res_channel, stride
    ):
        super().__init__()

        blocks = [nn.Conv2d(in_channel, channel, 3, padding=1)]

        for i in range(n_res_block):
            blocks.append(ResBlock(channel, n_res_channel))

        blocks.append(nn.ReLU(inplace=True))

        if stride == 4:
            blocks.extend(
                [
                    nn.ConvTranspose2d(channel, channel // 2, 4, stride=2, padding=1),
                    nn.ReLU(inplace=True),
                    nn.ConvTranspose2d(
                        channel // 2, out_channel, 4, stride=2, padding=1
                    ),
                ]
            )

        elif stride == 2:
            blocks.append(
                nn.ConvTranspose2d(channel, out_channel, 4, stride=2, padding=1)
            )

        self.blocks = nn.Sequential(*blocks)

    def forward(self, input):
        return self.blocks(input)


class TransferModel(nn.Module):
    def __init__(
        self,
        in_channel=64,
        channel=128,
        n_res_block=2,
        n_res_channel=32,
        embed_dim=64,
        n_embed=512,
        decay=0.99,
    ):
        super().__init__()

        models_t = []
        models_b = []
        models_t.extend([ResBlock(in_channel, channel),
                         nn.ReLU(inplace=True),
                         ResBlock(channel, in_channel),
                         nn.ReLU(inplace=True),
                         ])
        models_b.extend([ResBlock(in_channel, channel),
                         nn.ReLU(inplace=True),
                         ResBlock(channel, in_channel),
                         nn.ReLU(inplace=True),
                         ])
        self.models_t = nn.Sequential(*models_t)
        self.models_b = nn.Sequential(*models_b)
        # self.enc_b = Encoder(in_channel, channel, n_res_block, n_res_channel, stride=4)
        # self.enc_t = Encoder(in_channel, channel, n_res_block, n_res_channel, stride=2)
        # self.quantize_conv_t = nn.Conv2d(channel, embed_dim, 1)
        # # self.quantize_t = Quantize(embed_dim, n_embed)
        # # self.dec_t = Decoder(
        # #     embed_dim, embed_dim, channel, n_res_block, n_res_channel, stride=2
        # # )
        # self.quantize_conv_b = nn.Conv2d(channel, embed_dim, 1)
        # # self.quantize_b = Quantize(embed_dim, n_embed)
        # # self.upsample_t = nn.ConvTranspose2d(
        # #     embed_dim, embed_dim, 4, stride=2, padding=1
        # # )
        # self.dec_b = Decoder(embed_dim, in_channel, channel, n_res_block, n_res_channel, stride=4)
        # self.dec_t = Decoder(embed_dim, in_channel, channel, n_res_block, n_res_channel, stride=2)

    def forward(self, pose_s_quant_t, pose_t_quant_t, img_s_quant_t,
                      pose_s_quant_b, pose_t_quant_b, img_s_quant_b):
        transfer_quant_t = self._transfer_warp(pose_s_quant_t, pose_t_quant_t, img_s_quant_t)
        transfer_quant_b = self._transfer_warp(pose_s_quant_b, pose_t_quant_b, img_s_quant_b)

        transfer_quant_t = self.models_t(transfer_quant_t)
        transfer_quant_b = self.models_b(transfer_quant_b)
        # quant_t = transfer_quant_t
        # # quant_t : [batch_size=25, 64, 32, 32]
        # quant_t_1 = self.enc_t(quant_t)
        # # quant_t_1 : [batch_size=25, 128, 16, 16]
        # quant_t_2 = self.quantize_conv_t(quant_t_1)
        # # quant_t_2 : [batch_size=25, 64, 16, 16]
        # quant_t_out = self.dec_t(quant_t_2)
        # # quant_t : [batch_size=25, 64, 32, 32]
        #
        # quant_b = transfer_quant_b
        # quant_b_1 = self.enc_b(quant_b)
        # # quant_t_1 : [batch_size=25, 128, 16, 16]
        # quant_b_2 = self.quantize_conv_b(quant_b_1)
        # # quant_b_2: [batch_size=25, 64, 16, 16])
        # quant_b_out = self.dec_b(quant_b_2)
        # # quant_b_out: [batch_size=25, 64, 64, 64])

        return transfer_quant_t, transfer_quant_b
        # return quant_t_out, quant_b_out

    def _transfer_warp(self, pose_s_quant, pose_t_quant, img_s_quant):
        shape_tsr = pose_s_quant.shape
        pose_s_quant = pose_s_quant.reshape([shape_tsr[0], shape_tsr[1], -1])
        pose_t_quant = pose_t_quant.reshape([shape_tsr[0], shape_tsr[1], -1]).permute(0, 2, 1)
        attention = torch.matmul(pose_t_quant, pose_s_quant)

        # TODO what is dim
        attention = F.softmax(attention, dim=1)
        transfer_quant = torch.matmul(attention,
                                      img_s_quant.reshape([shape_tsr[0], shape_tsr[1], -1]).permute(0, 2, 1)) \
            .permute(0, 2, 1).reshape([shape_tsr[0], shape_tsr[1], shape_tsr[2], shape_tsr[3]])
        return transfer_quant
