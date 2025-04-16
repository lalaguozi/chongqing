######################  MSCAAttention ####     START   by  AI&CV  ###############################


import torch
import torch.nn as nn
from torch.nn import functional as F
from ultralytics.nn.modules.conv import Conv

class MSCAAttention(nn.Module):
    # SegNext NeurIPS 2022
    # https://github.com/Visual-Attention-Network/SegNeXt/tree/main
    def __init__(self, dim):
        super().__init__()
        self.conv0 = nn.Conv2d(dim, dim, 5, padding=2, groups=dim)
        self.conv0_1 = nn.Conv2d(dim, dim, (1, 7), padding=(0, 3), groups=dim)
        self.conv0_2 = nn.Conv2d(dim, dim, (7, 1), padding=(3, 0), groups=dim)

        self.conv1_1 = nn.Conv2d(dim, dim, (1, 11), padding=(0, 5), groups=dim)
        self.conv1_2 = nn.Conv2d(dim, dim, (11, 1), padding=(5, 0), groups=dim)

        self.conv2_1 = nn.Conv2d(dim, dim, (1, 21), padding=(0, 10), groups=dim)
        self.conv2_2 = nn.Conv2d(dim, dim, (21, 1), padding=(10, 0), groups=dim)
        self.conv3 = nn.Conv2d(dim, dim, 1)

    def forward(self, x):
        u = x.clone()
        attn = self.conv0(x)

        attn_0 = self.conv0_1(attn)
        attn_0 = self.conv0_2(attn_0)

        attn_1 = self.conv1_1(attn)
        attn_1 = self.conv1_2(attn_1)

        attn_2 = self.conv2_1(attn)
        attn_2 = self.conv2_2(attn_2)
        attn = attn + attn_0 + attn_1 + attn_2

        attn = self.conv3(attn)

         # 自注意力处理
        # b, c, h, w = attn.shape
        # attn_flat = attn.view(b, c, h * w).permute(2, 0, 1)  # [H*W, B, C]
        # attn_flat, _ = self.self_attn(attn_flat, attn_flat, attn_flat)
        # attn = attn_flat.permute(1, 2, 0).view(b, c, h, w)  # 恢复形状

        return attn * u



###################### MSCAAttention  ####     end   by  AI&CV  ###############################