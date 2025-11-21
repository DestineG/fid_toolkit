# src/models/inceptionV3_fid.py

from typing import Tuple, Literal
import torch
from torchvision import models
from torchvision.models.feature_extraction import create_feature_extractor
import warnings

InceptionNode = Literal[
    'Mixed_5d',
    'avgpool',
    'fc'
]

class InceptionV3_FID(torch.nn.Module):
    """
    通用 FID 特征提取网络
    可同时提取多个层次节点
    输出统一为字典: {节点名: B*H*W, C}
    """
    def __init__(self, nodes: Tuple[InceptionNode, ...] = ('avgpool',), device='cuda'):
        super().__init__()
        inception = models.inception_v3(
            weights=models.Inception_V3_Weights.IMAGENET1K_V1
        )
        inception.eval()
        for p in inception.parameters():
            p.requires_grad = False

        self.device = torch.device(device)
        self.nodes = nodes

        # 屏蔽 torchvision 特征提取 eval/train 节点警告
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message=".*nodes obtained by tracing the model in eval mode.*"
            )
            # 创建节点字典
            return_nodes = {n: n for n in nodes}  # {'avgpool':'avgpool', 'Mixed_5d':'Mixed_5d'}
            self.feature_extractor = create_feature_extractor(
                inception, return_nodes=return_nodes
            ).to(self.device)

    def forward(self, x):
        """
        输入: x [B, 3, 299, 299]
        输出: dict {节点名: [B*H*W, C] 或 [B, C]}
        """
        x = x.to(self.device)
        with torch.no_grad():
            feats = self.feature_extractor(x)
            out = {}
            for name, feat in feats.items():
                if feat.dim() == 4:
                    # BCHW
                    B, C, H, W = feat.shape
                    feat = feat.permute(0, 2, 3, 1).reshape(B * H * W, C)
                elif feat.dim() == 2:
                    # BC
                    # 这里保持为 [B, C]
                    # 或者你也可以 reshape 成 B*1*1,C → B,C
                    pass
                else:
                    raise ValueError(f"Unsupported feature shape {feat.shape} at {name}")

                out[name] = feat
            return out


if __name__ == "__main__":
    x = torch.randn(1, 3, 299, 299)

    # 同时提取 Mixed_5d 和 avgpool
    model = InceptionV3_FID(nodes=('Mixed_5d', 'avgpool', 'fc'))
    feats = model(x)
    # print(model)

    for k, v in feats.items():
        print(k, v.shape)
