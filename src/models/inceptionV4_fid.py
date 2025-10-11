# src/models/inceptionV4_fid.py

import torch
import timm

class InceptionV4_FID(torch.nn.Module):
    """
    Inception-V4 特征提取网络，用于 FID / sFID
    输出: [B, 1536] 特征向量
    """
    def __init__(self, device='cuda'):
        super().__init__()
        # timm 预训练 Inception-V4，num_classes=0 去掉分类头
        self.model = timm.create_model('inception_v4', pretrained=True, num_classes=0)
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad = False

        self.device = torch.device(device)
        self.model.to(self.device)

    def forward(self, x):
        """
        输入: x [B, 3, 299, 299]
        输出: [B, 1536]
        """
        x = x.to(self.device)
        with torch.no_grad():
            features = self.model(x)
        return features


if __name__ == "__main__":
    x = torch.randn(11, 3, 299, 299)

    model = InceptionV4_FID(device='cpu')  # 可改为 'cuda'
    feats = model(x)

    print("Feature shape:", feats.shape)
