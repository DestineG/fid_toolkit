# src/models/clip_fid.py

import torch
import clip

class CLIP_FID(torch.nn.Module):
    """
    CLIP 特征提取网络，用于 FID / CLIP-FID
    输出: [B, D] 特征向量 (D=512 或 768，取决于模型)
    """
    def __init__(self, model_name='ViT-B/32', device='cuda'):
        super().__init__()

        # 加载 CLIP 预训练模型
        self.model, self.preprocess = clip.load(model_name, device=device)
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad = False

        self.device = torch.device(device)

    def forward(self, x):
        """
        输入: x [B, 3, H, W], 值域 [0,1] 或 [0,255]
        输出: [B, D] 归一化特征
        """
        x = x.to(self.device)
        if x.max() > 1:
            x = x / 255.0  # 转到 [0,1] 范围

        # CLIP 输入尺寸通常为 224×224
        x = torch.nn.functional.interpolate(x, size=(224, 224), mode='bilinear', align_corners=False)

        with torch.no_grad():
            feats = self.model.encode_image(x)
            feats = feats / feats.norm(dim=-1, keepdim=True)  # L2 归一化
        return feats


if __name__ == "__main__":
    x = torch.randn(8, 3, 256, 256)
    model = CLIP_FID(model_name='ViT-B/32', device='cpu')
    feats = model(x)
    print("Feature shape:", feats.shape)  # -> [8, 512]
