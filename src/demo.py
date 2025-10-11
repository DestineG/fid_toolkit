import torch
import timm

# -----------------------------
# 1. 加载预训练 Inception-V4
# -----------------------------
model = timm.create_model('inception_v4', pretrained=True, num_classes=0)  # num_classes=0 只提特征
model.eval()

# -----------------------------
# 2. 准备一个假输入
# -----------------------------
x = torch.randn(1, 3, 299, 299)  # Batch=1, 3通道, 299x299

# -----------------------------
# 3. 前向传播
# -----------------------------
with torch.no_grad():
    features = model(x)

# -----------------------------
# 4. 打印输出特征尺寸
# -----------------------------
print("Output feature shape:", features.shape)  # 对应 FID 特征向量, 1 x 1536
