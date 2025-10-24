# /src/extension/inceptionV3_weather.py

import os
import glob
import random
from PIL import Image
from tqdm import tqdm
from collections import defaultdict

import torch
import torch.nn as nn
from torchvision import transforms
from torchvision.models import inception_v3, Inception_V3_Weights


# ============================================================
# 🧰 Dataset 封装
# ============================================================
class WeatherDataset(torch.utils.data.Dataset):
    def __init__(self, allowed_exts=("jpg", "jpeg", "png")):
        """
        images: List 或 Tensor，包含图像数据
        labels: List 或 Tensor，包含对应标签
        transform: 可选的图像预处理函数
        allowed_exts: 支持的图片后缀
        """
        self.labels = ["sunny1", "small_rainy", "mid_rainy", "small_foggy", "big_foggy"]
        self.dataDir = r'G:\aug'
        self.allowed_exts = allowed_exts

        # 保存每个类别的图片路径
        self.samples_per_label = {}
        for label in self.labels:
            folder = os.path.join(self.dataDir, f"{label}_640_512")
            img_paths = []
            for ext in self.allowed_exts:
                img_paths.extend(glob.glob(os.path.join(folder, f"*.{ext}")))
            if len(img_paths) == 0:
                print(f"⚠ Warning: folder {folder} has no images with allowed extensions {self.allowed_exts}")
            self.samples_per_label[label] = img_paths

        # transform
        self.transform = transforms.Compose([
            transforms.Resize(342),                 
            transforms.CenterCrop(299),            
            transforms.ToTensor(),                 
            transforms.Normalize(                  
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

    def __len__(self):
        return 1000  # 无限采样

    def __getitem__(self, idx):
        # 随机选择一个类别
        label = random.choice(self.labels)
        img_paths = self.samples_per_label[label]
        if len(img_paths) == 0:
            raise RuntimeError(f"No images found for label {label} with allowed extensions {self.allowed_exts}")

        # 随机选择该类别的一张图片
        img_path = random.choice(img_paths)
        img = Image.open(img_path).convert("RGB")

        if self.transform:
            img = self.transform(img)

        label_idx = self.labels.index(label)
        return img, label_idx


# ============================================================
# 🧩 模型定义
# ============================================================
class InceptionV3_Weather(nn.Module):
    def __init__(self, num_classes=5, pretrained=True):
        super().__init__()
        base = inception_v3(weights=Inception_V3_Weights.DEFAULT if pretrained else None, init_weights=False)

        # ---- 主干特征部分 ----
        self.features = nn.Sequential(
            base.Conv2d_1a_3x3,
            base.Conv2d_2a_3x3,
            base.Conv2d_2b_3x3,
            base.maxpool1,
            base.Conv2d_3b_1x1,
            base.Conv2d_4a_3x3,
            base.maxpool2,
            base.Mixed_5b,
            base.Mixed_5c,
            base.Mixed_5d,
            base.Mixed_6a,
            base.Mixed_6b,
            base.Mixed_6c,
            base.Mixed_6d,
            base.Mixed_6e,
            base.Mixed_7a,
            base.Mixed_7b,
            base.Mixed_7c
        )

        # ---- 池化 + 分类层 ----
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(base.fc.in_features, num_classes)

        self.return_features = False  # 是否返回特征

    def forward(self, x):
        """return_features=True 时返回 (features, logits)"""
        x = self.features(x)
        features = torch.flatten(self.pool(x), 1)
        logits = self.fc(features)
        return features if self.return_features else logits
    
    def setup_for_test(self, checkpoint_path, device=None):
        """设置模型以进行测试"""
        device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.load_state_dict(torch.load(checkpoint_path, map_location=device))
        self.to(device)
        self.eval()
        for p in self.parameters():
            p.requires_grad = False

    def setup_for_fid(self, checkpoint_path="./weights/inceptionv3_epoch55_loss0_0931.pth", device=None):
        """设置模型以提取 FID 特征"""
        device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.load_state_dict(torch.load(checkpoint_path, map_location=device))
        self.to(device)
        self.eval()
        for p in self.parameters():
            p.requires_grad = False
        self.return_features = True


# ============================================================
# 🧰 Trainer 封装
# ============================================================
class WeatherTrainer:
    def __init__(self, model, lr=1e-4, device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)

    def train(self, data_loader, epochs=2, save_interval=1, savedir=None):
        """
        data_loader: DataLoader
        epochs: 总训练轮数
        save_interval: 每隔多少个 epoch 保存一次
        savedir: 模型权重保存目录（每次保存都会加上 epoch 后缀）
        """
        if savedir is not None:
            os.makedirs(savedir, exist_ok=True)

        self.model.train()
        for epoch in range(1, epochs + 1):
            total_loss = 0
            for imgs, labels in tqdm(data_loader, desc=f"Epoch {epoch}/{epochs}"):
                imgs, labels = imgs.to(self.device), labels.to(self.device)
                logits = self.model(imgs)
                loss = self.criterion(logits, labels)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item()
            avg_loss = total_loss / len(data_loader)
            print(f"Epoch {epoch} | Avg Loss: {avg_loss:.4f}")

            # ---- 每隔 save_interval 个 epoch 保存一次权重 ----
            if savedir is not None and epoch % save_interval == 0:
                # 保留两位小数
                loss_str = f"{avg_loss:.4f}".replace(".", "_")
                save_path = os.path.join(savedir, f"inceptionv3_epoch{epoch}_loss{loss_str}.pth")
                torch.save(self.model.state_dict(), save_path)
                print(f"✅ Saved model weights to {save_path}")

    def extract_features(self, x):
        """返回 2048-D FID 特征"""
        self.model.eval()
        with torch.no_grad():
            features, _ = self.model(x.to(self.device), return_features=True)
        return features


# ============================================================
# 🧪 main：随机张量测试
# ============================================================
def main():
    batch_size = 16
    epochs = 1000
    save_dir = "./weights"

    # ---- 初始化模型与训练器 ----
    model = InceptionV3_Weather()
    trainer = WeatherTrainer(model, lr=1e-4)

    # ---- 使用真实 WeatherDataset ----
    dataset = WeatherDataset()
    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=True, num_workers=4
    )
    print("🚀 Using real dataset for training")

    # ---- 训练 ----
    trainer.train(
        data_loader, 
        epochs=epochs, 
        save_interval=5, 
        savedir=save_dir  # 保存到 weights 文件夹
    )


def test():
    # ---- 模型参数 ----
    weights_path = "./weights/inceptionv3_epoch55_loss0_0931.pth"  # 训练好的权重
    device="cuda" if torch.cuda.is_available() else "cpu"

    # ---- 初始化模型 ----
    model = InceptionV3_Weather(pretrained=False)
    model.setup_for_test(weights_path, device=device)

    # ---- 数据集 & DataLoader ----
    dataset = WeatherDataset()
    labels_list = dataset.labels
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=16, shuffle=False, num_workers=2)

    total = 0
    correct = 0
    per_class_total = defaultdict(int)
    per_class_correct = defaultdict(int)

    with torch.no_grad():
        for imgs, labels in tqdm(data_loader, desc="Testing"):
            imgs, labels = imgs.to(device), labels.to(device)
            logits = model(imgs)
            preds = torch.argmax(logits, dim=1)

            correct += (preds == labels).sum().item()
            total += labels.size(0)

            # 每类统计
            for i in range(labels.size(0)):
                label = labels[i].item()
                per_class_total[label] += 1
                if preds[i].item() == label:
                    per_class_correct[label] += 1

    overall_acc = correct / total
    print(f"\nOverall Accuracy: {overall_acc*100:.2f}%")

    print("Per-class Accuracy:")
    for idx, label_name in enumerate(labels_list):
        if per_class_total[idx] > 0:
            acc = per_class_correct[idx] / per_class_total[idx]
            print(f"  {label_name}: {acc*100:.2f}%")
        else:
            print(f"  {label_name}: No samples")


if __name__ == "__main__":
    # main()
    test()
