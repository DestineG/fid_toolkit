# src/dataset/fid_dataloader.py

import os
import random
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


class FIDDataset(Dataset):
    """
    通用 FID 图像加载类
    仅接收一个图像文件夹路径，用于加载其中的所有图片。
    """
    def __init__(self, root, image_size=299, use_finetuned=False, data_ratio=1.0, shuffle=False):
        """
        参数:
            root: 图像文件夹路径
            image_size: 图像大小（默认 299）
            use_finetuned: 是否使用微调的 Inception 预处理
            data_ratio: 读取数据的比例 (0 < data_ratio <= 1)
            shuffle: 是否在采样前打乱文件顺序
        """
        super().__init__()
        self.root = root

        # 支持常见图像格式
        self.files = [
            os.path.join(self.root, f)
            for f in os.listdir(self.root)
            if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".webp"))
        ]
        if len(self.files) == 0:
            raise RuntimeError(f"{self.root} 下未找到图像文件。")

        # 控制读取比例
        if not (0 < data_ratio <= 1):
            raise ValueError("data_ratio 必须在 (0, 1] 之间。")

        if shuffle:
            random.shuffle(self.files)
        subset_len = int(len(self.files) * data_ratio)
        self.files = self.files[:subset_len]

        # Inception 模型标准预处理 或者 微调预处理
        if not use_finetuned:
            self.transform = transforms.Compose([
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                ),
            ])
        else:
            self.transform = transforms.Compose([
                transforms.Resize(342),
                transforms.CenterCrop(image_size),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                )
            ])

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        path = self.files[idx]
        img = Image.open(path).convert("RGB")
        return self.transform(img)


class FIDDataModule:
    """
    用于同时加载 real 和 fake 数据集的封装类。
    内部各自创建独立的 DataLoader。
    """
    def __init__(self, real_path, fake_path, batch_size=32, num_workers=4, image_size=299, use_finetuned=False, data_ratio=1.0):
        self.real_ds = FIDDataset(real_path, image_size=image_size, use_finetuned=use_finetuned, data_ratio=data_ratio)
        self.fake_ds = FIDDataset(fake_path, image_size=image_size, use_finetuned=use_finetuned, data_ratio=data_ratio)

        self.real_loader = DataLoader(
            self.real_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers
        )
        self.fake_loader = DataLoader(
            self.fake_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers
        )

    def get_loaders(self):
        """返回 (real_loader, fake_loader)"""
        return self.real_loader, self.fake_loader


if __name__ == "__main__":
    real_path = r"G:\雨雾模型实验对比\test\sunny2midrainy_new_triAlpha011\test_latest\images\real_B"
    fake_path = r"G:\雨雾模型实验对比\test\sunny2midrainy_new_triAlpha011\test_latest\images\fake_B"

    data = FIDDataModule(real_path, fake_path, batch_size=8)
    real_loader, fake_loader = data.get_loaders()

    print(f"✅ Real dataset: {len(data.real_ds)} images")
    print(f"✅ Fake dataset: {len(data.fake_ds)} images")

    imgs = next(iter(real_loader))
    print("Batch shape:", imgs.shape)  # [B, 3, 299, 299]
