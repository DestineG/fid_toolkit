# src/dataset/fid_dataloader.py

import os
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


class FIDDataset(Dataset):
    """
    通用 FID 图像加载类
    仅接收一个图像文件夹路径，用于加载其中的所有图片。
    """
    def __init__(self, root, image_size=299, use_finetuned=False):
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
    如果路径不合法，会返回 '_'
    """
    def __init__(self, real_path, fake_path, batch_size=32, num_workers=4, image_size=299, use_finetuned=False):
        # 路径检查
        if not (real_path and os.path.exists(real_path) and os.path.isdir(real_path)):
            print(f"Warning: real_path '{real_path}' 不存在或不是文件夹")
            self.real_loader = "_"
        else:
            self.real_ds = FIDDataset(real_path, image_size=image_size, use_finetuned=use_finetuned)
            self.real_loader = DataLoader(
                self.real_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers
            )

        if not (fake_path and os.path.exists(fake_path) and os.path.isdir(fake_path)):
            print(f"Warning: fake_path '{fake_path}' 不存在或不是文件夹")
            self.fake_loader = "_"
        else:
            self.fake_ds = FIDDataset(fake_path, image_size=image_size, use_finetuned=use_finetuned)
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
