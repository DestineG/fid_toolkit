# Conv2d_1a_3x3: torch.Size([1, 32, 149, 149])
# Conv2d_2a_3x3: torch.Size([1, 32, 147, 147])
# Conv2d_2b_3x3: torch.Size([1, 64, 147, 147])
# maxpool1: torch.Size([1, 64, 73, 73])
# Conv2d_3b_1x1: torch.Size([1, 80, 73, 73])
# Conv2d_4a_3x3: torch.Size([1, 192, 71, 71])
# maxpool2: torch.Size([1, 192, 35, 35])
# Mixed_5b: torch.Size([1, 256, 35, 35])
# Mixed_5c: torch.Size([1, 288, 35, 35])
# Mixed_5d: torch.Size([1, 288, 35, 35])
# Mixed_6a: torch.Size([1, 768, 17, 17])
# Mixed_6b: torch.Size([1, 768, 17, 17])
# Mixed_6c: torch.Size([1, 768, 17, 17])
# Mixed_6d: torch.Size([1, 768, 17, 17])
# Mixed_6e: torch.Size([1, 768, 17, 17])
# Mixed_7a: torch.Size([1, 1280, 8, 8])
# Mixed_7b: torch.Size([1, 2048, 8, 8])
# Mixed_7c: torch.Size([1, 2048, 8, 8])
# avgpool: torch.Size([1, 2048, 1, 1])
# dropout: torch.Size([1, 2048, 1, 1])

from tqdm import tqdm
import torch

from src.dataset.fid_dataloader import FIDDataModule
from src.models.inceptionV3_fid import InceptionV3_FID
from src.utils.calc_metrics import fid_from_features

def calc_standard_fid(real_path, fake_path, batch_size=32, device='cuda'):
    data = FIDDataModule(real_path, fake_path, batch_size=batch_size)
    real_loader, fake_loader = data.get_loaders()

    model = InceptionV3_FID(nodes=('avgpool',), device=device)

    # real features
    real_feats = []
    for imgs in tqdm(real_loader, desc="Extracting real features"):
        imgs = imgs.to(device)
        # (B*H*W, C)
        feats = model(imgs)['avgpool']
        real_feats.append(feats.cpu())
    real_feats = torch.cat(real_feats, dim=0).numpy()

    # fake features
    fake_feats = []
    for imgs in tqdm(fake_loader, desc="Extracting fake features"):
        imgs = imgs.to(device)
        # (B*H*W, C)
        feats = model(imgs)['avgpool']
        fake_feats.append(feats.cpu())
    fake_feats = torch.cat(fake_feats, dim=0).numpy()

    # 计算 FID
    print("real_feats shape:", real_feats.shape)
    print("fake_feats shape:", fake_feats.shape)
    fid_value = fid_from_features(real_feats, fake_feats)
    return fid_value

def calc_spatial_fid(real_path, fake_path, batch_size=32, device='cuda'):
    data = FIDDataModule(real_path, fake_path, batch_size=batch_size)
    real_loader, fake_loader = data.get_loaders()

    model = InceptionV3_FID(nodes=('Mixed_5d',), device=device)

    # real features
    real_feats = []
    for imgs in tqdm(real_loader, desc="Extracting real features"):
        imgs = imgs.to(device)
        # (B*H*W, C)
        feats = model(imgs)['Mixed_5d']
        real_feats.append(feats.cpu())
    real_feats = torch.cat(real_feats, dim=0).numpy()

    # fake features
    fake_feats = []
    for imgs in tqdm(fake_loader, desc="Extracting fake features"):
        imgs = imgs.to(device)
        # (B*H*W, C)
        feats = model(imgs)['Mixed_5d']
        fake_feats.append(feats.cpu())
    fake_feats = torch.cat(fake_feats, dim=0).numpy()

    # 计算 FID
    print("real_feats shape:", real_feats.shape)
    print("fake_feats shape:", fake_feats.shape)
    fid_value = fid_from_features(real_feats, fake_feats)
    return fid_value

def calc_both_fid(real_path, fake_path, batch_size=32, device='cuda'):
    """
    一次性计算标准 FID（avgpool）和 Spatial FID（Mixed_5d）
    """
    data = FIDDataModule(real_path, fake_path, batch_size=batch_size)
    real_loader, fake_loader = data.get_loaders()

    model = InceptionV3_FID(nodes=('avgpool', 'Mixed_5d'), device=device)

    # 创建临时 list 字典存每个 batch 的特征
    real_feats_list = {'avgpool': [], 'Mixed_5d': []}
    fake_feats_list = {'avgpool': [], 'Mixed_5d': []}

    # 提取 real 特征
    for imgs in tqdm(real_loader, desc="Extracting real features"):
        imgs = imgs.to(device)
        feats = model(imgs)
        for k, v in feats.items():
            real_feats_list[k].append(v.cpu())

    # 提取 fake 特征
    for imgs in tqdm(fake_loader, desc="Extracting fake features"):
        imgs = imgs.to(device)
        feats = model(imgs)
        for k, v in feats.items():
            fake_feats_list[k].append(v.cpu())

    # 合并每个节点的特征并转换为 numpy
    real_feats = {k: torch.cat(v, dim=0).numpy() for k, v in real_feats_list.items()}
    fake_feats = {k: torch.cat(v, dim=0).numpy() for k, v in fake_feats_list.items()}

    # 计算 FID
    fid_standard = fid_from_features(real_feats['avgpool'], fake_feats['avgpool'])
    fid_spatial = fid_from_features(real_feats['Mixed_5d'], fake_feats['Mixed_5d'])

    return fid_standard, fid_spatial

if __name__ == "__main__":
    # Standard FID: 115.47917960606726
    # Spatial FID: 1.4002741699667083
    # real_path = r"G:\雨雾模型实验对比\test\sunny2midrainy_new_triAlpha011\test_latest\images\real_B"
    # fake_path = r"G:\雨雾模型实验对比\test\sunny2midrainy_new_triAlpha011\test_latest\images\fake_B"

    # Standard FID: 115.47917960606726
    # Spatial FID: 1.4002741699667083
    real_path = r"G:\雨雾模型实验对比\test\bdd100k_1_20_default\test_latest\images\real_B"
    fake_path = r"G:\雨雾模型实验对比\test\bdd100k_1_20_default\test_latest\images\fake_B"

    # python -m src.main
    fid_std, fid_spatial = calc_both_fid(real_path, fake_path, batch_size=16, device='cuda')
    print("Standard FID:", fid_std)
    print("Spatial FID:", fid_spatial)
