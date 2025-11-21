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

import random, shutil, tempfile, traceback
from pathlib import Path
from tqdm import tqdm
import torch

from src.dataset.fid_dataloader import FIDDataModule
from src.models.inceptionV3_fid import InceptionV3_FID
from src.models.inceptionV4_fid import InceptionV4_FID
from src.models.clip_fid import CLIP_FID
from src.extension.inceptionV3_weather import InceptionV3_Weather
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

def calc_standard_fid_weather(real_path, fake_path, batch_size=32, device='cuda'):
    data = FIDDataModule(real_path, fake_path, batch_size=batch_size, use_finetuned=True)
    real_loader, fake_loader = data.get_loaders()

    model = InceptionV3_Weather(pretrained=False)
    model.setup_for_fid(device=device)

    # real features
    real_feats = []
    for imgs in tqdm(real_loader, desc="Extracting real features"):
        imgs = imgs.to(device)
        # (B*H*W, C)
        feats = model(imgs)
        real_feats.append(feats.cpu())
    real_feats = torch.cat(real_feats, dim=0).numpy()

    # fake features
    fake_feats = []
    for imgs in tqdm(fake_loader, desc="Extracting fake features"):
        imgs = imgs.to(device)
        # (B*H*W, C)
        feats = model(imgs)
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

def calc_standard_fidV4(real_path, fake_path, batch_size=32, device='cuda'):
    data = FIDDataModule(real_path, fake_path, batch_size=batch_size)
    real_loader, fake_loader = data.get_loaders()

    model = InceptionV4_FID(device=device)

    # real features
    real_feats = []
    for imgs in tqdm(real_loader, desc="Extracting real features"):
        imgs = imgs.to(device)
        # (B*H*W, C)
        feats = model(imgs)
        real_feats.append(feats.cpu())
    real_feats = torch.cat(real_feats, dim=0).numpy()

    # fake features
    fake_feats = []
    for imgs in tqdm(fake_loader, desc="Extracting fake features"):
        imgs = imgs.to(device)
        # (B*H*W, C)
        feats = model(imgs)
        fake_feats.append(feats.cpu())
    fake_feats = torch.cat(fake_feats, dim=0).numpy()

    # 计算 FID
    print("real_feats shape:", real_feats.shape)
    print("fake_feats shape:", fake_feats.shape)
    fid_value = fid_from_features(real_feats, fake_feats)
    return fid_value

# 语义相似度
def calc_clip_fid(real_path, fake_path, batch_size=32, device='cuda'):
    data = FIDDataModule(real_path, fake_path, batch_size=batch_size, image_size=224)
    real_loader, fake_loader = data.get_loaders()

    model = CLIP_FID(device=device)

    # real features
    real_feats = []
    for imgs in tqdm(real_loader, desc="Extracting real features"):
        imgs = imgs.to(device)
        # (B*H*W, C)
        feats = model(imgs)
        real_feats.append(feats.cpu())
    real_feats = torch.cat(real_feats, dim=0).numpy()

    # fake features
    fake_feats = []
    for imgs in tqdm(fake_loader, desc="Extracting fake features"):
        imgs = imgs.to(device)
        # (B*H*W, C)
        feats = model(imgs)
        fake_feats.append(feats.cpu())
    fake_feats = torch.cat(fake_feats, dim=0).numpy()

    # 计算 FID
    print("real_feats shape:", real_feats.shape)
    print("fake_feats shape:", fake_feats.shape)
    fid_value = fid_from_features(real_feats, fake_feats)
    return fid_value

def compute_self_fid(dataset_path, fid_func, split_ratio=0.5, batch_size=32, device='cuda', keep_temp=False):
    """
    在一个数据集上自动划分两部分计算 self-FID
    """
    exts = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
    all_imgs = [str(p) for p in Path(dataset_path).rglob("*") if p.suffix.lower() in exts]
    if len(all_imgs) < 10:
        raise ValueError("数据太少，无法计算 self-FID")

    # random.shuffle(all_imgs)
    split_point = int(len(all_imgs) * split_ratio)
    real_imgs, fake_imgs = all_imgs[:split_point], all_imgs[split_point:]

    # ✅ 创建临时目录
    tmp_root = Path(tempfile.mkdtemp(prefix="self_fid_"))
    real_dir, fake_dir = tmp_root / "real", tmp_root / "fake"

    try:
        # ✅ 所有可能出错的逻辑都放在 try 内
        real_dir.mkdir(parents=True, exist_ok=True)
        fake_dir.mkdir(parents=True, exist_ok=True)

        # 拷贝部分图片
        for i, src in enumerate(real_imgs):
            shutil.copy(src, real_dir / f"{i:06d}{Path(src).suffix}")
        for i, src in enumerate(fake_imgs):
            shutil.copy(src, fake_dir / f"{i:06d}{Path(src).suffix}")

        print(f"✔ 临时划分完成: real={len(real_imgs)}, fake={len(fake_imgs)}")
        print(f"📁 临时路径: {tmp_root}")

        # 计算 FID
        fid_value = fid_func(str(real_dir), str(fake_dir), batch_size=batch_size, device=device)
        return fid_value

    except Exception as e:
        print("❌ self-FID 计算过程中出错：")
        traceback.print_exc()
        return None

    finally:
        # ✅ 无论是否出错，都执行清理逻辑
        if not keep_temp:
            shutil.rmtree(tmp_root, ignore_errors=True)
            print("🧹 临时目录已清理")
        else:
            print(f"⚠ 保留临时目录: {tmp_root}")


# python -m src.main
if __name__ == "__main__":
    # fid估计

    # Standard FID: 19.334657457878283
    # Spatial FID: 
    # Standard FIDV4: 13.671175358833805
    # CLIP FID: 4.53121456681344e-06
    # Self FID: 15.63691913239051(calc_standard_fid shuffle)
    # real_path = r"G:\雨雾模型实验对比\test\bdd100k_1_20_default\test_latest\images\real_B"
    # fake_path = r"G:\雨雾模型实验对比\test\bdd100k_1_20_default\test_latest\images\fake_B"

    # Standard FID: 19.65009738373634
    # Spatial FID: 
    # Standard FIDV4: 14.423221534487606
    # CLIP FID: 3.4266141302199352e-06
    # Self FID: 15.317200572103815(calc_standard_fid shuffle)
    # real_path = r"G:\雨雾模型实验对比\test\bdd100k_1_20_AB_tri0\test_latest\images\real_B"
    # fake_path = r"G:\雨雾模型实验对比\test\bdd100k_1_20_AB_tri0\test_latest\images\fake_B"

    # Standard FID: 115.47917960606726
    # Spatial FID: 1.4002741699667083
    # Standard FIDV4: 117.53998041378354
    # CLIP FID: 0.38515492563833625
    # Self FID: 52.834578067625415(calc_standard_fid)
    # real_path = r"G:\雨雾模型实验对比\test\sunny2midrainy_new_triAlpha011\test_latest\images\real_B"
    # fake_path = r"G:\雨雾模型实验对比\test\sunny2midrainy_new_triAlpha011\test_latest\images\fake_B"

    # Standard FID: 93.4126887919851
    # Spatial FID: 0.6372413260083234
    # Standard FIDV4: 102.39758010107957
    # CLIP FID: 0.35653384475078276
    # Self FID: 52.03511432028108(calc_standard_fid)
    # real_path = r"G:\雨雾模型实验对比\test\\sunny2midrainy_new_triAlpha011_1280\test_latest\images\real_B"
    # fake_path = r"G:\雨雾模型实验对比\test\\sunny2midrainy_new_triAlpha011_1280\test_latest\images\fake_B"

    # Standard FID: 40.41291449127269
    # Spatial FID: 
    # Standard FIDV4: 
    # CLIP FID: 
    # Self FID: (calc_standard_fid)
    # real_path = r"G:\雨雾模型实验对比\test\sunny2midrainy_aug_default\test_10\images\real_B"
    # fake_path = r"G:\雨雾模型实验对比\test\sunny2midrainy_aug_default\test_10\images\fake_B"

    # Standard FID: 32.75868479692938
    # Spatial FID: 
    # Standard FIDV4: 
    # CLIP FID: 
    # Self FID: (calc_standard_fid)
    # real_path = r"G:\雨雾模型实验对比\test\sunny2midrainy_aug_default\test_15\images\real_B"
    # fake_path = r"G:\雨雾模型实验对比\test\sunny2midrainy_aug_default\test_15\images\fake_B"

    # Standard FID: 37.803960369579265
    # Spatial FID: 
    # Standard FIDV4: 
    # CLIP FID: 
    # Self FID: (calc_standard_fid)
    # real_path = r"G:\雨雾模型实验对比\test\sunny2midrainy_aug_default\test_20\images\real_B"
    # fake_path = r"G:\雨雾模型实验对比\test\sunny2midrainy_aug_default\test_20\images\fake_B"

    # Standard FID: 38.44904141482934
    # Spatial FID: 
    # Standard FIDV4: 
    # CLIP FID: 
    # Self FID: (calc_standard_fid)
    # real_path = r"G:\雨雾模型实验对比\test\sunny2midrainy_aug_default\test_25\images\real_B"
    # fake_path = r"G:\雨雾模型实验对比\test\sunny2midrainy_aug_default\test_25\images\fake_B"

    # Standard FID: 26.124894112482234(diff=4.402442932128906, trace=21.72245118035333)
    # Spatial FID: 
    # Standard FIDV4: 
    # CLIP FID: 
    # Self FID: (calc_standard_fid)
    # real_path = r"G:\雨雾模型实验对比\test\sunny2midrainy_aug_default\test_30\images\real_B"
    # fake_path = r"G:\雨雾模型实验对比\test\sunny2midrainy_aug_default\test_30\images\fake_B"

    # Standard FID: 34.57874908631109
    # Spatial FID: 
    # Standard FIDV4: 
    # CLIP FID: 
    # Self FID: (calc_standard_fid)
    # real_path = r"G:\雨雾模型实验对比\test\sunny2midrainy_aug_default\test_35\images\real_B"
    # fake_path = r"G:\雨雾模型实验对比\test\sunny2midrainy_aug_default\test_35\images\fake_B"

    # Standard FID: 27.284779004811647
    # Spatial FID: 
    # Standard FIDV4: 
    # CLIP FID: 
    # Self FID: (calc_standard_fid)
    # real_path = r"G:\雨雾模型实验对比\test\sunny2midrainy_aug_default\test_40\images\real_B"
    # fake_path = r"G:\雨雾模型实验对比\test\sunny2midrainy_aug_default\test_40\images\fake_B"

    # Standard FID: 36.689509606946054
    # Spatial FID: 
    # Standard FIDV4: 
    # CLIP FID: 
    # Self FID: (calc_standard_fid)
    # real_path = r"G:\雨雾模型实验对比\test\sunny2midrainy_aug_default\test_45\images\real_B"
    # fake_path = r"G:\雨雾模型实验对比\test\sunny2midrainy_aug_default\test_45\images\fake_B"

    # Standard FID: 36.658175372059674
    # Spatial FID: 
    # Standard FIDV4: 
    # CLIP FID: 
    # Self FID: (calc_standard_fid)
    # real_path = r"G:\雨雾模型实验对比\test\sunny2midrainy_aug_default\test_50\images\real_B"
    # fake_path = r"G:\雨雾模型实验对比\test\sunny2midrainy_aug_default\test_50\images\fake_B"

    # Standard FID: 83.37788074114476(diff=30.512704849243164, trace=52.86517589190161)
    # Spatial FID: 
    # Standard FIDV4: 
    # CLIP FID: 
    # Self FID: (calc_standard_fid)
    # real_path = r"G:\雨雾模型实验对比\test\sunny2midrainy_aug_default\test_latest\images\real_B"
    # fake_path = r"G:\雨雾模型实验对比\test\sunny2midrainy_aug_default\test_latest\images\fake_B"

    # Standard FID: 70.23100119024217(diff=16.47844123840332, trace=53.75255995183885)
    # real_path = r"G:\aug\sunny2bigfoggy_aug\testB"
    # fake_path = r"G:\aug\sunny2bigfoggy_aug\outputs"

    # 1 epoch
    # Standard FID: 86.74668894261953(diff=32.16731262207031, trace=54.579376320549216)
    # real_path = r"G:\雨雾模型实验对比\test\sunny12foggy_aug_LHD_Train\real_B"
    # fake_path = r"G:\雨雾模型实验对比\test\sunny12foggy_aug_LHD_Train\fake_B_epoch1"

    # 5 epoch
    # Standard FID: 87.56510984051224(diff=31.51609992980957, trace=56.04900991070266)
    # real_path = r"G:\雨雾模型实验对比\test\sunny12foggy_aug_LHD_Train\real_B"
    # fake_path = r"G:\雨雾模型实验对比\test\sunny12foggy_aug_LHD_Train\fake_B_epoch5"

    # Standard FID: 40.20251339593317(diff=10.052574157714844, trace=30.14993923821832)
    # Spatial FID: 
    # Standard FIDV4: 
    # CLIP FID: 
    # Self FID: (calc_standard_fid)
    # real_path = r"G:\雨雾模型实验对比\test\sunny12midrainy_aug_default_testTRI\test_30\images\real_B"
    # fake_path = r"G:\雨雾模型实验对比\test\sunny12midrainy_aug_default_testTRI\test_30\images\fake_B"

    # Standard FID: 187.07449879953523(diff=21.001874923706055, trace=(166.07262387582918-1.579588551845214e-06j))
    # Spatial FID: 3.0905390168003914(diff=0.9482430815696716, trace=2.14229593523072)
    # Standard FIDV4: 
    # CLIP FID: 
    # Self FID: (calc_standard_fid)
    real_path = r"G:\雨雾模型数据集\实验结果\v4\test3\sunny12bigfoggy_aug_default_testTRI_ASNLoss_ori\test_latest\images\real_B"
    fake_path = r"G:\雨雾模型数据集\实验结果\v4\test3\sunny12bigfoggy_aug_default_testTRI_ASNLoss_ori\test_latest\images\fake_B"

    # fid下限估计

    # Standard FID: 23.940437518594827
    # real_path = r"G:\实验室服务器1内容备份\data\datasets\bdd100k_1_20\trainA"
    # fake_path = r"G:\实验室服务器1内容备份\data\datasets\bdd100k_1_20\trainB"

    # Standard FID: 24.84805000888763
    # real_path = r"G:\实验室服务器1内容备份\data\datasets\bdd100k_1_20\testA"
    # fake_path = r"G:\实验室服务器1内容备份\data\datasets\bdd100k_1_20\testB"

    # Standard FID: 177.4837225338313
    # real_path = r"G:\无人机实测数据整理\新\晴天\热成像_Thermal_T"
    # fake_path = r"G:\无人机实测数据整理\新\中雨\热成像_Thermal_T"

    # fid上限估计

    # Self FID: 19.446476164826954(calc_standard_fid)
    # real_path = r"G:\实验室服务器1内容备份\data\datasets\bdd100k_1_20\testB"

    # Self FID: 107.0260542818958(calc_standard_fid)
    # real_path = r"G:\无人机实测数据整理\新\中雨\热成像_Thermal_T"

    # Self FID: 113.84457093660306(calc_standard_fid)
    # CLIP FID: 
    # real_path = r"G:\无人机实测数据整理\新\晴天\热成像_Thermal_T"

    # Self FID: 37.43724729717688(calc_standard_fid)
    # real_path = r"G:\aug\sunny2midrainy_aug\testB"

    # Self FID: 28.424124005723744(calc_standard_fid)
    # real_path = r"G:\aug\sunny2midrainy_aug\trainB"

    # Self FID: 27.36940663571233(diff=4.896724700927734, trace=22.472681934784596)
    # real_path = r"G:\datasets\aug_overlap\mid_rainy_640_512_overlap"

    # 结果计算

    # fid_self = compute_self_fid(real_path, calc_standard_fid, batch_size=32, device='cuda')
    # print("Self FID:", fid_self)

    fid_std = calc_standard_fid(real_path, fake_path, batch_size=32, device='cuda')
    print("data_real_path: {}\nStandard FID: {}".format(real_path, fid_std))

    # fid_std_weather = calc_standard_fid_weather(real_path, fake_path, batch_size=32, device='cuda')
    # print("data_real_path: {}\nStandard FID Weather: {}".format(real_path, fid_std_weather))

    fid_spatial = calc_spatial_fid(real_path, fake_path, batch_size=32, device='cuda')
    print("Spatial FID:", fid_spatial)

    # fid_stdV4 = calc_standard_fidV4(real_path, fake_path, batch_size=32, device='cuda')
    # print("Standard FIDV4:", fid_stdV4)

    # fid_clip = calc_clip_fid(real_path, fake_path, batch_size=64, device='cuda')
    # print("CLIP FID:", fid_clip)