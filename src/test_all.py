from tqdm import tqdm
import torch
import torch.nn.functional as F
import os
from PIL import Image
import numpy as np
from src.dataset.fid_dataloader import FIDDataModule
from src.models.inceptionV3_fid import InceptionV3_FID
from src.utils.calc_metrics import fid_from_features, calc_inception_score


# --------------------------------------------------------
# FID & Spatial FID
# --------------------------------------------------------
def evaluate_fid(real_path, fake_path, batch_size=32, device='cuda'):
    data = FIDDataModule(real_path, fake_path, batch_size=batch_size)
    real_loader, fake_loader = data.get_loaders()

    # 一次性提取 Mixed_5d + avgpool
    model = InceptionV3_FID(nodes=('Mixed_5d', 'avgpool'), device=device)

    real_feats_m5d, real_feats_avg = [], []
    fake_feats_m5d, fake_feats_avg = [], []

    # ----------- real ----------- #
    for imgs in tqdm(real_loader, desc="Extracting real features"):
        imgs = imgs.to(device)
        feats = model(imgs)

        real_feats_m5d.append(feats['Mixed_5d'].cpu())
        real_feats_avg.append(feats['avgpool'].cpu())

    real_feats_m5d = torch.cat(real_feats_m5d, dim=0).numpy()
    real_feats_avg = torch.cat(real_feats_avg, dim=0).numpy()

    # ----------- fake ----------- #
    for imgs in tqdm(fake_loader, desc="Extracting fake features"):
        imgs = imgs.to(device)
        feats = model(imgs)

        fake_feats_m5d.append(feats['Mixed_5d'].cpu())
        fake_feats_avg.append(feats['avgpool'].cpu())

    fake_feats_m5d = torch.cat(fake_feats_m5d, dim=0).numpy()
    fake_feats_avg = torch.cat(fake_feats_avg, dim=0).numpy()

    # ----------- 计算 FID ----------- #
    fid_value = fid_from_features(real_feats_avg, fake_feats_avg)
    spatial_fid_value = fid_from_features(real_feats_m5d, fake_feats_m5d)

    return {
        "fid": fid_value,
        "spatial_fid": spatial_fid_value
    }

# --------------------------------------------------------
# Inception Score
# --------------------------------------------------------
def evaluate_inception_score(fake_path, batch_size=32, device="cuda"):
    """
    单独计算 Inception Score
    """
    data = FIDDataModule(None, fake_path, batch_size=batch_size)  # 只需要 fake 数据
    _, fake_loader = data.get_loaders()

    model = InceptionV3_FID(nodes=('fc',), device=device)

    fake_probs = []

    for imgs in tqdm(fake_loader, desc="Extracting features for IS"):
        imgs = imgs.to(device)
        feats = model(imgs)

        p_yx = F.softmax(feats['fc'], dim=1)
        fake_probs.append(p_yx.cpu())

    fake_probs = torch.cat(fake_probs, dim=0).numpy()

    return calc_inception_score(fake_probs)

# --------------------------------------------------------
# ISNR
# --------------------------------------------------------
def calc_isnr(real_path, fake_path, max_val=255.0):
    real_files = sorted([f for f in os.listdir(real_path) if f.lower().endswith(('png','jpg','jpeg'))])
    fake_files = sorted([f for f in os.listdir(fake_path) if f.lower().endswith(('png','jpg','jpeg'))])

    assert len(real_files) == len(fake_files), "real/fake 图像数量不一致"

    isnr_list = []
    for rf, ff in tqdm(zip(real_files, fake_files), total=len(real_files)):
        real_img = np.array(Image.open(os.path.join(real_path, rf)).convert("RGB"), dtype=np.float32)
        fake_img = np.array(Image.open(os.path.join(fake_path, ff)).convert("RGB"), dtype=np.float32)

        mse = np.mean((real_img - fake_img)**2)
        if mse == 0:
            isnr = float('inf')
        else:
            isnr = 10 * np.log10(max_val**2 / mse)
        isnr_list.append(isnr)

    return {"isnr_mean": float(np.mean(isnr_list)), "isnr_list": isnr_list}


if __name__ == '__main__':
    real_path = r"G:\雨雾模型数据集\实验结果\v4\test3\sunny12bigfoggy_aug_default_testTRI_ASNLoss_ori\test_latest\images\real_B"
    fake_path = r"G:\雨雾模型数据集\实验结果\v4\test3\sunny12bigfoggy_aug_default_testTRI_ASNLoss_ori\test_latest\images\fake_B"
    results = evaluate_fid(real_path, fake_path, batch_size=32, device='cuda')
    print(results)

    image_path = r"G:\雨雾模型数据集\实验结果\v4\test3\sunny12bigfoggy_aug_default_testTRI_ASNLoss_ori\test_latest\images\fake_B"
    is_value = evaluate_inception_score(image_path, batch_size=32, device='cuda')
    print("Inception Score:", is_value)

    input_path = r"G:\雨雾模型数据集\实验结果\v4\test3\sunny12bigfoggy_aug_default_testTRI_ASNLoss_ori\test_latest\images\real_A"
    output_path = r"G:\雨雾模型数据集\实验结果\v4\test3\sunny12bigfoggy_aug_default_testTRI_ASNLoss_ori\test_latest\images\fake_B"
    isnr_results = calc_isnr(input_path, output_path, max_val=255.0)
    print(isnr_results.get("isnr_mean"))