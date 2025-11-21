# /src/utils/calc_metrics.py

import numpy as np
from scipy.linalg import sqrtm
import warnings

def fid_from_features(feat_real: np.ndarray, feat_fake: np.ndarray) -> float:
    """
    计算 FID，稳定版本，屏蔽 ComplexWarning
    输入:
        feat_real: [N_real, C] np.ndarray
        feat_fake: [N_fake, C] np.ndarray
    输出:
        FID score (float)
    """
    # 屏蔽含 “Casting complex values to real” 的警告
    warnings.filterwarnings("ignore", message=".*Casting complex values to real.*")

    feat_real = feat_real.astype(np.float32)
    feat_fake = feat_fake.astype(np.float32)

    # 均值
    mu1 = np.mean(feat_real, axis=0)
    mu2 = np.mean(feat_fake, axis=0)

    # 协方差
    sigma1 = np.cov(feat_real, rowvar=False)
    sigma2 = np.cov(feat_fake, rowvar=False)

    diff = mu1 - mu2

    # 矩阵平方根
    covmean, _ = sqrtm(sigma1 @ sigma2, disp=False)
    covmean = np.real_if_close(covmean)

    # 计算 FID
    diff = diff.dot(diff)
    trace = np.trace(sigma1 + sigma2 - 2 * covmean)
    fid = diff + trace
    print(f"FID calculation details: diff={diff}, trace={trace}")
    return float(fid)

def calc_inception_score(prob_array, eps=1e-16):
    """
    prob_array: [N, num_cls]
    """
    p_y = np.mean(prob_array, axis=0, keepdims=True)

    kl = prob_array * (np.log(prob_array + eps) - np.log(p_y + eps))
    kl_mean = np.mean(np.sum(kl, axis=1))

    return float(np.exp(kl_mean))


if __name__ == "__main__":
    feat_real = np.random.randn(100, 2048)
    feat_fake = np.random.randn(100, 2048)

    fid_score = fid_from_features(feat_real, feat_fake)
    print("FID:", fid_score)
