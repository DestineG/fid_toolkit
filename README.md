# FID (Fréchet Inception Distance) Calculator

A comprehensive Python implementation for calculating FID scores between real and generated images using multiple Inception architectures.

## Features

- **Standard FID**: Uses avgpool features from InceptionV3
- **Spatial FID**: Uses Mixed_5d features for spatial-aware comparison  
- **InceptionV4 FID**: Alternative FID calculation using InceptionV4 architecture
- **Batch processing**: Efficient feature extraction with customizable batch sizes
- **GPU support**: CUDA acceleration for faster computation
- **Multiple architectures**: Support for both InceptionV3 and InceptionV4 models

## Installation

1. Clone this repository:
```bash
git clone <repository-url>
cd fid
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Basic Usage

```python
from src.main import calc_standard_fid, calc_spatial_fid, calc_both_fid, calc_standard_fidV4

# Calculate standard FID using InceptionV3
fid_score = calc_standard_fid(real_path, fake_path, batch_size=32, device='cuda')

# Calculate spatial FID using InceptionV3 Mixed_5d features
fid_score = calc_spatial_fid(real_path, fake_path, batch_size=32, device='cuda')

# Calculate both standard and spatial FID scores at once
fid_std, fid_spatial = calc_both_fid(real_path, fake_path, batch_size=32, device='cuda')

# Calculate FID using InceptionV4
fid_score_v4 = calc_standard_fidV4(real_path, fake_path, batch_size=32, device='cuda')
```

### Command Line Usage

```bash
python -m src.main
```

Make sure to update the `real_path` and `fake_path` variables in `src/main.py` before running.

## Project Structure

```
fid/
├── src/
│   ├── main.py                 # Main calculation functions
│   ├── models/
│   │   ├── inceptionV3_fid.py # InceptionV3 feature extractor
│   │   └── inceptionV4_fid.py # InceptionV4 feature extractor
│   ├── dataset/
│   │   └── fid_dataloader.py   # Data loading utilities
│   └── utils/
│       └── calc_metrics.py     # FID calculation implementation
├── requirements.txt            # Python dependencies
└── README.md                  # This file
```

## Notes

- Images are automatically resized to 299x299 pixels (InceptionV3/V4 input size)
- ImageNet normalization is applied to all images
- The implementation supports both CPU and GPU computation
- Lower FID scores indicate better image quality (closer to real images)
- InceptionV4 uses timm library for model loading and provides 1536-dimensional features
- InceptionV3 provides both avgpool (2048-dim) and Mixed_5d (288-dim) features
