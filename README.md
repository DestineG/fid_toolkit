# FID (Fréchet Inception Distance) Calculator

A simple Python implementation for calculating FID scores between real and generated images using InceptionV3 features.

## Features

- **Standard FID**: Uses avgpool features from InceptionV3
- **Spatial FID**: Uses Mixed_5d features for spatial-aware comparison
- **Batch processing**: Efficient feature extraction with customizable batch sizes
- **GPU support**: CUDA acceleration for faster computation

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
from src.main import calc_standard_fid, calc_spatial_fid, calc_both_fid

# Calculate standard FID only
fid_score = calc_standard_fid(real_path, fake_path, batch_size=32, device='cuda')

# Calculate spatial FID only  
fid_score = calc_spatial_fid(real_path, fake_path, batch_size=32, device='cuda')

# Calculate both FID scores at once
fid_std, fid_spatial = calc_both_fid(real_path, fake_path, batch_size=32, device='cuda')
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
│   │   └── inceptionV3_fid.py # InceptionV3 feature extractor
│   ├── dataset/
│   │   └── fid_dataloader.py   # Data loading utilities
│   └── utils/
│       └── calc_metrics.py     # FID calculation implementation
├── requirements.txt            # Python dependencies
└── README.md                  # This file
```

## Notes

- Images are automatically resized to 299x299 pixels (InceptionV3 input size)
- ImageNet normalization is applied to all images
- The implementation supports both CPU and GPU computation
- Lower FID scores indicate better image quality (closer to real images)
