# Local Mac M4 Training Tests

This directory contains test datasets and training outputs for local M4 Mac experiments before full Lambda Labs deployment.

## Structure

```
local_tests/
├── financial_district/          # Test dataset (3,000 images)
│   ├── images/                  # Raw downloaded images
│   └── download_metadata.json   # Mapillary metadata
├── outputs/                     # Training outputs (gitignored)
│   ├── gaussian_splatting/      # Brush/nerfstudio outputs
│   └── triangle_splatting/      # Triangle splatting outputs
└── README.md                    # This file
```

## Dataset Info

**Financial District**
- **Images**: 2,998 images
- **Area**: Downtown SF (Market St to Embarcadero)
- **Bbox**: (-122.407, 37.789) to (-122.396, 37.797)
- **Total available**: 4,427 images (downloaded 3,000)
- **Downloaded**: 2025-01-XX
- **Resolution**: 2048px (Mapillary max)

## Training Plan

### 1. Gaussian Splatting (Brush)
**Tool**: [Brush](https://github.com/ArthurBrussee/brush) - Mac-native 3DGS
**Expected Time**: ~4 hours (7K steps on M4)
**Output**: `outputs/gaussian_splatting/`

**Steps**:
1. Install Brush on M4 Mac
2. Run COLMAP (or use Brush's built-in processing)
3. Train overnight
4. Evaluate quality

### 2. Triangle Splatting
**Tool**: [Triangle Splatting](https://github.com/trianglesplatting/triangle-splatting)
**Expected Time**: TBD (likely longer than GS)
**Output**: `outputs/triangle_splatting/`

**Requirements**:
- CUDA GPU (requires Lambda Labs or similar)
- Cannot run on Mac M4 (CUDA-only)

**Alternative**: Train on Lambda Labs alongside full SF dataset

## Usage

### Download Additional Neighborhoods
```bash
# Mission District
uv run python download_images.py \
  --bbox "-122.426,37.749,-122.409,37.768" \
  --output-dir local_tests/mission \
  --limit 3000

# Castro
uv run python download_images.py \
  --bbox "-122.438,37.756,-122.425,37.766" \
  --output-dir local_tests/castro \
  --limit 3000
```

### Training Commands
```bash
# Gaussian Splatting (Brush)
# See Brush documentation: https://github.com/ArthurBrussee/brush

# Triangle Splatting (requires CUDA/Lambda)
# python train.py -s local_tests/financial_district -m outputs/triangle_splatting --eval
```

## Next Steps After Local Testing

1. **Evaluate quality** of both methods on Financial District test
2. **Compare rendering performance** (FPS, visual quality)
3. **Decide approach** for full SF reconstruction:
   - If Gaussian Splatting sufficient → Use CityGaussian on Lambda
   - If Triangle Splatting preferred → Train on Lambda with CUDA
4. **Full city training** on Lambda Labs (~$60-100 depending on method)

## Notes

- Local tests use **downsampled dataset** (3K images vs 496K for full SF)
- Quality will be lower than full-city reconstruction
- Purpose: **Validate pipeline** before spending Lambda credits
- Triangle Splatting requires CUDA (not available on Mac M4)
