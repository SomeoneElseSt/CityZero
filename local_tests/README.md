# Local Testing & Lambda Cloud Guide

This directory contains test datasets, training scripts, and cloud processing tools for local M4 Mac experiments and Lambda Cloud preprocessing.

---

## Directory Structure

```
local_tests/
├── financial_district/          # Test dataset (3,000 images)
│   ├── images/                  # Raw downloaded images (gitignored)
│   └── download_metadata.json   # Mapillary metadata
├── outputs/                     # Training outputs (gitignored)
│   ├── gaussian_splatting/      # Brush outputs
│   └── triangle_splatting/      # Triangle splatting outputs
├── scripts/                     # Helper scripts
│   ├── upload_to_lambda.sh      # Upload data to Lambda
│   └── download_from_lambda.sh  # Download results from Lambda
├── train_gaussian_mac.py        # COLMAP preprocessing for Mac
├── run_brush_training.py        # Automated Brush training
├── lambda_glomap_preprocessing.py  # Lambda Cloud GLOMAP script
└── README.md                    # This file
```

---

## Dataset Info

**Financial District Test Set**
- **Images**: 2,998 images
- **Area**: Downtown SF (Market St to Embarcadero)
- **Bbox**: (-122.407, 37.789) to (-122.396, 37.797)
- **Resolution**: 2048px (Mapillary max)
- **Purpose**: Local pipeline validation before full SF (~500K images)

---

## Local Mac Training

### Prerequisites

- Mac with Apple Silicon (M4 recommended)
- Homebrew installed
- COLMAP: `brew install colmap`
- ImageMagick: `brew install imagemagick`
- Rust/Cargo (for Brush): `curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh`

### Step 1: COLMAP Preprocessing

```bash
# Run COLMAP preprocessing (takes ~2-3 hours on Mac)
python3 train_gaussian_mac.py
```

**What it does:**
- Extracts features from 2,998 images
- Matches features between images
- Runs sparse reconstruction (camera poses + 3D points)
- Saves to `financial_district/colmap/`

**Important:** This runs on CPU (no GPU acceleration on Mac for COLMAP)

### Step 2: Gaussian Splatting with Brush

```bash
# After COLMAP completes, run Brush training
python3 run_brush_training.py
```

**What it does:**
- Pre-flight validation (checks COLMAP output, Brush installation)
- Runs Brush CLI training (~4 hours for 7K steps)
- Saves trained model to `outputs/gaussian_splatting/brush_output/`

**View results:**
```bash
~/.brush/target/release/brush_app outputs/gaussian_splatting/brush_output/
```

### Notes on Mac Training

- **Time**: ~6-7 hours total (COLMAP 2-3h + Brush 4h)
- **Cost**: $0 (local compute)
- **Quality**: Good for validation, lower than full city
- **Limitations**: CPU-only COLMAP, no Triangle Splatting support

---

## Lambda Cloud Processing

Lambda Cloud provides GPU instances for faster preprocessing. However, **important limitations exist**.

### ⚠️ Critical Performance Information

#### COLMAP GPU Support Issue

**Lambda Stack's default COLMAP does NOT have CUDA/GPU support.**

Verify on any Lambda instance:
```bash
colmap -h | grep -i cuda
# Output: "(Commit Unknown on Unknown without CUDA)"
```

**Impact:**
- Feature extraction: **CPU-only** (~1 image/sec)
- Feature matching: **CPU-only** 
- For 3,000 images: ~2-3 hours on CPU (similar to Mac)

#### GLOMAP Solution (Recommended)

GLOMAP provides 10-100x faster reconstruction in the final step:

**Workflow:**
1. COLMAP feature extraction (CPU, ~50 min)
2. COLMAP feature matching (CPU, ~25 min)
3. **GLOMAP mapper** (10-100x faster than COLMAP, ~10 min)

**Result: ~1.5 hours total** (vs 2-3 hours with COLMAP alone)

### Lambda Cloud Workflow

#### Step 1: Launch Instance

1. Go to https://cloud.lambdalabs.com/instances
2. Select: **Lambda Stack** (Ubuntu 24.04) image
3. Launch: **A100 (40GB)** @ $1.10/hour
4. Copy SSH IP and download `.pem` key

#### Step 2: Upload & Process

```bash
cd local_tests

# Compress images first (if not already done)
cd financial_district
tar -czf images.tar.gz images/
cd ..

# Upload to Lambda
scp -i *.pem financial_district/images.tar.gz ubuntu@YOUR_IP:~/
scp -i *.pem lambda_glomap_preprocessing.py ubuntu@YOUR_IP:~/

# SSH into Lambda
ssh -i *.pem ubuntu@YOUR_IP

# Decompress images
tar -xzf images.tar.gz

# Run in tmux (prevents disconnection issues)
tmux new -s preprocessing
python3 lambda_glomap_preprocessing.py --images ~/images --output ~/glomap_output

# Detach: Ctrl+B, then D
# Reconnect: tmux attach -t preprocessing
```

#### Step 3: Download Results

```bash
# Download from Lambda (local machine)
scp -i *.pem ubuntu@YOUR_IP:~/glomap_output.tar.gz .

# Extract
tar -xzf glomap_output.tar.gz

# IMPORTANT: Terminate Lambda instance to stop billing!
```

### Cost & Time Estimates

#### Using GLOMAP (Recommended)

**For 3,000 images:**
- Instance: A100 @ $1.10/hour
- Time: ~1.5 hours
- **Cost: ~$1.65**

**For Full SF (500K images):**
- Estimate: ~12-18 hours
- **Cost: ~$13-20**
- *Requires building COLMAP with CUDA for efficiency*

#### Building COLMAP with CUDA (Advanced)

For large-scale datasets (50K+ images), compile COLMAP with CUDA:

```bash
# Install dependencies (~5 min)
sudo apt update
sudo apt install -y \
  libceres-dev libsqlite3-dev libgl1-mesa-dev \
  libcgal-dev libboost-all-dev libeigen3-dev

# Clone and build COLMAP (~30-45 min)
git clone https://github.com/colmap/colmap.git
cd colmap && mkdir build && cd build
cmake .. -GNinja -DCMAKE_CUDA_ARCHITECTURES=native
ninja && sudo ninja install
```

**Worth it for:** 50K+ images where GPU acceleration significantly outweighs build time.

### Output Structure

```
glomap_output/
├── database.db              # COLMAP database with features
├── sparse/                  # Sparse reconstruction
│   ├── cameras.bin         # Camera intrinsics
│   ├── images.bin          # Camera poses
│   └── points3D.bin        # Sparse 3D points
└── processing_summary.json  # Processing stats
```

**Compatible with:**
- `run_brush_training.py` (Mac)
- 3DGS, Nerfstudio, CityGaussian
- Triangle Splatting

---

## Triangle Splatting

**Requirements:** CUDA GPU (Lambda Cloud)

Triangle Splatting cannot run on Mac M4. Options:

1. Train on Lambda alongside full SF dataset
2. Use preprocessed COLMAP output from Lambda, train elsewhere with CUDA

---

## Troubleshooting

### Lambda: Qt/OpenGL Errors

```bash
export QT_QPA_PLATFORM=offscreen
# Then run your script
```

The `lambda_glomap_preprocessing.py` script handles this automatically.

### Lambda: Check GPU Usage

```bash
nvidia-smi              # One-time check
watch -n 2 nvidia-smi   # Real-time monitoring
```

**Expected:** 0% GPU with default Lambda COLMAP (CPU-only).

### Mac: COLMAP Not Recognizing Installation

Update the script to use `colmap help` instead of `colmap --version`:
```python
cmd = ["colmap", "help"]  # Works on all COLMAP versions
```

---

## Key Takeaways

### Local Mac Testing
✅ **Zero cost** - No cloud charges  
✅ **Pipeline validation** - Verify workflow before full city  
⚠️ **Slower** - CPU-only, 6-7 hours for 3K images  
⚠️ **No Triangle Splatting** - Requires CUDA/Lambda  

### Lambda Cloud
✅ **GLOMAP recommended** - 10-100x faster reconstruction  
⚠️ **COLMAP no GPU support** - Feature extraction on CPU  
💰 **Cost-effective for small datasets** - ~$1.65 for 3K images  
🔧 **Build CUDA COLMAP for 50K+ images** - Worth the setup time  
⏱️ **Always use tmux** - Prevents connection drops  
🛑 **Terminate immediately** - Billing continues until terminated  

### Full SF Reconstruction (500K images)
- **Recommended:** Build COLMAP with CUDA on Lambda
- **Estimated Cost:** $13-20 (A100, 12-18 hours)
- **Alternative:** Use GLOMAP with CPU COLMAP (~$25-30, slower)

---

## Next Steps

1. ✅ Complete local test (Financial District)
2. ✅ Evaluate Gaussian Splatting quality
3. 🔄 Test Lambda preprocessing with GLOMAP
4. 📊 Compare local vs Lambda preprocessing
5. 🚀 Full SF reconstruction (if validated)
