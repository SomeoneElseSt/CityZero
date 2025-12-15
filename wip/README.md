# Local Tests - Lambda Cloud GPU Preprocessing

Testing 3D reconstruction pipeline on Financial District subset (2,998 images) before full SF dataset.

---

## What Works: COLMAP with CUDA

**Script:** `lambda_build_colmap_cuda.py`

### Learnings from first run:

**COLMAP Warnings Are Normal:**
- Mapper prints warnings like "Could not register" and "Discarding reconstruction".
- These indicate COLMAP trying different initial pairs - they are not failures.
- The script verifies that reconstruction succeeded by checking output files.
- Only 54 of 2998 images registered is normal for sparse street data.

1. Lambda Stack's default COLMAP has NO CUDA support - runs on CPU only.
2. COLMAP must be built from source with CUDA flags.
3. Flag names changed in COLMAP 3.14:
   - Old: `--SiftExtraction.use_gpu`
   - New: `--FeatureExtraction.use_gpu`

---

## The following libraries didn't work:

### CuSfM (NVIDIA)
**Issue:** Requires initial camera poses. Mapillary images are unordered with no pose data.

### GLOMAP
**Issue:** Still uses CPU-only COLMAP for feature extraction/matching (the bottleneck).

### InstantSfM
**Issue:** Released as a Gradio web UI, not a CLI tool.

---

## Lambda Workflow

This section outlines the workflow for running COLMAP with CUDA on Lambda Cloud, including steps for initial setup and reusing the pipeline for new datasets.

### Feature Matching Strategy

COLMAP's performance depends critically on the matching strategy. Choose based on data type:

**Sequential Matcher** (default)
- For: Video frames, drone footage, sequential captures
- Behavior: Matches each image to N nearest neighbors in sequence
- Complexity: O(N)
- Flag: `--matcher sequential`

**Exhaustive Matcher**
- For: Unordered images (Mapillary, street photos)
- Behavior: Compares every image to every other image
- Complexity: O(N²)
- Note: Impractical for large datasets (10K+ images)
- Flag: `--matcher exhaustive`

**Vocab Tree Matcher**
- For: Large (50K+) unordered collections
- Behavior: Visual vocabulary for efficient matching
- Requires: Pre-trained vocab tree file
- Flag: `--matcher vocab_tree`

Sequential is default. Exhaustive scales poorly with large image counts.

### 1. Launch Instance
- Image: Lambda Stack (Ubuntu 24.04)
- GPU: A100 (40GB) recommended for COLMAP

### 2. Upload Scripts to Lambda

**Quick upload command** (uploads all necessary scripts):
```bash
# From local_tests directory, upload all scripts in one command:
scp -i *.pem lambda_build_colmap_cuda.py lambda_train_gsplat.py scripts/youtube_splits_lambda.py scripts/cookies.txt ubuntu@YOUR_IP:~/
```

**Alternative: Upload individually**
```bash
# Upload COLMAP script
scp -i *.pem lambda_build_colmap_cuda.py ubuntu@YOUR_IP:~/

# Upload Gaussian Splatting script 
scp -i *.pem lambda_train_gsplat.py ubuntu@YOUR_IP:~/

# Upload YouTube frame extraction script
scp -i *.pem scripts/youtube_splits_lambda.py ubuntu@YOUR_IP:~/

# Upload YouTube cookies (required for bot detection bypass)
scp -i *.pem scripts/cookies.txt ubuntu@YOUR_IP:~/
```

**Note on cookies.txt:**
- Required for YouTube downloads on Lambda (bypasses bot detection)
- Export from browser using "Get cookies.txt LOCALLY" extension (Chrome/Firefox)
- Never commit to git (already in .gitignore)

### 3. SSH into Lambda Instance

```bash
ssh -i *.pem ubuntu@YOUR_IP
tmux new -s processing  # Use tmux to prevent disconnection
```

### 4. Extract Frames from YouTube Video (Optional)

If downloading video directly on Lambda (recommended to avoid large uploads):

```bash
# On Lambda instance:
python3 youtube_splits_lambda.py "YOUTUBE_URL" --fps 15 --cookies ~/cookies.txt

# Outputs to ~/youtube_train/images/
```

**Alternative: Upload pre-extracted frames**
If you already have frames extracted locally:
```bash
# On local machine (compress for faster upload):
tar -cJf images.tar.xz -C outputs/youtube_train images
scp -i *.pem images.tar.xz ubuntu@YOUR_IP:~/

# On Lambda:
tar -xJf images.tar.xz
```

### 5. Run COLMAP Pipeline

```bash
# First run (builds COLMAP with CUDA):
python3 lambda_build_colmap_cuda.py --images ~/youtube_train/images --output ~/colmap_output

# Subsequent runs (skip build):
python3 lambda_build_colmap_cuda.py --images ~/youtube_train/images --output ~/colmap_output --skip-build

# For unordered images (Mapillary), use exhaustive matcher:
# python3 lambda_build_colmap_cuda.py --images ~/images --output ~/colmap_output --matcher exhaustive
```

### 6. Manual Commands (if script breaks between stages)

If the automated script fails at any stage, you can resume manually using these commands:

```bash
# Feature extraction
colmap feature_extractor \
  --database_path ~/colmap_output/database.db \
  --image_path ~/images \
  --FeatureExtraction.use_gpu 1 \
  --FeatureExtraction.gpu_index 0 \
  --SiftExtraction.max_num_features 16384

# Feature matching
colmap exhaustive_matcher \
  --database_path ~/colmap_output/database.db \
  --FeatureMatching.use_gpu 1 \
  --FeatureMatching.gpu_index 0 \
  --FeatureMatching.max_num_matches 65536

# Sparse reconstruction
colmap mapper \
  --database_path ~/colmap_output/database.db \
  --image_path ~/images \
  --output_path ~/colmap_output/sparse
```

### 7. Download Output and Train Locally

```bash
# On Lambda instance: compress the COLMAP output
tar -czf colmap_output.tar.gz colmap_output/

# On your local machine: download the compressed output and extract
scp -i *.pem ubuntu@YOUR_IP:~/colmap_output.tar.gz .
tar -xzf colmap_output.tar.gz
mkdir -p outputs/processed_colmap_data
mv colmap_output outputs/processed_colmap_data/

# For new datasets, create a symlink so Brush can find the images (example for 'new_are-images')
# cd outputs/processed_colmap_data/colmap_output
# ln -s ../../../new_are-images/images images
# cd ../../..

# Update run_brush_training.py paths to point to the correct data (example for 'new_are-images')
# IMAGES_DIR = SCRIPT_DIR / "new_are-images" / "images"
# COLMAP_SPARSE_DIR = SCRIPT_DIR / "outputs" / "new_area_colmap" / "colmap_output" / "sparse" / "0"

# Run local training (e.g., ~4 hours on M4 Mac)
python3 run_brush_training.py

# View trained model
# Option 1: Start Brush viewer, then open file in UI
~/.brush/target/release/brush_app
# Then: File > Open > Navigate to outputs/gaussian_splatting/brush/export_<training_steps>.ply

# Option 2: Direct load (if supported)
~/.brush/target/release/brush_app outputs/gaussian_splatting/brush/export_<training_steps>.ply
```

---

## Gaussian Splatting Training with gsplat

**Script:** `lambda_train_gsplat.py`

### What This Does:
- Takes COLMAP output and trains 3D Gaussian Splatting models on Lambda GPU instances
- Uses `gsplat` library: 4x less GPU memory and faster than original 3DGS implementation
- Outputs `.ply` files compatible with Brush viewer and other splat viewers
- Saves checkpoints at regular intervals for quality comparison

### Lambda Workflow for Gaussian Splatting

#### Typical Workflow: Run on Same Lambda Instance as COLMAP

After running `lambda_build_colmap_cuda.py`, the COLMAP output is already on your Lambda instance. Simply run gsplat training directly:

```bash
# Already SSH'd into Lambda instance from COLMAP step
# If you closed your session, reconnect:
ssh -i *.pem ubuntu@YOUR_IP

# Upload gsplat training script (only needed once)
# On your local machine:
scp -i *.pem lambda_train_gsplat.py ubuntu@YOUR_IP:~/

# Back on Lambda instance:
# Run training on the COLMAP output that's already there
python3 lambda_train_gsplat.py \
  --colmap ~/colmap_output \
  --output ~/gsplat_output

# For subsequent training runs (skip dependency installation):
python3 lambda_train_gsplat.py \
  --colmap ~/colmap_output \
  --output ~/gsplat_output \
  --skip-install
```

#### Alternative: Upload Pre-existing COLMAP Output

If you have COLMAP output from elsewhere (e.g., processed locally or on a different machine):

```bash
# Ensure COLMAP output has this structure:
# colmap_output/
# ├── images/              # Original images
# │   └── *.jpg
# └── sparse/
#     └── 0/
#         ├── cameras.bin
#         ├── images.bin
#         └── points3D.bin

# Compress and upload
tar -czf colmap_output.tar.gz colmap_output/
scp -i *.pem colmap_output.tar.gz lambda_train_gsplat.py ubuntu@YOUR_IP:~/

# SSH into Lambda and extract
ssh -i *.pem ubuntu@YOUR_IP
tar -xzf colmap_output.tar.gz

# Run training
python3 lambda_train_gsplat.py \
  --colmap ~/colmap_output \
  --output ~/gsplat_output
```

#### Custom Training Parameters

```bash
python3 lambda_train_gsplat.py \
  --colmap ~/colmap_output \
  --output ~/gsplat_output \
  --iterations 30000 \      # Total training iterations (default: 30000)
  --save-interval 5000 \    # Save .ply every N iterations (default: 5000)
  --data-factor 4 \         # Image downsampling factor (default: 4)
  --skip-install
```

**Parameter Guidelines:**
- `--iterations`: 7000 (quick test), 30000 (standard), 50000+ (high quality)
- `--save-interval`: 5000 (default), 0 (save only final model)
- `--data-factor`: 1 (full res), 2 (half), 4 (quarter, default), 8 (eighth)

#### Download Results

```bash
# On Lambda instance: output is automatically compressed
# On your local machine:
scp -i *.pem ubuntu@YOUR_IP:~/gsplat_output.tar.gz .

# Extract and view
tar -xzf gsplat_output.tar.gz

# View with Brush on Mac
~/.brush/target/release/brush_app gsplat_output/ply/point_cloud_30000.ply
```

### Output Structure

```
gsplat_output/
├── ply/                           # Checkpoint .ply files
│   ├── point_cloud_5000.ply      # 5k iteration checkpoint
│   ├── point_cloud_10000.ply     # 10k iteration checkpoint
│   ├── point_cloud_15000.ply
│   ├── point_cloud_20000.ply
│   ├── point_cloud_25000.ply
│   └── point_cloud_30000.ply     # Final output
├── checkpoints/                   # Model checkpoints for resuming
└── training_summary.json          # Training metadata
```

### Troubleshooting

**CUDA out of memory:**
- Increase `--data-factor` to downsample images more (try 8)
- Reduce `--iterations` for testing

**No images found:**
- Ensure COLMAP directory contains `images/` folder with original images

**Training very slow:**
- Check GPU usage: `nvidia-smi`
- Increase `--data-factor` to reduce image resolution
- Consider using fewer images

---

## Helpful commands for monitoring GPU inside Lambda instance

```bash
nvidia-smi                # One-time check
watch -n 2 nvidia-smi     # Real-time monitoring
nvidia-smi dmon -s u      # Utilization only
```

---

## Directory Structure

```
local_tests/
├── financial_district_images/
│   ├── images/                    # 2,998 images (gitignored)
│   └── download_metadata.json
├── outputs/                       # Training outputs (gitignored)
├── lambda_build_colmap_cuda.py    # Lambda COLMAP preprocessing script
├── lambda_train_gsplat.py         # Lambda Gaussian Splatting training script
├── run_brush_training.py          # Mac training script
└── README.md                      # This file
```
