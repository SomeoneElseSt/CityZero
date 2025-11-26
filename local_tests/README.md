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

### 1. Launch Instance
- Image: Lambda Stack (Ubuntu 24.04)
- GPU: A100 (40GB) @ $1.10/hour

### 2. Upload Code and Images

```bash
# On your local machine: compress images and upload the script and images to Lambda
# Note: tar compression (-z flag) is optional for JPEGs as they're already optimally compressed.
# Use tar without -z for JPEGs (just bundling), or skip tar and scp the directory directly.
tar -czf images.tar.gz financial_district/images/
scp -i *.pem images.tar.gz lambda_build_colmap_cuda.py ubuntu@YOUR_IP:~/
```

### 3. SSH into Lambda Instance and Run COLMAP

```bash
# On Lambda instance: SSH in, start a tmux session, and decompress images
ssh -i *.pem ubuntu@YOUR_IP
tmux new -s colmap
tar -xzf images.tar.gz

# Run the COLMAP pipeline. For the first run, it will build COLMAP with CUDA.
# For subsequent runs with new datasets, use --skip-build.
python3 lambda_build_colmap_cuda.py --images ~/images --output ~/colmap_output

# To process another dataset after COLMAP is built:
# python3 lambda_build_colmap_cuda.py --images ~/new_dataset_images --output ~/new_dataset_output --skip-build
```

### 4. Manual Commands (if script breaks between stages)

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

### 5. Download Output and Train Locally

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
