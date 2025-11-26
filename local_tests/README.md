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
├── lambda_build_colmap_cuda.py    # Lambda preprocessing script
├── run_brush_training.py          # Mac training script 
└── README.md # This file
```
