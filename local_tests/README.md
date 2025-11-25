# Local Tests - Lambda Cloud GPU Preprocessing

Testing 3D reconstruction pipeline on Financial District subset (2,998 images) before full SF dataset.

---

## What Works: COLMAP with CUDA ✅

**Script:** `lambda_build_colmap_cuda.py`

### Critical Findings

**COLMAP Warnings Are Normal:**
- Mapper prints scary warnings like "Could not register" and "Discarding reconstruction"
- These are COLMAP trying different initial pairs - **not failures**
- Script now verifies reconstruction succeeded by checking output files
- Only 54 of 2998 images registered? **Normal for sparse street data**

1. **Lambda Stack's default COLMAP has NO CUDA support** - runs on CPU only
2. Must build COLMAP from source with CUDA flags
3. **Flag names changed in COLMAP 3.14:**
   - ❌ Old: `--SiftExtraction.use_gpu`
   - ✅ New: `--FeatureExtraction.use_gpu`

### Performance (3K images, A100)

| Step | Time | GPU Usage |
|------|------|-----------|
| Build COLMAP | 30-45 min | - |
| Feature extraction | 1.76 min | 80-95% |
| Feature matching | 1.2 min | 80-95% |
| Mapper | 10-30 min | ~0% (CPU-bound) |
| **Total** | **45-80 min** | - |

**vs alternatives:**
- M4 Mac: ~10 hours
- CPU-only Lambda: 3-4 hours

---

## What Didn't Work

### CuSfM (NVIDIA) ❌
**Issue:** Requires initial camera poses. Mapillary images are unordered with no pose data.

### GLOMAP ❌  
**Issue:** Still uses CPU-only COLMAP for feature extraction/matching (the bottleneck).

### InstantSfM ❌
**Issue:** Released as Gradio web UI, not a CLI tool.

---

## Lambda Workflow

### 1. Launch Instance
- Image: Lambda Stack (Ubuntu 24.04)
- GPU: A100 (40GB) @ $1.10/hour

### 2. Upload & Run

```bash
# Local: compress and upload
tar -czf images.tar.gz financial_district/images/
scp -i *.pem images.tar.gz lambda_build_colmap_cuda.py ubuntu@YOUR_IP:~/

# Lambda: decompress and run
ssh -i *.pem ubuntu@YOUR_IP
tmux new -s colmap
tar -xzf images.tar.gz
python3 lambda_build_colmap_cuda.py --images ~/images --output ~/colmap_output
```

### 3. Manual Commands (if script breaks between stages)

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

### 4. Download & Train

```bash
# Lambda: compress output
tar -czf colmap_output.tar.gz colmap_output/

# Mac: download and train
scp -i *.pem ubuntu@YOUR_IP:~/colmap_output.tar.gz .
tar -xzf colmap_output.tar.gz
python3 run_brush_training.py  # ~4 hours on M4 Mac

# View trained model
# Option 1: Start Brush viewer, then open file in UI
~/.brush/target/release/brush_app
# Then: File > Open > Navigate to outputs/gaussian_splatting/brush/export_15000.ply

# Option 2: Direct load (if supported)
~/.brush/target/release/brush_app outputs/gaussian_splatting/brush/export_15000.ply
```

---

## Monitoring GPU

```bash
nvidia-smi                # One-time check
watch -n 2 nvidia-smi     # Real-time monitoring
nvidia-smi dmon -s u      # Utilization only
```

**Expected:** 80-95% during extraction/matching, ~0% during mapper.

---

## Cost Estimates

### Current Dataset (3K images)
- Build + processing: ~1.5 hours
- Cost: **~$1.65**
- Local training: ~4 hours (free)

### Full SF Dataset (500K images)
- Estimated time: 12-18 hours
- Estimated cost: **$13-20**
- GPU utilization: 80-95%

---

## Reusing Pipeline for New Datasets

### Quick Steps

1. **Download new images** (use main downloader from project root)
2. **Create new directory** in `local_tests/`:
   ```bash
   mkdir new_area_images
   mv /path/to/images new_area_images/images/
   ```

3. **Run Lambda preprocessing** (reuse built COLMAP with `--skip-build`):
   ```bash
   tar -czf images.tar.gz new_area_images/images/
   scp -i *.pem images.tar.gz ubuntu@IP:~/
   # On Lambda:
   python3 lambda_build_colmap_cuda.py --images ~/images --output ~/colmap_output --skip-build
   ```

4. **Download and extract**:
   ```bash
   scp -i *.pem ubuntu@IP:~/colmap_output.tar.gz .
   tar -xzf colmap_output.tar.gz
   mkdir -p outputs/new_area_colmap
   mv colmap_output outputs/new_area_colmap/
   ```

5. **Create symlink** (critical - Brush needs to find images):
   ```bash
   cd outputs/new_area_colmap/colmap_output
   ln -s ../../../new_area_images/images images
   cd ../../..
   ```

6. **Update `run_brush_training.py` paths**:
   ```python
   IMAGES_DIR = SCRIPT_DIR / "new_area_images" / "images"
   COLMAP_SPARSE_DIR = SCRIPT_DIR / "outputs" / "new_area_colmap" / "colmap_output" / "sparse" / "0"
   ```

7. **Train and view**:
   ```bash
   python3 run_brush_training.py
   ~/.brush/target/release/brush_app  # Then open the .ply file
   ```

---

## Directory Structure

```
local_tests/
├── financial_district/
│   ├── images/                    # 2,998 images (gitignored)
│   └── download_metadata.json
├── outputs/                       # Training outputs (gitignored)
├── lambda_build_colmap_cuda.py    # Lambda preprocessing script
├── run_brush_training.py          # Mac training script
└── README.md
```
