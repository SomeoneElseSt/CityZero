# CityZero - Virtual City Simulation Platform

Create virtual simulations of real cities for traffic and autonomous vehicle testing by reconstructing cities from street-level imagery.

## Quick Start

### 1. Prerequisites

- Python 3.14+
- [UV](https://github.com/astral-sh/uv) package manager

### 2. Installation

```bash
# Clone and navigate to the project
cd cityzero

# Install dependencies (UV handles virtual environment automatically)
uv sync
```

### 3. Configure Mapillary API

1. Get a free Mapillary API token:
   - Go to https://www.mapillary.com/dashboard/developers
   - Sign up/login
   - Create a new application
   - Copy your Client Token

2. Set up environment:
   ```bash
   # Copy the example environment file
   cp .env.example .env
   
   # Edit .env and add your token
   # MAPILLARY_CLIENT_TOKEN=your_token_here
   ```

### 4. Test the Setup

```bash
# Run the test script
uv run python test_mapillary.py
```

You should see coverage statistics for San Francisco!

## Project Structure

```
cityzero/
├── src/cityzero/          # Main source code
│   ├── config.py          # Configuration management
│   └── mapillary_client.py # Mapillary API client
├── data/
│   ├── raw/               # Raw downloaded images
│   ├── processed/         # Processed images
│   └── maps/              # HD maps and road networks
├── outputs/
│   ├── models/            # 3D models (Gaussian Splats, meshes)
│   └── renders/           # Rendered outputs
├── test_mapillary.py      # API test script
└── .env                   # Your API credentials (not in git)
```

## Current Status

✅ Project setup complete
✅ Mapillary API client ready
⏳ Image downloading pipeline (next)
⏳ 3D reconstruction pipeline
⏳ Semantic segmentation
⏳ HD map generation

## Next Steps

After successful API test, we'll implement:
1. Image downloading for San Francisco area
2. 3D reconstruction using Gaussian Splatting or COLMAP
3. Semantic segmentation (roads, lanes, signs)
4. HD map generation (Lanelet2 format)
5. Simulation integration (CARLA/SUMO)

## Tech Stack

- **Data Source**: Mapillary (open street-level imagery)
- **3D Reconstruction**: Gaussian Splatting / COLMAP + OpenMVS
- **HD Maps**: Lanelet2 + OpenStreetMap
- **3D Format**: GLTF 2.0 for simulation engines

## License

TBD
