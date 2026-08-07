# OccPlanner

Project page for **OccPlanner: Goal-Aware Occupancy-Conditioned Diffusion Planner** and **L3ROcc: Local 3D Reconstruction with Occupancy**.

OccPlanner grounds a pixel goal in robot-centric metric space, predicts visibility-aware local 3D occupancy from RGB-D history, and generates continuous obstacle-aware trajectories with a diffusion planner. L3ROcc supplies dense training supervision by converting monocular navigation videos into temporally consistent occupancy, visibility, and trajectory annotations.

**Project page:** <https://hbl-0624.github.io/>

> Paper and code are coming soon.

## Highlights

- Closed-loop simulation evaluation on 6,000 episodes across 60 unseen indoor scenes.
- Short- and long-range navigation over home, commercial, cluttered-easy, and cluttered-hard environments.
- Open-loop sim-to-real comparison and real-world fine-tuning on Unitree G2 RGB-D sequences.
- Qualitative physical closed-loop navigation demonstrations on a Unitree G2.
- L3ROcc supervision-generation and OccPlanner prediction demos.

## Repository structure

```text
.
├── index.html                    # Page content
├── style.css                    # Layout and visual design
├── script.js                    # Video autoplay and page interactions
├── assets/
│   ├── demos/                   # L3ROcc and OccPlanner system demos
│   ├── l3rocc/                  # L3ROcc figures and posters
│   ├── occplanner/              # OccPlanner figures and results
│   ├── real/
│   │   ├── video/               # Original Unitree G2 recordings
│   │   ├── web/                 # Web-encoded real-world videos
│   │   ├── poster/              # Video posters
│   │   └── trail/results/       # Motion-trail images used by the page
│   └── sim/                     # Simulation demo video and poster
└── scripts/
    ├── make_sim_demo_grid.sh    # Build the synchronized simulation grid
    ├── make_motion_trails.py    # OpenCV motion-trail generator
    └── make_sam2_motion_trails.py
                                  # SAM2 robot-segmented trail generator
```

## Local preview

The site is fully static. From the repository root, start a local server:

```bash
python -m http.server 8000
```

Then open <http://localhost:8000/>. Using a local server is recommended because browser behavior for videos and relative paths can differ when opening `index.html` directly.

## Media generation

### Simulation demo grid

The grid script arranges the 12 simulation runs into three rows and freezes shorter runs on their last frame until the longest sequence finishes:

```bash
bash scripts/make_sim_demo_grid.sh
```

Raw simulation clips under `assets/sim/video/` are intentionally excluded from Git; the generated web video and poster are stored under `assets/sim/web/` and `assets/sim/poster/`.

### SAM2 motion trails

The current real-world trail style uses six robot poses per run. The first and last poses remain sharp and fully opaque, while intermediate poses are softly blurred with a minimum opacity of 50%:

```bash
conda run --no-capture-output -n sam2 \
  python scripts/make_sam2_motion_trails.py \
  assets/real/video/real_1.mp4 \
  --poses 6 \
  --style temporal \
  --min-opacity 0.50 \
  --output-dir assets/real/trail/results
```

Pass multiple video paths to generate several trail images in one run. The script currently expects the SAM2 repository and `sam2.1_hiera_base_plus.pt` checkpoint at the default paths declared in the script; both can be overridden with `--sam2-root` and `--checkpoint`.

## Updating the page

- Edit page copy, metrics, captions, and links in `index.html`.
- Edit typography, colors, spacing, and responsive layout in `style.css`.
- Replace media with files of the same name, then update the corresponding `?v=` cache key in `index.html`.
- The nine motion-trail cards currently load `assets/real/trail/results/real_1_trail.png` through `real_9_trail.png`.

When the paper and source code are released, replace the two “coming soon” elements in `index.html` with their public links.
