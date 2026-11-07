# FloodImpactNet

**FloodImpactNet** is an interactive, uncertainty-aware flood-analysis prototype for eight-channel Sen1Floods11 GeoTIFF scenes. It transforms one uploaded satellite scene into three linked decision-support products:

1. **Flood extent** — where the model detects floodwater.
2. **Synthetic temporal scenario** — how that condition could contract or expand under controlled counterfactual edits.
3. **Review priority** — which affected pixels need the most human verification.

> This is a research decision-support system. Synthetic scenarios are not observed satellite acquisitions, hydrological forecasts, verified damage maps, or calibrated probabilities of error.

## Highlights

- Eight-channel U-Net flood segmentation with geospatial GeoTIFF input.
- Image-conditioned residual GAN for local flood-boundary scenarios.
- U-Net, DeepLabV3+, entropy, and six-transform TTA uncertainty ensemble.
- Categorical map: not affected, low, medium, and high review priority.
- React/Vite frontend and Flask/PyTorch backend.
- Interactive narrative pages, before/after comparisons, and animated model diagrams.

## Workflow

```text
8-channel GeoTIFF
      ↓
Module 1 — U-Net flood segmentation
      ↓
Flood probability + baseline flood mask
      ↓
Module 2 — image-conditioned residual GAN
      ↓
Synthetic pre/post images + temporal change
      ↓
Module 3 — uncertainty ensemble
      ↓
Continuous uncertainty + priority map
```

## Modules

### Module 1 — Flood extent segmentation

Module 1 accepts an `8 × 512 × 512` image in Sen1Floods11 channel order and predicts background/flood logits. A softmax threshold of `0.50` produces the binary flood mask.

**Primary model: U-Net**

```text
Encoder:    8 → 32 → 64 → 128 → 256
Bottleneck: 256 → 512
Decoder:    512 → 256 → 128 → 64 → 32
Output:     2 classes (background, flood)
```

Each block uses two `3×3 convolution → batch normalization → ReLU` operations. Skip connections retain high-resolution spatial information.

**Training loss**

```text
Weighted cross-entropy + Dice loss
Background weight: 1.0
Flood weight:      3.0
Ignored label:     -1
```

**Comparison model:** DeepLabV3+ with an eight-channel ResNet-50 backbone, ASPP branches at rates 6/12/18, feature fusion, and decoder refinement.

| Held-out test metric | U-Net | DeepLabV3+ |
|---|---:|---:|
| Flood IoU | **0.7660** | 0.7356 |
| Mean IoU | **0.8694** | 0.8519 |
| Dice | **0.8675** | 0.8477 |
| Pixel accuracy | **97.50%** | 97.08% |

### Module 2 — Counterfactual temporal scenario

Module 2 starts from Module 1’s predicted baseline flood mask.

```text
Baseline mask
   ├─ erosion  → synthetic pre-flood target
   └─ dilation → synthetic post-flood target
```

The generator receives an eight-channel source image, source flood mask, target flood mask, and valid-data mask: **11 input channels** in total. It predicts an eight-channel residual:

```text
Synthetic image = source image + local residual
```

The residual is multiplied by the editable XOR region between source and target masks. Pixels outside the requested boundary change are preserved.

- **Generator:** image-conditioned encoder-decoder with GroupNorm, LeakyReLU, skip connections, and bounded residual output.
- **Discriminator:** spectral-normalized PatchGAN for local realism.

**Generator objective**

```text
1.00 × adversarial loss
0.50 × feature-matching loss
1.50 × frozen U-Net segmentation-consistency loss
0.02 × residual-magnitude loss
0.05 × residual-smoothness loss
```

The frozen U-Net verifies that the generated image expresses the requested flood condition. The UI’s primary comparison is **Module 1 baseline → generated post-flood scenario**, followed by a secondary synthetic pre-flood → post-flood comparison.

### Module 3 — Uncertainty and review priority

Module 3 is an inference-time ensemble; it does not train a third neural network. For every post-flood pixel, it combines:

- Predictive entropy from U-Net probability.
- TTA variation from six U-Net predictions: identity, horizontal/vertical flips, and 90°/180°/270° rotations.
- U-Net/DeepLabV3+ disagreement.

```math
U(x,y) = 0.5H(x,y) + 0.3σ_TTA(x,y) + 0.2|P_U-Net(x,y) - P_DeepLab(x,y)|
```

The result is a **relative, uncalibrated review-priority score**:

- Grey — not affected
- Green — low priority
- Yellow — medium priority
- Red — high priority

Thresholds are derived from the current scene: median for medium priority and 85th percentile for high priority.

## Dataset and preprocessing

The project uses the **Sen1Floods11 8-channel dataset**.

| Split | Scenes |
|---|---:|
| Train | 312 |
| Validation | 67 |
| Held-out test | 67 |
| Total | 446 |

Input preprocessing:

1. Replace non-finite values with the corresponding training-channel mean.
2. Apply channel-wise normalization: `(x - mean) / std`.
3. Preserve ignored pixels (`-1`) during Module 1 training and evaluation.

## Repository structure

```text
FloodImpactNet_App/
├── backend/
│   ├── app.py                 # Flask + live PyTorch inference
│   └── requirements.txt
├── frontend/
│   ├── src/
│   │   ├── App.jsx            # Guided product experience
│   │   └── styles.css
│   ├── public/assets/         # Lightweight demo imagery
│   ├── package.json
│   └── vite.config.js
├── scripts/
│   └── build_demo_assets.py
├── .gitignore
└── README.md
```

## Local setup

### 1. Place model asset packages beside this repository

Large checkpoints and GeoTIFF products are intentionally not committed to GitHub. Keep the application beside the unzipped notebook packages:

```text
D:\FloodImpactNet\
├── FloodImpactNet_App\
├── FloodImpactNet_Module1_UI_Assets\
├── FloodImpactNet_Module2_UI_Assets (1)\
└── FloodImpactNet_Module3_UI_Assets\
```

The backend discovers these folders automatically. Alternatively, set `FLOODIMPACTNET_ROOT` to their parent directory.

### 2. Start Flask

```powershell
cd D:\FloodImpactNet\FloodImpactNet_App\backend
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python app.py
```

### 3. Start React

Open a second PowerShell window:

```powershell
cd D:\FloodImpactNet\FloodImpactNet_App\frontend
npm install
npm run dev
```

Open the local URL shown by Vite, normally `http://localhost:5173`.

## Live inference

Selecting an image requests an immediate visual preview. Clicking **Run all models** executes fresh on-demand inference:

```text
U-Net baseline segmentation
→ residual GAN pre/post generation
→ U-Net verification of synthetic images
→ six-transform U-Net TTA
→ DeepLabV3+ comparison
→ uncertainty and priority products
```

The app is on-demand real-time inference, not continuous streaming. A CPU request takes several seconds.

## Input requirements

- GeoTIFF (`.tif` or `.tiff`)
- Exactly 8 channels
- Exactly `512 × 512` pixels
- Sen1Floods11 channel order

## Limitations and responsible use

- Module 2 creates counterfactual scenarios, not real future acquisitions.
- Erosion and dilation are geometric operations, not physical flood-flow simulation.
- Flood extent is not structural or monetary damage.
- Module 3 uncertainty is relative and uncalibrated.
- Human verification is required before operational decisions.

## Research foundations

- [U-Net: Convolutional Networks for Biomedical Image Segmentation](https://arxiv.org/abs/1505.04597)
- [DeepLabV3+: Encoder-Decoder with Atrous Separable Convolution](https://arxiv.org/abs/1802.02611)
- [Image-to-Image Translation with Conditional Adversarial Networks](https://arxiv.org/abs/1611.07004)

## Future work

- Integrate terrain, drainage, rainfall, and hydrological constraints.
- Add buildings, roads, settlement, and population exposure layers.
- Add depth-damage curves for defensible damage estimation.
- Calibrate uncertainty with independent real temporal observations.
- Package checkpoint assets through a controlled release/download process.
