# Versioned model assets

This folder contains the checkpoints required for fully local FloodImpactNet inference.
They are stored with **Git LFS**, so a normal clone must be followed by `git lfs pull`
when LFS objects were not downloaded automatically.

Required files:

```text
module1/config/preprocessing.json
module1/models/best_unet.pth
module1/models/best_deeplabv3plus.pth
module2/config/module2_v2_config.json
module2/models/best_residual_gan.pth
```

The backend prefers this directory. Its legacy fallback still supports the separately
unzipped Kaggle UI packages when this directory is absent.
