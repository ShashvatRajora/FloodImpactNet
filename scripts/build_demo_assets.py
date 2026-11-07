import json
from pathlib import Path

import numpy as np
import rasterio
from PIL import Image

SOURCE_ROOT = Path(r"D:\FloodImpactNet")
OUTPUT = Path(__file__).resolve().parents[1] / "frontend" / "public" / "assets"
OUTPUT.mkdir(parents=True, exist_ok=True)


def nested(name):
    matches = list(SOURCE_ROOT.glob(f"{name}*/{name}"))
    if not matches:
        raise FileNotFoundError(f"Could not locate unzipped {name} under {SOURCE_ROOT}")
    return matches[0]


MODULE1 = nested("FloodImpactNet_Module1_UI_Assets")
MODULE2 = nested("FloodImpactNet_Module2_UI_Assets")
MODULE3 = nested("FloodImpactNet_Module3_UI_Assets")


def read_tiff(path):
    with Image.open(path) as image:
        return np.asarray(image, dtype=np.float32)


def save_gradient(source, destination, stops):
    values = read_tiff(source)
    finite = np.isfinite(values)
    lo, hi = np.nanpercentile(values[finite], [2, 98]) if finite.any() else (0, 1)
    scaled = np.clip((values - lo) / max(float(hi - lo), 1e-6), 0, 1)
    positions = np.linspace(0, 1, len(stops))
    rgb = np.empty(values.shape + (3,), dtype=np.uint8)
    for channel in range(3):
        rgb[..., channel] = np.interp(scaled, positions, [c[channel] for c in stops]).astype(np.uint8)
    rgb[~finite] = 238
    Image.fromarray(rgb).save(destination, optimize=True)


def save_classes(source, destination, colors):
    values = read_tiff(source).astype(np.uint8)
    rgb = np.zeros(values.shape + (3,), dtype=np.uint8)
    for value, color in colors.items():
        rgb[values == value] = color
    Image.fromarray(rgb).save(destination, optimize=True)


def composite_array(source):
    with rasterio.open(source) as dataset:
        image = dataset.read([3, 2, 1]).astype(np.float32)
    rgb = np.moveaxis(image, 0, -1)
    output = np.zeros_like(rgb, dtype=np.uint8)
    for channel in range(3):
        values = rgb[..., channel]
        finite = np.isfinite(values)
        lo, hi = np.percentile(values[finite], [2, 98])
        output[..., channel] = np.clip((np.nan_to_num(values, nan=lo) - lo) / max(float(hi - lo), 1e-6) * 255, 0, 255)
    return output


def save_composite(source, destination):
    Image.fromarray(composite_array(source)).save(destination, optimize=True)


SCENARIO = MODULE2 / "scenarios" / "294583_rising_flood_v2"
SOURCE_IMAGE = Path(r"D:\Downloads\archive\dataset\Sen1Floods11_8Channel\image\294583_image.tif")
save_composite(SOURCE_IMAGE, OUTPUT / "module1-source.png")
source_rgb = composite_array(SOURCE_IMAGE)
source_mask = read_tiff(MODULE1 / "predictions/flood_masks/294583_flood_mask.tif") == 1
overlay = source_rgb.copy()
overlay[source_mask] = (.25 * overlay[source_mask] + .75 * np.array([47, 190, 210])).astype(np.uint8)
Image.fromarray(overlay).save(OUTPUT / "module1-overlay.png", optimize=True)
save_composite(SCENARIO / "synthetic_images/294583_synthetic_pre_flood.tif", OUTPUT / "module2-pre-image.png")
save_composite(SCENARIO / "synthetic_images/294583_synthetic_post_flood.tif", OUTPUT / "module2-post-image.png")
save_gradient(MODULE1 / "predictions/flood_probabilities/294583_flood_probability.tif", OUTPUT / "module1-probability.png", [(21,34,28),(43,106,103),(91,193,184),(245,220,80)])
save_classes(MODULE1 / "predictions/flood_masks/294583_flood_mask.tif", OUTPUT / "module1-mask.png", {0:(226,234,228),1:(34,124,151)})
for state in ("pre", "post"):
    save_gradient(SCENARIO / f"flood_probabilities/294583_synthetic_{state}_probability.tif", OUTPUT / f"module2-{state}-probability.png", [(19,37,30),(40,105,111),(91,190,181),(246,212,70)])
save_classes(SCENARIO / "change_maps/294583_synthetic_change_map.tif", OUTPUT / "module2-change.png", {0:(239,242,239),1:(35,105,142),2:(214,88,63),3:(54,151,104),255:(35,35,35)})
for state in ("pre", "post"):
    save_gradient(MODULE3 / f"uncertainty_maps/294583_synthetic_{state}_composite_uncertainty.tif", OUTPUT / f"module3-{state}-uncertainty.png", [(37,91,70),(93,174,126),(230,185,64),(202,78,66)])
    save_classes(MODULE3 / f"uncertainty_maps/294583_synthetic_{state}_uncertainty_class.tif", OUTPUT / f"module3-{state}-classes.png", {0:(37,37,37),1:(69,150,112),2:(225,181,60),3:(203,80,67)})

demo = {
    "sample_id": "294583",
    "module1": {"threshold": 0.5, "probability_preview": "assets/module1-probability.png", "mask_preview": "assets/module1-mask.png"},
    "module2": {"pre_probability": "assets/module2-pre-probability.png", "post_probability": "assets/module2-post-probability.png", "change_map": "assets/module2-change.png"},
    "module3": {"pre_uncertainty": "assets/module3-pre-uncertainty.png", "post_uncertainty": "assets/module3-post-uncertainty.png", "post_classes": "assets/module3-post-classes.png"},
}
(OUTPUT / "demo-data.json").write_text(json.dumps(demo, indent=2), encoding="utf-8")
print(f"Built {len(list(OUTPUT.iterdir()))} browser assets in {OUTPUT}")
