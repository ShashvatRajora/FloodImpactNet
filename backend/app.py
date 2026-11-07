import base64
import io
import json
import math
import os
import tempfile
from pathlib import Path

import numpy as np
import rasterio
import torch
import torch.nn as nn
import torch.nn.functional as F
from flask import Flask, jsonify, request
from flask_cors import CORS
from PIL import Image

APP_ROOT = Path(__file__).resolve().parent.parent
SOURCE_ROOT = Path(os.getenv("FLOODIMPACTNET_ROOT", APP_ROOT.parent))
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def locate_bundle(name):
    direct = SOURCE_ROOT / name / name
    if direct.exists():
        return direct
    matches = list(SOURCE_ROOT.glob(f"{name}*/{name}"))
    return matches[0] if matches else SOURCE_ROOT / name


MODULE1_ROOT = locate_bundle("FloodImpactNet_Module1_UI_Assets")
MODULE2_ROOT = locate_bundle("FloodImpactNet_Module2_UI_Assets")
app = Flask(__name__)
CORS(app)


def extract_state_dict(checkpoint, preferred=None):
    if preferred and isinstance(checkpoint, dict) and preferred in checkpoint:
        checkpoint = checkpoint[preferred]
    elif isinstance(checkpoint, dict):
        checkpoint = next((checkpoint[k] for k in ("model_state_dict", "state_dict", "model") if isinstance(checkpoint.get(k), dict)), checkpoint)
    if all(key.startswith("module.") for key in checkpoint):
        checkpoint = {key[7:]: value for key, value in checkpoint.items()}
    return checkpoint


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False), nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False), nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class UNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.pool = nn.MaxPool2d(2)
        self.enc1, self.enc2, self.enc3, self.enc4 = DoubleConv(8, 32), DoubleConv(32, 64), DoubleConv(64, 128), DoubleConv(128, 256)
        self.bottleneck = DoubleConv(256, 512)
        self.up4, self.dec4 = nn.ConvTranspose2d(512, 256, 2, 2), DoubleConv(512, 256)
        self.up3, self.dec3 = nn.ConvTranspose2d(256, 128, 2, 2), DoubleConv(256, 128)
        self.up2, self.dec2 = nn.ConvTranspose2d(128, 64, 2, 2), DoubleConv(128, 64)
        self.up1, self.dec1 = nn.ConvTranspose2d(64, 32, 2, 2), DoubleConv(64, 32)
        self.output = nn.Conv2d(32, 2, 1)

    def forward(self, x):
        e1 = self.enc1(x); e2 = self.enc2(self.pool(e1)); e3 = self.enc3(self.pool(e2)); e4 = self.enc4(self.pool(e3))
        b = self.bottleneck(self.pool(e4))
        d4 = self.dec4(torch.cat([self.up4(b), e4], 1)); d3 = self.dec3(torch.cat([self.up3(d4), e3], 1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], 1)); d1 = self.dec1(torch.cat([self.up1(d2), e1], 1))
        return self.output(d1)


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super().__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, 3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        if self.downsample is not None:
            identity = self.downsample(x)
        return self.relu(out + identity)


class ResNet50Backbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.inplanes = 64
        self.conv1 = nn.Conv2d(8, 64, 7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(3, stride=2, padding=1)
        self.layer1 = self._make_layer(64, 3)
        self.layer2 = self._make_layer(128, 4, stride=2)
        self.layer3 = self._make_layer(256, 6, stride=2)
        self.layer4 = self._make_layer(512, 3, stride=1)

    def _make_layer(self, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * Bottleneck.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * Bottleneck.expansion, 1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * Bottleneck.expansion),
            )
        layers = [Bottleneck(self.inplanes, planes, stride, downsample)]
        self.inplanes = planes * Bottleneck.expansion
        layers.extend(Bottleneck(self.inplanes, planes) for _ in range(1, blocks))
        return nn.Sequential(*layers)


class ASPPConv(nn.Sequential):
    def __init__(self, in_channels, out_channels, dilation):
        super().__init__(nn.Conv2d(in_channels, out_channels, 3, padding=dilation, dilation=dilation, bias=False), nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True))


class ASPPPooling(nn.Sequential):
    def __init__(self, in_channels, out_channels):
        super().__init__(nn.AdaptiveAvgPool2d(1), nn.Conv2d(in_channels, out_channels, 1, bias=False), nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True))

    def forward(self, x):
        size = x.shape[-2:]
        return F.interpolate(super().forward(x), size=size, mode="bilinear", align_corners=False)


class ASPP(nn.Module):
    def __init__(self, in_channels, out_channels=256):
        super().__init__()
        self.branches = nn.ModuleList([
            nn.Sequential(nn.Conv2d(in_channels, out_channels, 1, bias=False), nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True)),
            ASPPConv(in_channels, out_channels, 6), ASPPConv(in_channels, out_channels, 12), ASPPConv(in_channels, out_channels, 18), ASPPPooling(in_channels, out_channels),
        ])
        self.project = nn.Sequential(nn.Conv2d(out_channels * 5, out_channels, 1, bias=False), nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True), nn.Dropout(.5))

    def forward(self, x):
        return self.project(torch.cat([branch(x) for branch in self.branches], 1))


class DeepLabV3Plus(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = ResNet50Backbone()
        self.aspp = ASPP(2048, 256)
        self.low_level_projection = nn.Sequential(nn.Conv2d(256, 48, 1, bias=False), nn.BatchNorm2d(48), nn.ReLU(inplace=True))
        self.decoder = nn.Sequential(
            nn.Conv2d(304, 256, 3, padding=1, bias=False), nn.BatchNorm2d(256), nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1, bias=False), nn.BatchNorm2d(256), nn.ReLU(inplace=True),
        )
        self.classifier = nn.Conv2d(256, 2, 1)

    def forward(self, x):
        input_size = x.shape[-2:]
        x = self.backbone.maxpool(self.backbone.relu(self.backbone.bn1(self.backbone.conv1(x))))
        low = self.backbone.layer1(x)
        x = self.backbone.layer4(self.backbone.layer3(self.backbone.layer2(low)))
        x = F.interpolate(self.aspp(x), size=low.shape[-2:], mode="bilinear", align_corners=False)
        x = self.decoder(torch.cat([x, self.low_level_projection(low)], 1))
        return F.interpolate(self.classifier(x), size=input_size, mode="bilinear", align_corners=False)


def group_norm(channels, maximum_groups=32):
    groups = min(maximum_groups, channels)
    while channels % groups:
        groups -= 1
    return nn.GroupNorm(groups, channels)


class ResidualConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1), group_norm(out_channels), nn.LeakyReLU(.2, inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1), group_norm(out_channels), nn.LeakyReLU(.2, inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class ImageConditionedResidualGenerator(nn.Module):
    def __init__(self, lower_bounds, upper_bounds):
        super().__init__()
        self.pool = nn.MaxPool2d(2)
        self.enc1, self.enc2, self.enc3, self.enc4 = ResidualConvBlock(11, 32), ResidualConvBlock(32, 64), ResidualConvBlock(64, 128), ResidualConvBlock(128, 256)
        self.bottleneck = ResidualConvBlock(256, 256)
        self.dec4, self.dec3, self.dec2, self.dec1 = ResidualConvBlock(512, 128), ResidualConvBlock(256, 64), ResidualConvBlock(128, 32), ResidualConvBlock(64, 32)
        self.residual_output = nn.Conv2d(32, 8, 3, padding=1)
        lower = torch.tensor(lower_bounds).view(1, 8, 1, 1)
        upper = torch.tensor(upper_bounds).view(1, 8, 1, 1)
        self.register_buffer("lower_bounds", lower); self.register_buffer("upper_bounds", upper)
        self.register_buffer("residual_scale", .5 * (upper - lower))

    def forward(self, source, source_mask, target_mask, valid):
        condition = torch.cat([source_mask, target_mask, valid], 1)
        e1 = self.enc1(torch.cat([source, condition], 1)); e2 = self.enc2(self.pool(e1)); e3 = self.enc3(self.pool(e2)); e4 = self.enc4(self.pool(e3)); b = self.bottleneck(self.pool(e4))
        d4 = self.dec4(torch.cat([F.interpolate(b, size=e4.shape[-2:], mode="bilinear", align_corners=False), e4], 1))
        d3 = self.dec3(torch.cat([F.interpolate(d4, size=e3.shape[-2:], mode="bilinear", align_corners=False), e3], 1))
        d2 = self.dec2(torch.cat([F.interpolate(d3, size=e2.shape[-2:], mode="bilinear", align_corners=False), e2], 1))
        d1 = self.dec1(torch.cat([F.interpolate(d2, size=e1.shape[-2:], mode="bilinear", align_corners=False), e1], 1))
        editable = (source_mask != target_mask).float() * valid
        residual = torch.tanh(self.residual_output(d1)) * self.residual_scale * editable
        source = torch.maximum(torch.minimum(source, self.upper_bounds), self.lower_bounds)
        synthetic = torch.maximum(torch.minimum(source + residual, self.upper_bounds), self.lower_bounds)
        return synthetic, residual, editable


unet_model = None
generator_model = None
deeplab_model = None
preprocessing = None


def load_models():
    global unet_model, generator_model, deeplab_model, preprocessing
    if unet_model is not None and generator_model is not None and deeplab_model is not None:
        return unet_model, generator_model, deeplab_model, preprocessing
    preprocessing = json.loads((MODULE1_ROOT / "config/preprocessing.json").read_text())
    unet_checkpoint = torch.load(MODULE1_ROOT / "models/best_unet.pth", map_location="cpu", weights_only=False)
    unet_model = UNet().to(DEVICE)
    unet_model.load_state_dict(extract_state_dict(unet_checkpoint), strict=True)
    unet_model.eval()
    gan_checkpoint = torch.load(MODULE2_ROOT / "models/best_residual_gan.pth", map_location="cpu", weights_only=False)
    gan_config = gan_checkpoint["config"]
    generator_model = ImageConditionedResidualGenerator(gan_config["generator_lower_bounds"], gan_config["generator_upper_bounds"]).to(DEVICE)
    generator_model.load_state_dict(extract_state_dict(gan_checkpoint, "generator_state_dict"), strict=True)
    generator_model.eval()
    deeplab_checkpoint = torch.load(MODULE1_ROOT / "models/best_deeplabv3plus.pth", map_location="cpu", weights_only=False)
    deeplab_model = DeepLabV3Plus().to(DEVICE)
    deeplab_model.load_state_dict(extract_state_dict(deeplab_checkpoint), strict=True)
    deeplab_model.eval()
    return unet_model, generator_model, deeplab_model, preprocessing


def read_upload(uploaded):
    if not uploaded or not uploaded.filename.lower().endswith((".tif", ".tiff")):
        raise ValueError("Choose an 8-channel GeoTIFF.")
    with tempfile.TemporaryDirectory(prefix="floodimpactnet_") as temp_dir:
        upload_path = Path(temp_dir) / "uploaded_scene.tif"
        uploaded.save(str(upload_path))
        with rasterio.open(upload_path) as src:
            image = src.read()
            transform, crs = src.transform, src.crs
    if image.shape != (8, 512, 512):
        raise ValueError(f"Expected (8, 512, 512); received {image.shape}.")
    return image, transform, crs


def data_url(array, mode="L"):
    buffer = io.BytesIO(); Image.fromarray(array, mode=mode).save(buffer, "PNG")
    return "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode()


def rgb_array(image):
    rgb = np.stack([image[2], image[1], image[0]], -1).astype(np.float32)
    output = np.zeros_like(rgb, dtype=np.uint8)
    for channel in range(3):
        values = rgb[..., channel]; finite = np.isfinite(values)
        if finite.any():
            lo, hi = np.percentile(values[finite], [2, 98])
            output[..., channel] = np.clip((np.nan_to_num(values, nan=lo) - lo) / max(hi - lo, 1e-6) * 255, 0, 255)
    return output


def rgb_preview(image):
    return data_url(rgb_array(image), "RGB")


def overlay_preview(image, mask, color=(47, 190, 210)):
    output = rgb_array(image)
    selected = mask.astype(bool)
    output[selected] = (.25 * output[selected] + .75 * np.asarray(color)).astype(np.uint8)
    return data_url(output, "RGB")


def color_preview(values, colors):
    values = np.clip(np.nan_to_num(values), 0, 1)
    positions = np.linspace(0, 1, len(colors))
    rgb = np.empty((*values.shape, 3), dtype=np.uint8)
    for channel in range(3):
        rgb[..., channel] = np.interp(values, positions, [color[channel] for color in colors]).astype(np.uint8)
    return data_url(rgb, "RGB")


def probability_preview(values):
    return color_preview(values, [(30, 17, 64), (48, 103, 141), (56, 183, 137), (241, 219, 64)])


def uncertainty_preview(values):
    return color_preview(values, [(32, 22, 72), (48, 121, 145), (64, 186, 139), (238, 214, 68), (220, 62, 54)])


def priority_overlay(raw, affected, uncertainty, moderate, high):
    base = rgb_array(raw)
    gray = np.mean(base, axis=2, keepdims=True).astype(np.uint8)
    output = np.repeat(gray, 3, axis=2)
    classes = np.zeros(affected.shape, dtype=np.uint8)
    classes[affected & (uncertainty < moderate)] = 1
    classes[affected & (uncertainty >= moderate) & (uncertainty < high)] = 2
    classes[affected & (uncertainty >= high)] = 3
    palette = {1: np.array([57, 181, 115]), 2: np.array([246, 190, 63]), 3: np.array([226, 67, 52])}
    for class_id, color in palette.items():
        selected = classes == class_id
        output[selected] = (.2 * output[selected] + .8 * color).astype(np.uint8)
    return data_url(output, "RGB"), classes


def normalize_image(raw, config):
    mean = np.asarray(config["channel_mean"], np.float32)[:, None, None]
    std = np.asarray(config["channel_std"], np.float32)[:, None, None]
    normalized = (np.where(np.isfinite(raw), raw, mean).astype(np.float32) - mean) / std
    return torch.from_numpy(normalized).unsqueeze(0).to(DEVICE)


def predict(network, image):
    return torch.softmax(network(image), 1)[:, 1:2]


def morph(mask, pixels, mode):
    pixels = max(0, int(pixels)); kernel = pixels * 2 + 1
    if pixels == 0:
        return mask.clone()
    if mode == "dilate":
        padded = F.pad(mask, (pixels,) * 4, value=0)
        return F.max_pool2d(padded, kernel, stride=1)
    padded_inverse = F.pad(1 - mask, (pixels,) * 4, value=1)
    return 1 - F.max_pool2d(padded_inverse, kernel, stride=1)


def tta_statistics(network, image):
    variants = [
        (image, lambda x: x),
        (torch.flip(image, [3]), lambda x: torch.flip(x, [3])),
        (torch.flip(image, [2]), lambda x: torch.flip(x, [2])),
        (torch.rot90(image, 1, [2, 3]), lambda x: torch.rot90(x, 3, [2, 3])),
        (torch.rot90(image, 2, [2, 3]), lambda x: torch.rot90(x, 2, [2, 3])),
        (torch.rot90(image, 3, [2, 3]), lambda x: torch.rot90(x, 1, [2, 3])),
    ]
    predictions = torch.stack([undo(predict(network, transformed)) for transformed, undo in variants])
    return predictions.mean(0), predictions.std(0, unbiased=False)


def pixel_area_m2(transform, crs):
    determinant = abs(transform.a * transform.e - transform.b * transform.d)
    if crs and crs.is_projected:
        try:
            factor = crs.linear_units_factor[1]
            return determinant * factor * factor
        except Exception:
            return determinant
    if crs and crs.is_geographic:
        try:
            # Spherical cell area at the scene centre; avoids adding a heavy
            # geodesy dependency for north-up geographic Sen1Floods11 tiles.
            lon1, lat1 = rasterio.transform.xy(transform, 256, 256, offset="ul")
            lon2, lat2 = rasterio.transform.xy(transform, 256, 256, offset="lr")
            radius = 6_371_008.8
            return abs(radius * radius * math.radians(lon2 - lon1) * (math.sin(math.radians(lat2)) - math.sin(math.radians(lat1))))
        except Exception:
            pass
    return 100.0


@app.get("/health")
def health():
    available = all(path.exists() for path in (
        MODULE1_ROOT / "models/best_unet.pth",
        MODULE1_ROOT / "models/best_deeplabv3plus.pth",
        MODULE2_ROOT / "models/best_residual_gan.pth",
    ))
    return jsonify(status="ok", artifacts_available=available, device=str(DEVICE))


@app.post("/api/preview")
def preview_upload():
    try:
        raw, _, _ = read_upload(request.files.get("file"))
        return jsonify(sample_id=Path(request.files["file"].filename).stem, source_preview=rgb_preview(raw))
    except (ValueError, rasterio.errors.RasterioIOError, OSError) as error:
        return jsonify(error=str(error)), 400


@app.post("/api/analyze")
def analyze():
    try:
        raw, transform, crs = read_upload(request.files.get("file"))
        erosion = int(request.form.get("erosion_pixels", 3)); expansion = int(request.form.get("expansion_pixels", 6))
        network, generator, deeplab, config = load_models()
        image = normalize_image(raw, config)
        valid_np = np.isfinite(raw).all(0).astype(np.float32)
        valid = torch.from_numpy(valid_np).view(1, 1, 512, 512).to(DEVICE)
        area_per_pixel = pixel_area_m2(transform, crs)

        with torch.inference_mode():
            baseline_probability = predict(network, image)
            source_mask = (baseline_probability >= .5).float() * valid
            pre_condition = morph(source_mask, erosion, "erode") * valid
            post_condition = morph(source_mask, expansion, "dilate") * valid
            synthetic_pre, _, _ = generator(image, source_mask, pre_condition, valid)
            synthetic_post, _, _ = generator(image, source_mask, post_condition, valid)
            pre_probability = predict(network, synthetic_pre)
            post_probability = predict(network, synthetic_post)
            pre_mask = (pre_probability >= .5).float() * valid
            post_mask = (post_probability >= .5).float() * valid
            newly = (post_mask > .5) & (pre_mask <= .5)
            receded = (pre_mask > .5) & (post_mask <= .5)

            post_mean, post_std = tta_statistics(network, synthetic_post)
            deeplab_probability = predict(deeplab, synthetic_post)
            p = post_mean.clamp(1e-6, 1 - 1e-6)
            entropy = -(p * torch.log2(p) + (1 - p) * torch.log2(1 - p))
            disagreement = torch.abs(post_mean - deeplab_probability)
            uncertainty = (.5 * entropy + .3 * post_std + .2 * disagreement) * valid

        base_np = baseline_probability[0, 0].cpu().numpy(); pre_np = pre_probability[0, 0].cpu().numpy(); post_np = post_probability[0, 0].cpu().numpy()
        pre_pixels = int(pre_mask.sum().item()); post_pixels = int(post_mask.sum().item()); new_pixels = int(newly.sum().item()); receded_pixels = int(receded.sum().item())
        uncertainty_np = uncertainty[0, 0].cpu().numpy(); valid_values = uncertainty_np[valid_np > 0]
        new_values = uncertainty_np[newly[0, 0].cpu().numpy()]
        moderate, high = np.percentile(valid_values, [50, 85]) if valid_values.size else (0, 0)
        affected_np = post_mask[0, 0].cpu().numpy() > .5
        priority_preview, priority_classes = priority_overlay(raw, affected_np, uncertainty_np, moderate, high)
        priority_counts = {name: int((priority_classes == class_id).sum()) for name, class_id in (("low", 1), ("moderate", 2), ("high", 3))}

        return jsonify(
            sample_id=Path(request.files["file"].filename).stem,
            parameters={"erosion_pixels": erosion, "expansion_pixels": expansion},
            module1={"source_preview": rgb_preview(raw), "overlay_preview": overlay_preview(raw, (base_np >= .5) * valid_np), "probability_preview": probability_preview(base_np), "flood_pixels": int(((base_np >= .5) * valid_np).sum()), "mean_probability": float(base_np[valid_np > 0].mean())},
            module2={"pre_image_preview": rgb_preview(synthetic_pre[0].cpu().numpy()), "post_image_preview": rgb_preview(synthetic_post[0].cpu().numpy()), "pre_probability_preview": probability_preview(pre_np), "post_probability_preview": probability_preview(post_np), "pre_flood_pixels": pre_pixels, "post_flood_pixels": post_pixels, "newly_inundated_pixels": new_pixels, "receded_pixels": receded_pixels, "pre_area_km2": pre_pixels * area_per_pixel / 1e6, "post_area_km2": post_pixels * area_per_pixel / 1e6, "increase_area_km2": new_pixels * area_per_pixel / 1e6},
            module3={"post_uncertainty_preview": uncertainty_preview(uncertainty_np), "priority_preview": priority_preview, "priority_counts": priority_counts, "mean_uncertainty": float(valid_values.mean()) if valid_values.size else 0, "new_area_uncertainty": float(new_values.mean()) if new_values.size else 0, "mean_predictive_entropy": float(entropy[valid > 0].mean().item()), "mean_tta_std": float(post_std[valid > 0].mean().item()), "mean_model_disagreement": float(disagreement[valid > 0].mean().item()), "moderate_threshold": float(moderate), "high_threshold": float(high), "method": "live entropy + six-transform TTA + U-Net/DeepLab disagreement"},
        )
    except (ValueError, rasterio.errors.RasterioIOError, OSError) as error:
        return jsonify(error=str(error)), 400
    except Exception as error:
        app.logger.exception("Analysis failed")
        return jsonify(error=f"Analysis failed: {error}"), 500


if __name__ == "__main__":
    print(f"FloodImpactNet assets: {SOURCE_ROOT}")
    print(f"Inference device: {DEVICE}")
    app.run(host="127.0.0.1", port=8000, debug=True)
