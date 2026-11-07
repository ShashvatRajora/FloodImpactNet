import io
import json
import os
import tempfile
from pathlib import Path

import numpy as np
import rasterio
import torch
import torch.nn as nn
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image

APP_ROOT = Path(__file__).resolve().parents[2]
SOURCE_ROOT = Path(os.getenv("FLOODIMPACTNET_ROOT", APP_ROOT.parent))


def locate_bundle(name):
    direct = SOURCE_ROOT / name / name
    if direct.exists():
        return direct
    matches = list(SOURCE_ROOT.glob(f"{name}*/{name}"))
    return matches[0] if matches else SOURCE_ROOT / name


MODULE1_ROOT = locate_bundle("FloodImpactNet_Module1_UI_Assets")

app = FastAPI(title="FloodImpactNet API", version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv("CORS_ORIGINS", "*").split(","),
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


class DoubleConv(nn.Module):
    def __init__(self, input_channels, output_channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(input_channels, output_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(output_channels), nn.ReLU(inplace=True),
            nn.Conv2d(output_channels, output_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(output_channels), nn.ReLU(inplace=True),
        )
    def forward(self, x): return self.block(x)


class UNet(nn.Module):
    def __init__(self):
        super().__init__(); self.pool = nn.MaxPool2d(2)
        self.enc1=DoubleConv(8,32); self.enc2=DoubleConv(32,64); self.enc3=DoubleConv(64,128); self.enc4=DoubleConv(128,256); self.bottleneck=DoubleConv(256,512)
        self.up4=nn.ConvTranspose2d(512,256,2,2); self.dec4=DoubleConv(512,256)
        self.up3=nn.ConvTranspose2d(256,128,2,2); self.dec3=DoubleConv(256,128)
        self.up2=nn.ConvTranspose2d(128,64,2,2); self.dec2=DoubleConv(128,64)
        self.up1=nn.ConvTranspose2d(64,32,2,2); self.dec1=DoubleConv(64,32); self.output=nn.Conv2d(32,2,1)
    def forward(self,x):
        e1=self.enc1(x); e2=self.enc2(self.pool(e1)); e3=self.enc3(self.pool(e2)); e4=self.enc4(self.pool(e3)); b=self.bottleneck(self.pool(e4))
        d4=self.dec4(torch.cat([self.up4(b),e4],1)); d3=self.dec3(torch.cat([self.up3(d4),e3],1)); d2=self.dec2(torch.cat([self.up2(d3),e2],1)); d1=self.dec1(torch.cat([self.up1(d2),e1],1)); return self.output(d1)


_model = None
_config = None


def load_module1():
    global _model, _config
    if _model is not None: return _model, _config
    config_path = MODULE1_ROOT / "config" / "preprocessing.json"
    checkpoint_path = MODULE1_ROOT / "models" / "best_unet.pth"
    if not config_path.exists() or not checkpoint_path.exists():
        raise HTTPException(503, "Module 1 artifacts are not installed on this API host.")
    _config=json.loads(config_path.read_text())
    checkpoint=torch.load(checkpoint_path,map_location="cpu",weights_only=False)
    state=next((checkpoint[k] for k in ("model_state_dict","state_dict","model") if isinstance(checkpoint,dict) and k in checkpoint and isinstance(checkpoint[k],dict)),checkpoint)
    if all(k.startswith("module.") for k in state): state={k[7:]:v for k,v in state.items()}
    _model=UNet(); _model.load_state_dict(state,strict=True); _model.eval()
    return _model,_config


def png_data(values):
    values=np.nan_to_num(values); lo,hi=np.percentile(values,[2,98]); scaled=np.clip((values-lo)/max(hi-lo,1e-6)*255,0,255).astype(np.uint8)
    buffer=io.BytesIO(); Image.fromarray(scaled).save(buffer,format="PNG")
    import base64
    return "data:image/png;base64,"+base64.b64encode(buffer.getvalue()).decode()


@app.get("/health")
def health():
    return {"status":"ok","artifacts_available":MODULE1_ROOT.exists(),"module1_root":str(MODULE1_ROOT)}


@app.post("/api/analyze")
async def analyze(file: UploadFile=File(...), expansion_pixels: int=Form(6), erosion_pixels: int=Form(3)):
    if not file.filename.lower().endswith((".tif",".tiff")): raise HTTPException(400,"Upload an 8-channel GeoTIFF.")
    content=await file.read()
    if len(content)>80*1024*1024: raise HTTPException(413,"GeoTIFF exceeds the 80 MB demo limit.")
    with tempfile.NamedTemporaryFile(suffix=".tif") as tmp:
        tmp.write(content); tmp.flush()
        with rasterio.open(tmp.name) as src: image=src.read()
    if image.shape!=(8,512,512): raise HTTPException(400,f"Expected (8, 512, 512); received {image.shape}.")
    model,config=load_module1(); mean=np.asarray(config["channel_mean"],np.float32)[:,None,None]; std=np.asarray(config["channel_std"],np.float32)[:,None,None]
    image=np.where(np.isfinite(image),image,mean).astype(np.float32); image=(image-mean)/std
    with torch.inference_mode(): probability=torch.softmax(model(torch.from_numpy(image).unsqueeze(0)),1)[0,1].numpy()
    mask=(probability>=.5).astype(np.uint8)
    return {"sample_id":Path(file.filename).stem,"module1":{"flood_pixels":int(mask.sum()),"mean_probability":float(probability.mean()),"probability_preview":png_data(probability),"mask_preview":png_data(mask)},"module2":{"status":"available_when_module2_artifacts_are_installed","erosion_pixels":erosion_pixels,"expansion_pixels":expansion_pixels},"module3":{"status":"available_when_module3_artifacts_are_installed"}}
