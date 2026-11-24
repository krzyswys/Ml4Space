"""Download real Sentinel-5P (NO2, CO, O3) COG crops over Warsaw from AWS Open Data.

Requires: boto3, rasterio, numpy
Source: s3://meeo-s5p (no AWS account; AWS_NO_SIGN_REQUEST=YES)
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import boto3
import numpy as np
import rasterio
from botocore import UNSIGNED
from botocore.config import Config
from rasterio.enums import Resampling
from rasterio.transform import from_bounds as transform_from_bounds
from rasterio.warp import reproject
from rasterio.windows import from_bounds

os.environ["AWS_NO_SIGN_REQUEST"] = "YES"
os.environ["AWS_DEFAULT_REGION"] = "eu-central-1"

BBOX = (20.5, 51.8, 21.6, 52.5)  # W,S,E,N
SIZE = 200
OUT = Path(__file__).resolve().parent / "data" / "s5p"
OUT.mkdir(parents=True, exist_ok=True)

PRODUCTS = {
    "NO2": ("L2__NO2___", "PRODUCT_nitrogendioxide_tropospheric_column_4326.tif"),
    "CO": ("L2__CO____", "PRODUCT_carbonmonoxide_total_column_4326.tif"),
    "O3": ("L2__O3____", "PRODUCT_ozone_total_vertical_column_4326.tif"),
}

s3 = boto3.client("s3", config=Config(signature_version=UNSIGNED), region_name="eu-central-1")


def list_keys(prod_dir: str, suffix: str, y: int, m: int, d: int) -> list[str]:
    prefix = f"COGT/OFFL/{prod_dir}/{y}/{m:02d}/{d:02d}/"
    keys, token = [], None
    while True:
        kw = {"Bucket": "meeo-s5p", "Prefix": prefix}
        if token:
            kw["ContinuationToken"] = token
        r = s3.list_objects_v2(**kw)
        for c in r.get("Contents", []):
            if c["Key"].endswith(suffix):
                keys.append(c["Key"])
        if not r.get("IsTruncated"):
            break
        token = r["NextContinuationToken"]
    return keys


def read_crop(key: str) -> np.ndarray | None:
    with rasterio.Env(AWS_NO_SIGN_REQUEST="YES", AWS_DEFAULT_REGION="eu-central-1"):
        with rasterio.open(f"/vsis3/meeo-s5p/{key}") as src:
            west, south, east, north = BBOX
            win = from_bounds(west, south, east, north, transform=src.transform)
            data = src.read(1, window=win, boundless=True, fill_value=np.nan).astype(np.float32)
            if src.nodata is not None:
                data = np.where(data == src.nodata, np.nan, data)
            valid = np.isfinite(data) & (np.abs(data) > 0)
            if int(valid.sum()) < 40:
                return None
            fill = float(np.nanmean(data[valid]))
            filled = np.where(np.isfinite(data), data, fill)
            dst = np.empty((SIZE, SIZE), dtype=np.float32)
            dst_transform = transform_from_bounds(west, south, east, north, SIZE, SIZE)
            reproject(
                source=filled,
                destination=dst,
                src_transform=src.window_transform(win),
                src_crs=src.crs,
                dst_transform=dst_transform,
                dst_crs="EPSG:4326",
                resampling=Resampling.bilinear,
            )
            return dst


def tstamp(key: str) -> str:
    name = Path(key).name
    i = name.find("_20")
    return name[i + 1 : i + 16] if i >= 0 else name


def nearest(keys: list[str], nt: str) -> str | None:
    if not keys:
        return None
    return min(keys, key=lambda k: abs(int(tstamp(k)[9:13]) - int(nt[9:13])))


def stacks_for_day(y: int, m: int, d: int):
    no2_keys = list_keys(*PRODUCTS["NO2"], y, m, d)
    co_keys = list_keys(*PRODUCTS["CO"], y, m, d)
    o3_keys = list_keys(*PRODUCTS["O3"], y, m, d)
    out = []
    for nk in no2_keys:
        no2 = read_crop(nk)
        if no2 is None:
            continue
        nt = tstamp(nk)
        ck, ok = nearest(co_keys, nt), nearest(o3_keys, nt)
        co = read_crop(ck) if ck else None
        o3 = read_crop(ok) if ok else None
        if co is None or o3 is None:
            continue
        out.append((nt, np.stack([no2, co, o3], axis=-1), nk))
    return out


def harvest(dates, season: str, max_n: int = 12):
    meta, n = [], 0
    for y, m, d in dates:
        stacks = stacks_for_day(y, m, d)
        print(f"{season} {y}-{m:02d}-{d:02d}: {len(stacks)} stacks")
        for nt, stack, src in stacks:
            path = OUT / f"{season}_{y}{m:02d}{d:02d}_{n:02d}.npy"
            np.save(path, stack)
            meta.append(
                {
                    "file": path.name,
                    "season": season,
                    "date": f"{y}-{m:02d}-{d:02d}",
                    "tstamp": nt,
                    "src": src,
                }
            )
            print(" saved", path.name)
            n += 1
            if n >= max_n:
                return meta
    return meta


if __name__ == "__main__":
    # wipe previous npy
    for f in OUT.glob("*.npy"):
        f.unlink()
    winter = [(2024, 1, d) for d in range(8, 28)]
    summer = [(2024, 7, d) for d in range(8, 28)]
    meta = harvest(winter, "winter", 12) + harvest(summer, "summer", 12)
    (OUT / "manifest.json").write_text(json.dumps(meta, indent=2))
    print("done", len(meta), "files →", OUT)
