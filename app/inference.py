import sys
import os
import numpy as np
import torch
import xarray as xr
from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent / "icefm" / "downstream-seg"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

try:
    import mmengine
    from mmengine.config import Config
    from mmseg.apis import init_model
    MMLAB_AVAILABLE = True
except ImportError:
    MMLAB_AVAILABLE = False

# Нормализационные константы из датасета AI4Arctic
CHANNEL_MEAN = [-14.508254953309349, -24.701211250236728]
CHANNEL_STD  = [5.659745919326586,   4.746759336539111]

CHANNELS = ['nersc_sar_primary', 'nersc_sar_secondary']

CROP_SIZE = (384, 384)
DOWNSAMPLE_FACTOR = 5   


def load_model(checkpoint_path: str):
    if not MMLAB_AVAILABLE:
        return None

    ckpt = Path(checkpoint_path)
    if not ckpt.exists():
        raise FileNotFoundError(f"Checkpoint не найден: {checkpoint_path}")

    cfg_path = (Path(__file__).parent.parent /
                "icefm" / "downstream-seg" / "configs" /
                "multi_task_ai4arctic" / "vit" /
                "mae_vit-huge_4xb8-amp-coslr-30ki_ai4arctic_ft60.py")

    if not cfg_path.exists():
        raise FileNotFoundError(f"Конфиг не найден: {cfg_path}")

    cfg = Config.fromfile(str(cfg_path))
    cfg = _patch_config(cfg)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = init_model(cfg, str(ckpt), device=device)
    model.eval()
    return model


def _patch_config(cfg):
    cfg.vis_backends = [dict(type='LocalVisBackend')]
    cfg.visualizer = dict(
        type='SegLocalVisualizer',
        vis_backends=[dict(type='LocalVisBackend')],
        name='visualizer'
    )
    return cfg


def load_nc_file(file_bytes: bytes) -> dict:
    import tempfile
    with tempfile.NamedTemporaryFile(suffix='.nc', delete=False) as f:
        f.write(file_bytes)
        tmp_path = f.name

    try:
        xarr = xr.open_dataset(tmp_path, engine='netcdf4')

        arrays = []
        for ch in CHANNELS:
            arr = xarr[ch].values.astype(np.float32)
            arrays.append(arr)

        img = np.stack(arrays, axis=-1)  # (H, W, 2)

        # Нормализация
        mean = np.array(CHANNEL_MEAN, dtype=np.float32)
        std  = np.array(CHANNEL_STD,  dtype=np.float32)
        img = (img - mean) / std
        img = np.nan_to_num(img, nan=255.0)

        raw_hh = xarr['nersc_sar_primary'].values.astype(np.float32)
        xarr.close()
    finally:
        os.unlink(tmp_path)

    return {
        'img_norm': img,   
        'hh_raw':   raw_hh,
    }


def run_inference(model, sar_data: dict, patch_size: int = 384) -> dict:
    if model is None:
        raise RuntimeError("Модель не загружена")

    device = next(model.parameters()).device

    img = sar_data['img_norm']  # (H, W, 2)
    H, W = img.shape[:2]

    img_t = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).float()  # (1, 2, H, W)
    H_ds = max(1, H // DOWNSAMPLE_FACTOR)
    W_ds = max(1, W // DOWNSAMPLE_FACTOR)
    img_ds = torch.nn.functional.interpolate(
        img_t, size=(H_ds, W_ds), mode='bilinear', align_corners=False
    )  # (1, 2, H_ds, W_ds)

    sic_acc  = np.zeros((H_ds, W_ds), dtype=np.float32)
    sod_acc  = np.zeros((H_ds, W_ds), dtype=np.float32)
    floe_acc = np.zeros((H_ds, W_ds), dtype=np.float32)
    count    = np.zeros((H_ds, W_ds), dtype=np.float32)

    crop_h, crop_w = CROP_SIZE
    stride_h = max(1, crop_h * 5 // 100)
    stride_w = max(1, crop_w * 5 // 100)

    from mmseg.structures import SegDataSample

    model.eval()
    with torch.no_grad():
        ys = list(range(0, max(1, H_ds - crop_h + 1), stride_h))
        xs = list(range(0, max(1, W_ds - crop_w + 1), stride_w))

        if not ys or ys[-1] + crop_h < H_ds:
            ys.append(max(0, H_ds - crop_h))
        if not xs or xs[-1] + crop_w < W_ds:
            xs.append(max(0, W_ds - crop_w))

        if H_ds < crop_h or W_ds < crop_w:
            ys = [0]
            xs = [0]

        for y in ys:
            for x in xs:
                y2 = min(y + crop_h, H_ds)
                x2 = min(x + crop_w, W_ds)
                y1 = max(0, y2 - crop_h)
                x1 = max(0, x2 - crop_w)

                patch = img_ds[:, :, y1:y2, x1:x2].to(device)

                pad_h = crop_h - patch.shape[2]
                pad_w = crop_w - patch.shape[3]
                if pad_h > 0 or pad_w > 0:
                    patch = torch.nn.functional.pad(
                        patch, (0, pad_w, 0, pad_h), value=255.0)

                data_sample = SegDataSample()
                data_sample.set_metainfo({
                    'img_shape': (patch.shape[2], patch.shape[3]),
                    'ori_shape': (patch.shape[2], patch.shape[3]),
                    'scale_factor': (1.0, 1.0),
                })

                seg_logits = model.encode_decode(patch, [data_sample])

                ph = y2 - y1
                pw = x2 - x1

                sic_pred = (seg_logits[0][0, :, :ph, :pw]
                            .argmax(dim=0).float().cpu().numpy() * 10.0)

                sod_pred = (seg_logits[1][0, :, :ph, :pw]
                            .argmax(dim=0).float().cpu().numpy())

                floe_pred = (seg_logits[2][0, :, :ph, :pw]
                             .argmax(dim=0).float().cpu().numpy())

                sic_acc [y1:y2, x1:x2] += sic_pred
                sod_acc [y1:y2, x1:x2] += sod_pred
                floe_acc[y1:y2, x1:x2] += floe_pred
                count   [y1:y2, x1:x2] += 1.0

    count = np.maximum(count, 1.0)

    return {
        'sic':  np.clip(sic_acc  / count, 0, 100).astype(np.float32),
        'sod':  np.clip(np.round(sod_acc  / count), 0, 5).astype(np.int32),
        'floe': np.clip(np.round(floe_acc / count), 0, 6).astype(np.int32),
    }
