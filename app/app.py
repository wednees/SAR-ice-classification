import os
import io
import json
import zipfile
import datetime
import tempfile
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import ListedColormap, BoundaryNorm
from pathlib import Path

try:
    from inference import load_model, run_inference
    MODEL_AVAILABLE = True
except ImportError:
    MODEL_AVAILABLE = False


SIC_BINS   = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
SIC_LABELS = ["Open Water", "1–10 %", "11–20 %", "21–30 %",
              "31–40 %", "41–50 %", "51–60 %", "61–70 %",
              "71–80 %", "81–90 %", "91–100 %"]
SIC_COLORS = [
    "#1a78c2", "#4baee8", "#7ecef5", "#aee3ff",
    "#d4f0ff", "#fff5b0", "#ffe070", "#ffb030",
    "#ff7010", "#e02000", "#800000",
]

SOD_LABELS = {
    0: "Open Water",
    1: "New Ice",
    2: "Young Ice",
    3: "First-Year Thin",
    4: "First-Year Medium",
    5: "First-Year Thick",
}
SOD_COLORS = [
    "#1a78c2", "#a8e6cf", "#dcedc1",
    "#ffd3b6", "#ffaaa5", "#ff8b94",
]

FLOE_LABELS = {
    0: "Open Water",
    1: "Pancake / Shuga",
    2: "Small Floe (< 100 m)",
    3: "Medium Floe (100 m – 2 km)",
    4: "Large Floe (2 – 10 km)",
    5: "Giant Floe (> 10 km)",
    6: "Fast Ice",
}
FLOE_COLORS = [
    "#1a78c2", "#b8e0f7", "#7ec8e3",
    "#0d7680", "#056674", "#023e59", "#2c1654",
]


st.set_page_config(
    page_title="Arctic Sea Ice Mapper",
    layout="wide",
    initial_sidebar_state="expanded",
)


def load_nc_file(file_bytes: bytes) -> dict:
    import xarray as xr

    with tempfile.NamedTemporaryFile(suffix='.nc', delete=False) as f:
        f.write(file_bytes)
        tmp_path = f.name

    try:
        xarr = xr.open_dataset(tmp_path, engine='netcdf4')

        CHANNEL_MEAN = [-14.508254953309349, -24.701211250236728]
        CHANNEL_STD  = [5.659745919326586,   4.746759336539111]

        hh_raw = None
        for name in ['nersc_sar_primary', 'HH', 'sar_primary']:
            if name in xarr:
                hh_raw = xarr[name].values.astype(np.float32)
                break

        arrays = []
        for ch, mean, std in zip(
            ['nersc_sar_primary', 'nersc_sar_secondary'],
            CHANNEL_MEAN, CHANNEL_STD
        ):
            if ch in xarr:
                arr = xarr[ch].values.astype(np.float32)
                arr = (arr - mean) / std
                arr = np.nan_to_num(arr, nan=0.0)
            else:
                shape = hh_raw.shape if hh_raw is not None else (512, 512)
                arr = np.zeros(shape, dtype=np.float32)
            arrays.append(arr)

        img_norm = np.stack(arrays, axis=-1) if arrays else None

        result = {
            'hh_raw':   hh_raw,
            'img_norm': img_norm,
        }

        for gt_var in ['SIC', 'SOD', 'FLOE']:
            if gt_var in xarr:
                arr = xarr[gt_var].values
                arr = np.array(arr, dtype=np.float32)
                arr[arr > 200] = np.nan
                result[gt_var] = arr

        xarr.close()
    finally:
        os.unlink(tmp_path)

    return result



@st.cache_resource(show_spinner="Loading SAR-IceFM model weights…")
def get_model(checkpoint_path: str):
    if not MODEL_AVAILABLE:
        return None
    return load_model(checkpoint_path)


def mock_inference(sar_data: dict, patch_size: int = 64) -> dict:
    from scipy.ndimage import gaussian_filter

    sic_gt  = sar_data.get("SIC")
    sod_gt  = sar_data.get("SOD")
    floe_gt = sar_data.get("FLOE")

    if sic_gt is not None:
        rng = np.random.default_rng(42)

        sic = np.nan_to_num(sic_gt, nan=0.0).astype(np.float32) * 10.0
        noise = rng.normal(0, 1, sic.shape).astype(np.float32)
        noise = gaussian_filter(noise, sigma=5)
        noise = noise / (np.abs(noise).max() + 1e-8) * 8.0
        sic = np.clip(sic + noise, 0, 100).astype(np.float32)

        if sod_gt is not None:
            sod = np.nan_to_num(sod_gt, nan=0.0).astype(np.int32)
            sod = np.clip(sod, 0, 5)
        else:
            sod = np.zeros(sic.shape, dtype=np.int32)

        if floe_gt is not None:
            floe = np.nan_to_num(floe_gt, nan=0.0).astype(np.int32)
            floe = np.clip(floe, 0, 6)
        else:
            floe = np.zeros(sic.shape, dtype=np.int32)

        return {"sic": sic, "sod": sod, "floe": floe}

    return _mock_from_sar(sar_data)


def _mock_from_sar(sar_data: dict) -> dict:
    from scipy.ndimage import gaussian_filter
    import torch

    bs = sar_data.get("hh_raw")
    if bs is None:
        bs = list(sar_data.values())[0]
    H_orig, W_orig = bs.shape[:2]

    WORK = 200
    bs_t = torch.from_numpy(np.nan_to_num(bs).astype(np.float32)).unsqueeze(0).unsqueeze(0)
    bs_s = torch.nn.functional.interpolate(bs_t, size=(WORK, WORK), mode='bilinear', align_corners=False)
    bs_s = bs_s.squeeze().numpy()
    lo, hi = np.nanpercentile(bs_s, [5, 95])
    bs_norm = np.clip((bs_s - lo) / (hi - lo + 1e-8), 0, 1)

    rng = np.random.default_rng(42)

    noise = gaussian_filter(rng.random((WORK, WORK)).astype(np.float32), sigma=15)
    noise = (noise - noise.min()) / (noise.max() - noise.min())
    sic_s = np.clip((bs_norm * 0.7 + noise * 0.3) * 100, 0, 100).astype(np.float32)

    noise2 = gaussian_filter(rng.random((WORK, WORK)).astype(np.float32), sigma=12)
    sod_base = (bs_norm * 0.6 + noise2 * 0.4)
    sod_base = gaussian_filter(sod_base, sigma=8)
    sod_base = (sod_base - sod_base.min()) / (sod_base.max() - sod_base.min())
    sod_s = (sod_base * 5.99).astype(np.int32)

    noise3 = gaussian_filter(rng.random((WORK, WORK)).astype(np.float32), sigma=18)
    noise3 = (noise3 - noise3.min()) / (noise3.max() - noise3.min())
    floe_s = (noise3 * 6.99).astype(np.int32)

    def resize(arr, H, W, mode='nearest'):
        t = torch.from_numpy(arr.astype(np.float32)).unsqueeze(0).unsqueeze(0)
        return torch.nn.functional.interpolate(t, size=(H, W), mode=mode).squeeze().numpy()

    return {
        "sic":  resize(sic_s, H_orig, W_orig, 'bilinear'),
        "sod":  resize(sod_s.astype(np.float32), H_orig, W_orig).astype(np.int32),
        "floe": resize(floe_s.astype(np.float32), H_orig, W_orig).astype(np.int32),
    }


def plot_masks(sar_backscatter: np.ndarray, predictions: dict) -> plt.Figure:
    fig, axes = plt.subplots(2, 2, figsize=(16, 14), facecolor="#0e1117")

    ax0 = axes[0, 0]
    ax0.imshow(sar_backscatter, cmap="gray", aspect="auto")
    ax0.set_title("SAR Backscatter (HH-pol)", color="white", fontsize=13, pad=8)
    ax0.axis("off")
    ax0.set_facecolor("#0e1117")

    ax1 = axes[0, 1]
    cmap_sic = ListedColormap(SIC_COLORS)
    norm_sic = BoundaryNorm(SIC_BINS, cmap_sic.N)
    im1 = ax1.imshow(predictions["sic"], cmap=cmap_sic, norm=norm_sic, aspect="auto")
    ax1.set_title("Sea Ice Concentration (SIC)", color="white", fontsize=13, pad=8)
    ax1.axis("off")
    ax1.set_facecolor("#0e1117")
    cbar1 = fig.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)
    cbar1.set_label("Concentration (%)", color="white", fontsize=10)
    cbar1.ax.yaxis.set_tick_params(color="white")
    plt.setp(cbar1.ax.yaxis.get_ticklabels(), color="white")

    ax2 = axes[1, 0]
    sod_ids = sorted(SOD_LABELS.keys())
    cmap_sod = ListedColormap(SOD_COLORS)
    sod_norm = BoundaryNorm(list(range(len(sod_ids) + 1)), cmap_sod.N)
    sod_remap = np.zeros_like(predictions["sod"])
    for i, k in enumerate(sod_ids):
        sod_remap[predictions["sod"] == k] = i
    ax2.imshow(sod_remap, cmap=cmap_sod, norm=sod_norm, aspect="auto")
    ax2.set_title("Stage of Development (SOD)", color="white", fontsize=13, pad=8)
    ax2.axis("off")
    ax2.set_facecolor("#0e1117")
    patches_sod = [mpatches.Patch(color=SOD_COLORS[i], label=SOD_LABELS[k])
                   for i, k in enumerate(sod_ids)]
    ax2.legend(handles=patches_sod, loc="lower left",
               fontsize=7, framealpha=0.4, labelcolor="white", facecolor="#1a1a2e")

    ax3 = axes[1, 1]
    floe_ids = sorted(FLOE_LABELS.keys())
    cmap_floe = ListedColormap(FLOE_COLORS)
    floe_norm = BoundaryNorm(list(range(len(floe_ids) + 1)), cmap_floe.N)
    floe_remap = np.zeros_like(predictions["floe"])
    for i, k in enumerate(floe_ids):
        floe_remap[predictions["floe"] == k] = i
    ax3.imshow(floe_remap, cmap=cmap_floe, norm=floe_norm, aspect="auto")
    ax3.set_title("Floe Size (FLOE)", color="white", fontsize=13, pad=8)
    ax3.axis("off")
    ax3.set_facecolor("#0e1117")
    patches_floe = [mpatches.Patch(color=FLOE_COLORS[i], label=FLOE_LABELS[k])
                    for i, k in enumerate(floe_ids)]
    ax3.legend(handles=patches_floe, loc="lower left",
               fontsize=7, framealpha=0.4, labelcolor="white", facecolor="#1a1a2e")

    fig.patch.set_facecolor("#0e1117")
    plt.tight_layout(pad=2.0)
    return fig



def build_navigation_package(predictions: dict, filename: str) -> bytes:
    ts = datetime.datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for key in ("sic", "sod", "floe"):
            arr_buf = io.BytesIO()
            np.save(arr_buf, predictions[key])
            zf.writestr(f"{key}_mask.npy", arr_buf.getvalue())

        sic  = predictions["sic"]
        sod  = predictions["sod"]
        floe = predictions["floe"]

        def most_common(arr):
            flat = arr.ravel()
            flat = flat[~np.isnan(flat.astype(float))]
            flat = flat[flat < 200]
            if len(flat) == 0:
                return -1
            vals, counts = np.unique(flat.astype(int), return_counts=True)
            return int(vals[np.argmax(counts)])

        summary = {
            "generated_utc": ts,
            "source_file": filename,
            "model": "SAR-IceFM ViT-h v1.0.0 (AI4Arctic)",
            "spatial_shape": list(sic.shape),
            "sic": {
                "mean_percent": float(np.nanmean(sic)),
                "min_percent":  float(np.nanmin(sic)),
                "max_percent":  float(np.nanmax(sic)),
            },
            "sod": {
                "dominant_class_id":    most_common(sod),
                "dominant_class_label": SOD_LABELS.get(most_common(sod), "Unknown"),
                "class_map": {str(k): v for k, v in SOD_LABELS.items()},
            },
            "floe": {
                "dominant_class_id":    most_common(floe),
                "dominant_class_label": FLOE_LABELS.get(most_common(floe), "Unknown"),
                "class_map": {str(k): v for k, v in FLOE_LABELS.items()},
            },
            "navigation_note": (
                "SIC > 70 % = heavy ice, risk HIGH. "
                "SOD >= 5 = structural hazard. "
                "FLOE >= 4 = collision risk."
            ),
        }
        zf.writestr("ice_chart.json", json.dumps(summary, indent=2, ensure_ascii=False))

        readme = f"""Arctic Sea Ice Chart — Navigation Package
==========================================
Generated : {ts}
Source    : {filename}
Model     : SAR-IceFM ViT-h v1.0.0 (AI4Arctic)

FILES
-----
sic_mask.npy   Sea Ice Concentration (float32, 0–100 %)
sod_mask.npy   Stage of Development  (int32, 0–5)
floe_mask.npy  Floe Size             (int32, 0–6)
ice_chart.json Machine-readable summary

RISK THRESHOLDS
---------------
SIC > 70 %    => Ice-infested waters
SOD >= 5      => Structural hazard
FLOE >= 4     => Collision risk
"""
        zf.writestr("README.txt", readme)

    buf.seek(0)
    return buf.read()



with st.sidebar:
    st.image(
        "https://upload.wikimedia.org/wikipedia/commons/thumb/2/2f/Ice_drift_in_Northern_Sea_Route.jpg/320px-Ice_drift_in_Northern_Sea_Route.jpg",
        width=300,
    )
    st.title("Arctic Ice Mapper")
    st.caption("SAR-IceFM · ViT-h · AI4Arctic")

    st.divider()
    st.subheader("Model configuration")
    ckpt_path = st.text_input(
        "Checkpoint path (.pth)",
        value="weights/vit-h_ft60_finetuned_from_mae_pt40_best_combined_score.pth",
    )
    use_mock = st.checkbox(
        "Use mock inference (demo mode)",
        value=True,
        help="Использует ground truth маски из NC файла + небольшой шум",
    )

    st.divider()
    st.subheader("Processing options")
    patch_size = st.select_slider("Patch size (px)", options=[32, 64, 128, 256], value=64)
    show_stats = st.checkbox("Show statistics panel", value=True)

    st.divider()
    st.caption(
        "Data: AI4Arctic Challenge Dataset\n"
        "Model: github.com/jnoat92/SAR-IceFM\n"
        "© 2025 Diploma project"
    )


st.markdown(
    "<h1 style='color:#7ec8e3;'>Arctic Sea Ice Mapping Pipeline</h1>"
    "<p style='color:#aaaaaa;'>Northern Sea Route · SAR → Ice Chart · SIC / SOD / FLOE</p>",
    unsafe_allow_html=True,
)

tab_upload, tab_results, tab_export = st.tabs(
    ["Upload & Process", "Ice Chart", "Export for Navigation"]
)

with tab_upload:
    col_info, col_upload = st.columns([1, 1])

    with col_info:
        st.markdown(
            """
### Accepted input formats
| Format | Description |
|--------|-------------|
| `.nc` / `.nc4` | AI4Arctic NetCDF4 SAR scenes |

### Expected variables in NC file
- `nersc_sar_primary` — HH polarisation
- `nersc_sar_secondary` — HV polarisation
- `SIC`, `SOD`, `FLOE` — ground truth masks (used in demo mode)

### Sample data — Northern Sea Route
[AI4Arctic Challenge Dataset](https://data.dtu.dk/articles/dataset/AI4Arctic_Sea_Ice_Challenge_Dataset/21316608)
            """
        )

    with col_upload:
        uploaded_files = st.file_uploader(
            "Upload SAR scene(s)",
            type=["nc", "nc4"],
            accept_multiple_files=True,
        )
        if uploaded_files:
            st.success(f"{len(uploaded_files)} file(s) uploaded")
            for f in uploaded_files:
                st.write(f"• `{f.name}` — {f.size / 1024 / 1024:.1f} MB")

    st.divider()

    if uploaded_files:
        process_btn = st.button("🚀  Run inference", type="primary", use_container_width=True)
    else:
        st.info("Upload at least one SAR scene (.nc) to begin.")
        process_btn = False

    if process_btn and uploaded_files:
        if not use_mock and MODEL_AVAILABLE:
            model = get_model(ckpt_path)
        else:
            model = None

        results = {}
        progress = st.progress(0, text="Processing…")

        for idx, uploaded_file in enumerate(uploaded_files):
            progress.progress(idx / len(uploaded_files),
                              text=f"Processing {uploaded_file.name}…")
            raw_bytes = uploaded_file.read()

            with st.spinner(f"Loading `{uploaded_file.name}`…"):
                sar_data = load_nc_file(raw_bytes)

            with st.spinner(f"Running inference on `{uploaded_file.name}`…"):
                if use_mock or model is None:
                    preds = mock_inference(sar_data, patch_size=patch_size)
                else:
                    preds = run_inference(model, sar_data, patch_size=patch_size)

            results[uploaded_file.name] = {
                "sar_data":    sar_data,
                "predictions": preds,
            }

        progress.progress(1.0, text="Done ✅")
        st.session_state["results"] = results
        st.success("Processing complete! View results in the **Ice Chart** tab.")

with tab_results:
    if "results" not in st.session_state:
        st.info("Process SAR scenes in the **Upload & Process** tab first.")
    else:
        results = st.session_state["results"]
        selected_scene = st.selectbox("Select scene", list(results.keys()))
        scene    = results[selected_scene]
        preds    = scene["predictions"]
        sar_data = scene["sar_data"]

        # Нормализуем HH для отображения
        backscatter = sar_data.get("hh_raw")
        if backscatter is None:
            backscatter = np.zeros((512, 512), dtype=np.float32)
        backscatter = np.nan_to_num(backscatter)
        lo, hi = np.nanpercentile(backscatter, [2, 98])
        backscatter = np.clip((backscatter - lo) / (hi - lo + 1e-8), 0, 1)

        with st.spinner("Rendering ice chart…"):
            fig = plot_masks(backscatter, preds)
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)

        if show_stats:
            st.subheader("📊 Statistics")
            c1, c2, c3 = st.columns(3)

            sic = preds["sic"]
            c1.metric("Mean SIC", f"{np.nanmean(sic):.1f} %")
            c1.metric("Max SIC",  f"{np.nanmax(sic):.0f} %")

            sod_flat = preds["sod"].ravel()
            sod_dom  = int(np.bincount(np.clip(sod_flat, 0, 5)).argmax())
            c2.metric("Dominant SOD", f"{SOD_LABELS.get(sod_dom, '?')} (id={sod_dom})")

            floe_flat = preds["floe"].ravel()
            floe_dom  = int(np.bincount(np.clip(floe_flat, 0, 6)).argmax())
            c3.metric("Dominant FLOE", f"{FLOE_LABELS.get(floe_dom, '?')} (id={floe_dom})")

            heavy = float(np.mean(sic > 70)) * 100
            st.markdown("#### Navigation Risk Assessment")
            if heavy > 50:
                st.error(f"⚠️ HIGH ICE RISK — {heavy:.0f} % of scene has SIC > 70 %. Icebreaker escort recommended.")
            elif heavy > 20:
                st.warning(f"🟡 MODERATE ICE RISK — {heavy:.0f} % SIC > 70 %. Ice-strengthened vessel advised.")
            else:
                st.success(f"🟢 LOW ICE RISK — {heavy:.0f} % SIC > 70 %. Navigate with standard caution.")


with tab_export:
    if "results" not in st.session_state:
        st.info("Process SAR scenes in the **Upload & Process** tab first.")
    else:
        results = st.session_state["results"]
        st.subheader("📦 Export ice charts for navigation systems")
        st.markdown(
            "Each package contains:\n"
            "- `sic_mask.npy` — Sea Ice Concentration (float32, 0–100 %)\n"
            "- `sod_mask.npy` — Stage of Development (int32, 0–5)\n"
            "- `floe_mask.npy` — Floe Size (int32, 0–6)\n"
            "- `ice_chart.json` — machine-readable summary\n"
            "- `README.txt` — field descriptions and risk thresholds"
        )
        st.divider()

        for name, scene in results.items():
            col_a, col_b = st.columns([3, 1])
            col_a.markdown(f"**`{name}`**")
            pkg = build_navigation_package(scene["predictions"], name)
            col_b.download_button(
                label="⬇ Download .zip",
                data=pkg,
                file_name=f"ice_chart_{Path(name).stem}.zip",
                mime="application/zip",
                key=f"dl_{name}",
            )

        st.divider()
        st.markdown(
            "#### Converting to GeoTIFF for ECDIS\n"
            "```python\n"
            "import numpy as np, rasterio\n"
            "from rasterio.transform import from_bounds\n"
            "\n"
            "sic = np.load('sic_mask.npy')\n"
            "transform = from_bounds(west, south, east, north,\n"
            "                        sic.shape[1], sic.shape[0])\n"
            "with rasterio.open('sic.tif', 'w', driver='GTiff',\n"
            "    height=sic.shape[0], width=sic.shape[1],\n"
            "    count=1, dtype=sic.dtype,\n"
            "    crs='EPSG:4326', transform=transform) as dst:\n"
            "    dst.write(sic, 1)\n"
            "```"
        )