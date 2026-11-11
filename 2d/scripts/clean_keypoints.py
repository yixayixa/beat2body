"""4.4 Keypoint cleaning, interpolation and normalization.

Reads the raw output of ONE pose tool (4.3) and writes the cleaned dataset used
for training:

    2d/keypoints_raw/<tool>/<category>/<video_id>.npz  ->  2d/keypoints/<category>/<video_id>.npz

Steps (per joint, along time):
  1. Confidence threshold: joints below --conf become missing (NaN).
     Default depends on the tool because the scores are on different scales
     (dwpose 0.3, mediapipe 0.5, openpose 0.1).
  2. Outlier removal: a single-frame spike (jumps away and comes straight back,
     with both jumps > --outlier-factor x the joint's median displacement) is
     marked missing. Fast but consistent motion is kept.
  3. Linear interpolation over gaps of at most --max-gap frames.
  4. Savitzky-Golay smoothing (window --sg-window, order 2); 0 disables it.
  5. Normalization: the pelvis (mid-point of the two hips; COCO-17 has no pelvis
     joint) is moved to (0, 0), and coordinates are divided by one constant:
     the median torso length (mid-shoulder to mid-hip) over the whole dataset.

Output npz:
    keypoints (T, 17, 2) float32  normalized x, y; long gaps are filled linearly
    valid     (T, 17)    bool     False where a value was filled, not observed
    pelvis    (T, 2)              pelvis position in pixels (to undo the centering)
    scale                         torso length in pixels (to undo the scaling)
The normalization parameters are also written to 2d/keypoints/normalization.json.

    python 2d/scripts/clean_keypoints.py --tool dwpose
"""
import argparse
import glob
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import (CONF_THRESHOLD, COCO17_NAMES, KEYPOINTS_DIR, KEYPOINTS_RAW_DIR,  # noqa: E402
                    L_HIP, L_SHOULDER, R_HIP, R_SHOULDER)


# ---------------------------------------------------------------------------
# 1-D helpers on a single coordinate track (T,), NaN = missing
# ---------------------------------------------------------------------------
def nan_runs(mask):
    """Start/end (exclusive) of consecutive True runs."""
    m = np.concatenate([[False], mask, [False]])
    d = np.diff(m.astype(int))
    return list(zip(np.where(d == 1)[0], np.where(d == -1)[0]))


def interpolate_gaps(x, max_gap=None):
    """Linear interpolation over interior NaN gaps (all gaps if max_gap is None).
    Leading / trailing gaps are held at the nearest value only when max_gap is None."""
    x = x.copy()
    ok = np.isfinite(x)
    if ok.sum() == 0:
        return x
    idx = np.arange(len(x))
    for s, e in nan_runs(~ok):
        interior = s > 0 and e < len(x)
        if max_gap is not None and (not interior or e - s > max_gap):
            continue
        x[s:e] = np.interp(idx[s:e], idx[ok], x[ok])
    return x


def smooth_runs(x, window, order=2):
    """Savitzky-Golay filter applied separately to each run of valid values."""
    if window < 3:
        return x
    from scipy.signal import savgol_filter
    x = x.copy()
    for s, e in nan_runs(np.isfinite(x)):
        if e - s >= window:
            x[s:e] = savgol_filter(x[s:e], window, order)
    return x


# ---------------------------------------------------------------------------
# Per-video cleaning
# ---------------------------------------------------------------------------
def remove_spikes(xy, factor):
    """xy: (T, 2) for one joint. Mark single-frame spikes as NaN."""
    xy = xy.copy()
    step = np.linalg.norm(np.diff(xy, axis=0), axis=1)          # (T-1,)
    med = np.nanmedian(step) if np.isfinite(step).any() else np.nan
    if not np.isfinite(med):
        return xy, 0
    thr = factor * max(med, 1.0)
    d_in, d_out = step[:-1], step[1:]                             # around frames 1..T-2
    back = np.linalg.norm(xy[2:] - xy[:-2], axis=1)               # distance prev -> next
    spike = (d_in > thr) & (d_out > thr) & (back < 0.5 * np.minimum(d_in, d_out))
    idx = np.where(spike)[0] + 1
    xy[idx] = np.nan
    return xy, len(idx)


def clean_video(kpts, conf_thr, max_gap, outlier_factor, sg_window):
    """kpts: (T, 17, 3) raw -> (T, 17, 2) cleaned pixels with NaN for long gaps."""
    xy = kpts[:, :, :2].astype(np.float64).copy()
    low = ~(kpts[:, :, 2] >= conf_thr)                            # also True for NaN
    xy[low] = np.nan
    n_spikes = 0
    for j in range(xy.shape[1]):
        xy[:, j], n = remove_spikes(xy[:, j], outlier_factor)
        n_spikes += n
        for c in range(2):
            xy[:, j, c] = interpolate_gaps(xy[:, j, c], max_gap)
            xy[:, j, c] = smooth_runs(xy[:, j, c], sg_window)
    return xy, n_spikes


def torso_lengths(xy):
    sh = (xy[:, L_SHOULDER] + xy[:, R_SHOULDER]) / 2
    hp = (xy[:, L_HIP] + xy[:, R_HIP]) / 2
    return np.linalg.norm(sh - hp, axis=1)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--tool", required=True, choices=["dwpose", "mediapipe", "openpose"])
    p.add_argument("--categories", nargs="*", default=None)
    p.add_argument("--conf", type=float, default=None, help="confidence threshold (default: per tool)")
    p.add_argument("--max-gap", type=int, default=5, help="interpolate gaps up to this many frames")
    p.add_argument("--outlier-factor", type=float, default=3.0)
    p.add_argument("--sg-window", type=int, default=7, help="Savitzky-Golay window (odd); 0 = off")
    args = p.parse_args()
    conf_thr = CONF_THRESHOLD[args.tool] if args.conf is None else args.conf
    if args.sg_window and args.sg_window % 2 == 0:
        p.error("--sg-window must be odd")

    files = sorted(glob.glob(os.path.join(KEYPOINTS_RAW_DIR, args.tool, "*", "*.npz")))
    if args.categories:
        files = [f for f in files if os.path.basename(os.path.dirname(f)) in args.categories]
    if not files:
        sys.exit(f"no raw keypoints in {KEYPOINTS_RAW_DIR}/{args.tool}; run pose_{args.tool}.py first")

    # Pass 1: clean every video and collect torso lengths for the global scale.
    cleaned, torsos = {}, []
    for f in files:
        raw = np.load(f)
        xy, n_spikes = clean_video(raw["keypoints"], conf_thr, args.max_gap, args.outlier_factor, args.sg_window)
        cleaned[f] = (xy, raw, n_spikes)
        t = torso_lengths(xy)
        torsos.append(t[np.isfinite(t)])
    torsos = np.concatenate(torsos)
    if len(torsos) == 0:
        sys.exit("no frame has both shoulders and both hips; cannot normalize")
    scale = float(np.median(torsos))
    print(f"[{args.tool}] {len(files)} videos, median torso length {scale:.1f} px")

    # Pass 2: center on the pelvis, scale, fill long gaps, save.
    for f, (xy, raw, n_spikes) in cleaned.items():
        cat, vid = os.path.basename(os.path.dirname(f)), os.path.splitext(os.path.basename(f))[0]
        pelvis = (xy[:, L_HIP] + xy[:, R_HIP]) / 2
        for c in range(2):
            pelvis[:, c] = interpolate_gaps(pelvis[:, c], None)  # also bridges long gaps / edges
        pelvis = np.nan_to_num(pelvis, nan=0.0)
        norm = (xy - pelvis[:, None, :]) / scale
        valid = np.isfinite(norm).all(-1)
        filled = norm.copy()
        for j in range(filled.shape[1]):
            for c in range(2):
                filled[:, j, c] = interpolate_gaps(filled[:, j, c], None)
        filled = np.nan_to_num(filled, nan=0.0)                  # joints never observed

        out = os.path.join(KEYPOINTS_DIR, cat, vid + ".npz")
        os.makedirs(os.path.dirname(out), exist_ok=True)
        np.savez_compressed(out, keypoints=filled.astype(np.float32), valid=valid,
                            pelvis=pelvis.astype(np.float32), scale=scale, fps=raw["fps"],
                            width=raw["width"], height=raw["height"], tool=args.tool,
                            joint_names=np.array(COCO17_NAMES))
        print(f"  {cat}/{vid}: {len(filled)} frames, {100 * valid.mean():.1f}% joints observed, "
              f"{n_spikes} spikes removed")

    params = dict(tool=args.tool, conf_threshold=conf_thr, max_gap=args.max_gap,
                  outlier_factor=args.outlier_factor, savgol_window=args.sg_window, savgol_order=2,
                  scale_px=scale, origin="mid_hip", fps=float(cleaned[files[0]][1]["fps"]),
                  joint_names=COCO17_NAMES)
    with open(os.path.join(KEYPOINTS_DIR, "normalization.json"), "w") as fh:
        json.dump(params, fh, indent=2)


if __name__ == "__main__":
    main()
