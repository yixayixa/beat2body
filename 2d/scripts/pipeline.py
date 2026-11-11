"""Shared code and every non-pose step of the 2D pipeline (2d/README.md, section 4).

The three pose tools (4.3) are separate scripts: pose_dwpose.py, pose_mediapipe.py,
pose_openpose.py. Everything they share and all other steps live in this file:

    python 2d/scripts/pipeline.py frames ...           4.2 frame extraction (optional)
    python 2d/scripts/pipeline.py clean --tool NAME    4.4 keypoint cleaning and normalization
    python 2d/scripts/pipeline.py audio ...            4.5 audio extraction and features
    python 2d/scripts/pipeline.py clips ...            4.6 alignment + 4.7 split, clips and meta.csv
    python 2d/scripts/pipeline.py all --tool NAME      clean + audio + clips (add --pose to run 4.3 first)

Run from the repository root. Use `python 2d/scripts/pipeline.py <step> --help` for options.
"""
import argparse
import csv
import glob
import json
import os
import random
import subprocess
import sys
from collections import defaultdict

import numpy as np

sys.dont_write_bytecode = True  # keep the repo free of __pycache__ folders

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT_2D = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
RAW_VIDEO_DIR = os.path.join(ROOT_2D, "raw_video")
FRAMES_DIR = os.path.join(ROOT_2D, "frames")
KEYPOINTS_DIR = os.path.join(ROOT_2D, "keypoints")
AUDIO_DIR = os.path.join(ROOT_2D, "audio")
FEATURES_DIR = os.path.join(ROOT_2D, "features")
CLIPS_DIR = os.path.join(ROOT_2D, "clips")
META_CSV = os.path.join(ROOT_2D, "meta.csv")
CATEGORIES_JSON = os.path.join(ROOT_2D, "categories.json")

VIDEO_EXTS = (".mp4", ".mov", ".avi", ".mkv", ".webm")
DEFAULT_FPS = 30

# ---------------------------------------------------------------------------
# Unified skeleton: COCO-17 (x, y in pixels, confidence)
# ---------------------------------------------------------------------------
COCO17_NAMES = [
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle",
]
COCO17_EDGES = [
    (5, 7), (7, 9), (6, 8), (8, 10), (5, 6), (5, 11), (6, 12), (11, 12),
    (11, 13), (13, 15), (12, 14), (14, 16), (0, 1), (0, 2), (1, 3), (2, 4),
]
L_SHOULDER, R_SHOULDER, L_HIP, R_HIP = 5, 6, 11, 12

# OpenPose BODY-18 order (used by DWPose's body output and OpenPose)
OPENPOSE18_NAMES = [
    "nose", "neck", "right_shoulder", "right_elbow", "right_wrist",
    "left_shoulder", "left_elbow", "left_wrist", "right_hip", "right_knee",
    "right_ankle", "left_hip", "left_knee", "left_ankle", "right_eye",
    "left_eye", "right_ear", "left_ear",
]
OPENPOSE18_TO_COCO17 = [0, 15, 14, 17, 16, 5, 2, 6, 3, 7, 4, 11, 8, 12, 9, 13, 10]

# MediaPipe Pose 33 landmarks -> COCO-17
MEDIAPIPE33_TO_COCO17 = [0, 2, 5, 7, 8, 11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28]

# Default "joint is reliable" confidence threshold per tool. The three tools
# report confidence on different scales, so one global threshold is not fair.
CONF_THRESHOLD = {"dwpose": 0.3, "mediapipe": 0.5, "openpose": 0.1}


# ---------------------------------------------------------------------------
# Videos and categories
# ---------------------------------------------------------------------------
def list_categories(video_dir=RAW_VIDEO_DIR):
    """Categories are the sub-folders of raw_video/ that contain videos."""
    cats = []
    for d in sorted(os.listdir(video_dir)):
        p = os.path.join(video_dir, d)
        if os.path.isdir(p) and any(f.lower().endswith(VIDEO_EXTS) for f in os.listdir(p)):
            cats.append(d)
    return cats


def list_videos(categories=None, video_dir=RAW_VIDEO_DIR, limit=None):
    """Return [(category, video_id, path)] sorted by category and name."""
    categories = categories or list_categories(video_dir)
    out = []
    for c in categories:
        files = sorted(f for f in glob.glob(os.path.join(video_dir, c, "*")) if f.lower().endswith(VIDEO_EXTS))
        if limit:
            files = files[:limit]
        out += [(c, os.path.splitext(os.path.basename(f))[0], f) for f in files]
    return out


def dancer_id(video_id):
    """AIST file names carry the dancer as 'dXX' (e.g. gLH_sFM_c01_d16_...)."""
    for part in video_id.split("_"):
        if len(part) == 3 and part[0] == "d" and part[1:].isdigit():
            return part
    return None


def write_categories_json(categories, path=CATEGORIES_JSON):
    """Fixed name -> index mapping used for the category embedding."""
    mapping = {c: i for i, c in enumerate(sorted(categories))}
    with open(path, "w") as f:
        json.dump(mapping, f, indent=2)
    return mapping


# ---------------------------------------------------------------------------
# Frame reading at a fixed frame rate (ffmpeg resampling)
# ---------------------------------------------------------------------------
def probe_video(path):
    cmd = ["ffprobe", "-v", "error", "-select_streams", "v:0",
           "-show_entries", "stream=width,height,r_frame_rate", "-of", "json", path]
    s = json.loads(subprocess.check_output(cmd))["streams"][0]
    num, den = s["r_frame_rate"].split("/")
    return int(s["width"]), int(s["height"]), float(num) / float(den)


def iter_frames(path, fps=DEFAULT_FPS, height=None):
    """Yield BGR uint8 frames resampled to a constant `fps` (AIST videos are 59.94 fps).

    Using the same constant fps here and in the audio features keeps motion
    frame t aligned with audio feature frame t (README 4.6).
    """
    w, h, _ = probe_video(path)
    vf = f"fps={fps}"
    if height and height != h:
        w = int(round(w * height / h / 2)) * 2
        h = height
        vf += f",scale={w}:{h}"
    cmd = ["ffmpeg", "-v", "error", "-i", path, "-vf", vf, "-f", "rawvideo", "-pix_fmt", "bgr24", "-"]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE)
    size = w * h * 3
    try:
        while True:
            buf = proc.stdout.read(size)
            if len(buf) < size:
                break
            yield np.frombuffer(buf, np.uint8).reshape(h, w, 3)
    finally:
        proc.stdout.close()
        proc.wait()


# ---------------------------------------------------------------------------
# Keypoint files
# ---------------------------------------------------------------------------
def raw_keypoint_path(tool, category, video_id):
    """Output of a pose tool (4.3): 2d/keypoints/<category>/<video_id>_<tool>.npz"""
    return os.path.join(KEYPOINTS_DIR, category, f"{video_id}_{tool}.npz")


def preview_path(tool, category, video_id):
    """Skeleton overlay video: 2d/keypoints/<category>/<video_id>_<tool>.mp4"""
    return os.path.join(KEYPOINTS_DIR, category, f"{video_id}_{tool}.mp4")


def feature_path(kind, video_id):
    """Audio features (4.5): 2d/features/<kind>/<video_id>.npy (AIST ids are unique across categories)"""
    return os.path.join(FEATURES_DIR, kind, video_id + ".npy")


def clean_keypoint_path(category, video_id):
    """Cleaned keypoints used for training (4.4): 2d/keypoints/<category>/<video_id>.npz"""
    return os.path.join(KEYPOINTS_DIR, category, video_id + ".npz")


def save_raw_keypoints(path, keypoints, native, native_names, tool, fps, width, height, seconds):
    """keypoints: (T, 17, 3) COCO-17 x, y (pixels), confidence; NaN where no person was found.
    native: (T, J, 3) the tool's own joint layout, kept for later use."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.savez_compressed(
        path,
        keypoints=keypoints.astype(np.float32),
        native=native.astype(np.float32),
        native_joint_names=np.array(native_names),
        joint_names=np.array(COCO17_NAMES),
        tool=tool, fps=fps, width=width, height=height,
        processing_seconds=seconds,
    )


# ---------------------------------------------------------------------------
# Drawing (preview videos)
# ---------------------------------------------------------------------------
def draw_skeleton(img, kpts, thr, color=(0, 255, 0)):
    import cv2
    ok = lambda j: np.isfinite(kpts[j, 0]) and kpts[j, 2] >= thr
    for a, b in COCO17_EDGES:
        if ok(a) and ok(b):
            cv2.line(img, tuple(int(v) for v in kpts[a, :2]), tuple(int(v) for v in kpts[b, :2]),
                     color, 3, cv2.LINE_AA)
    for j in range(len(kpts)):
        if ok(j):
            cv2.circle(img, tuple(int(v) for v in kpts[j, :2]), 5, (255, 255, 255), -1, cv2.LINE_AA)
            cv2.circle(img, tuple(int(v) for v in kpts[j, :2]), 5, color, 2, cv2.LINE_AA)
    return img


class PreviewWriter:
    """Writes an H.264 mp4 of frames with the skeleton drawn on top."""

    def __init__(self, path, fps, width, height, title, max_height=720):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self.path, self.title = path, title
        self.scale = min(1.0, max_height / height)
        self.size = (int(width * self.scale) // 2 * 2, int(height * self.scale) // 2 * 2)
        cmd = ["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "bgr24",
               "-s", f"{self.size[0]}x{self.size[1]}", "-r", str(fps), "-i", "-",
               "-c:v", "libx264", "-preset", "veryfast", "-crf", "23", "-pix_fmt", "yuv420p", path]
        self.proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)

    def write(self, frame, kpts, thr, color=(0, 255, 0)):
        import cv2
        img = draw_skeleton(frame.copy(), kpts, thr, color)
        img = cv2.resize(img, self.size, interpolation=cv2.INTER_AREA)
        cv2.rectangle(img, (0, 0), (img.shape[1], 40), (0, 0, 0), -1)
        cv2.putText(img, self.title, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
        self.proc.stdin.write(np.ascontiguousarray(img).tobytes())

    def close(self):
        self.proc.stdin.close()
        self.proc.wait()


# ---------------------------------------------------------------------------
# Shared command line for the three pose scripts
# ---------------------------------------------------------------------------
def add_preview_arg(p, tool="<tool>"):
    p.add_argument("--preview", type=int, nargs="?", const=-1, default=0, metavar="N",
                   help=f"also write skeleton preview videos (<video_id>_{tool}.mp4): "
                        "--preview alone = every video, --preview N = the first N videos")


def wants_preview(args, n):
    return args.preview < 0 or n < args.preview


def pose_arg_parser(tool, description):
    p = argparse.ArgumentParser(description=description)
    p.add_argument("--categories", nargs="*", default=None,
                   help="sub-folders of 2d/raw_video to process (default: all)")
    p.add_argument("--videos", nargs="*", default=None,
                   help="process these video files instead (category = parent folder name)")
    p.add_argument("--limit", type=int, default=None, help="at most N videos per category")
    p.add_argument("--fps", type=float, default=DEFAULT_FPS, help="resample videos to this frame rate")
    p.add_argument("--height", type=int, default=None,
                   help="resize frames to this height before inference (default: original size)")
    add_preview_arg(p, tool)
    p.add_argument("--overwrite", action="store_true", help="re-run videos that already have output")
    return p


def select_videos(args):
    if args.videos:
        return [(os.path.basename(os.path.dirname(os.path.abspath(v))),
                 os.path.splitext(os.path.basename(v))[0], v) for v in args.videos]
    return list_videos(args.categories, limit=args.limit)


def run_pose_tool(tool, args, make_infer, native_names, color):
    """Loop over videos. make_infer(fps) is called once per video and returns
    infer(frame_bgr) -> (coco17 (17, 3) or None, native (J, 3) or None)."""
    import time
    videos = select_videos(args)
    print(f"[{tool}] {len(videos)} videos")
    for n, (cat, vid, path) in enumerate(videos):
        out = raw_keypoint_path(tool, cat, vid)
        if os.path.exists(out) and not args.overwrite:
            print(f"[{tool}] skip {cat}/{vid} (exists)")
            continue
        w, h, _ = probe_video(path)
        if args.height:
            w, h = int(round(w * args.height / h / 2)) * 2, args.height
        preview = None
        if wants_preview(args, n):
            preview = PreviewWriter(preview_path(tool, cat, vid), args.fps, w, h, f"{tool}  {cat}/{vid}")
        infer = make_infer(args.fps)
        coco, native = [], []
        t0 = time.perf_counter()
        for frame in iter_frames(path, args.fps, args.height):
            k, kn = infer(frame)
            k = np.full((17, 3), np.nan, np.float32) if k is None else k
            kn = np.full((len(native_names), 3), np.nan, np.float32) if kn is None else kn
            coco.append(k)
            native.append(kn)
            if preview:
                preview.write(frame, k, CONF_THRESHOLD[tool], color)
        dt = time.perf_counter() - t0
        if preview:
            preview.close()
        coco, native = np.stack(coco), np.stack(native)
        save_raw_keypoints(out, coco, native, native_names, tool, args.fps, w, h, dt)
        found = 100 * (~np.isnan(coco[:, :, 0]).all(1)).mean()
        print(f"[{tool}] {cat}/{vid}: {len(coco)} frames, {len(coco) / dt:.1f} fps, person found in {found:.0f}% of frames")


# ===========================================================================
# 4.2 Frame extraction
# ===========================================================================
FRAMES_DOC = """4.2 Frame extraction: dump JPEG frames at a fixed frame rate.

Optional. The pose scripts in 4.3 read videos directly at the same fixed frame
rate, so JPEGs are only needed for inspection or for external tools. Note that
1080p JPEGs for every video take a lot of disk space (roughly 20+ GB for all
210 AIST clips); use --categories / --limit / --height to keep it small.

    python 2d/scripts/pipeline.py frames --categories la_hiphop --limit 1
    -> 2d/frames/la_hiphop/<video_id>/000001.jpg ...
"""


def add_frames_args(p):
    p.add_argument("--categories", nargs="*", default=None)
    p.add_argument("--limit", type=int, default=None, help="at most N videos per category")
    p.add_argument("--fps", type=float, default=DEFAULT_FPS)
    p.add_argument("--height", type=int, default=None, help="resize to this height (default: original)")
    p.add_argument("--quality", type=int, default=3, help="ffmpeg JPEG quality, 2 (best) to 31")


def run_frames(args):
    for cat, vid, path in list_videos(args.categories, limit=args.limit):
        out_dir = os.path.join(FRAMES_DIR, cat, vid)
        if os.path.isdir(out_dir) and os.listdir(out_dir):
            print("skip", out_dir)
            continue
        os.makedirs(out_dir, exist_ok=True)
        vf = f"fps={args.fps}" + (f",scale=-2:{args.height}" if args.height else "")
        subprocess.run(["ffmpeg", "-v", "error", "-i", path, "-vf", vf, "-q:v", str(args.quality),
                        os.path.join(out_dir, "%06d.jpg")], check=True)
        print(f"{cat}/{vid}: {len(os.listdir(out_dir))} frames")


# ===========================================================================
# 4.4 Keypoint cleaning, interpolation and normalization
# ===========================================================================
CLEAN_DOC = """4.4 Keypoint cleaning, interpolation and normalization.

Reads the raw output of ONE pose tool (4.3) and writes the cleaned dataset used
for training:

    2d/keypoints/<category>/<video_id>_<tool>.npz  ->  2d/keypoints/<category>/<video_id>.npz

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

    python 2d/scripts/pipeline.py clean --tool dwpose
"""


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


def add_clean_args(p):
    p.add_argument("--tool", required=True, choices=["dwpose", "mediapipe", "openpose"])
    p.add_argument("--categories", nargs="*", default=None)
    p.add_argument("--conf", type=float, default=None, help="confidence threshold (default: per tool)")
    p.add_argument("--max-gap", type=int, default=5, help="interpolate gaps up to this many frames")
    p.add_argument("--outlier-factor", type=float, default=3.0)
    p.add_argument("--sg-window", type=int, default=7, help="Savitzky-Golay window (odd); 0 = off")


def run_clean(args):
    conf_thr = CONF_THRESHOLD[args.tool] if args.conf is None else args.conf
    if args.sg_window and args.sg_window % 2 == 0:
        sys.exit("--sg-window must be odd")

    files = sorted(glob.glob(os.path.join(KEYPOINTS_DIR, "*", f"*_{args.tool}.npz")))
    if args.categories:
        files = [f for f in files if os.path.basename(os.path.dirname(f)) in args.categories]
    if not files:
        sys.exit(f"no *_{args.tool}.npz files in {KEYPOINTS_DIR}; run pose_{args.tool}.py first")

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
        cat = os.path.basename(os.path.dirname(f))
        vid = os.path.basename(f)[: -len(f"_{args.tool}.npz")]
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

        out = clean_keypoint_path(cat, vid)
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


# ===========================================================================
# 4.5 Audio extraction and features
# ===========================================================================
AUDIO_DOC = """4.5 Audio extraction and features.

For every video:
    2d/audio/<category>/<video_id>.wav                  mono, 22.05 kHz
    2d/features/audio_mel/<video_id>.npy     (T, 64) log-mel spectrogram (dB)
    2d/features/audio_onset/<video_id>.npy   (T,)    onset strength
    2d/features/beats/<video_id>.npy         (num_beats,) beat frame indices

hop_length = sample_rate / fps (22050 / 30 = 735), so feature frame t lines up
with motion frame t from 4.3 (both use the same fixed fps).

Install:
    pip install librosa soundfile

    python 2d/scripts/pipeline.py audio --categories la_hiphop house
"""


def extract_wav(video, wav, sr):
    os.makedirs(os.path.dirname(wav), exist_ok=True)
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", video, "-vn", "-ac", "1", "-ar", str(sr), wav], check=True)


def save(kind, vid, arr):
    path = feature_path(kind, vid)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.save(path, arr)


def add_audio_args(p):
    p.add_argument("--categories", nargs="*", default=None)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--sr", type=int, default=22050)
    p.add_argument("--fps", type=float, default=DEFAULT_FPS)
    p.add_argument("--n-mels", type=int, default=64)
    p.add_argument("--n-fft", type=int, default=2048)
    p.add_argument("--overwrite", action="store_true")


def run_audio(args):
    import librosa

    hop = args.sr / args.fps
    if abs(hop - round(hop)) > 1e-6:
        sys.exit(f"sample rate {args.sr} / fps {args.fps} is not an integer hop length")
    hop = int(round(hop))

    for cat, vid, path in list_videos(args.categories, limit=args.limit):
        wav = os.path.join(AUDIO_DIR, cat, vid + ".wav")
        done = feature_path("beats", vid)
        if os.path.exists(done) and not args.overwrite:
            print("skip", f"{cat}/{vid}")
            continue
        extract_wav(path, wav, args.sr)
        y, sr = librosa.load(wav, sr=args.sr, mono=True)

        mel = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=args.n_fft, hop_length=hop, n_mels=args.n_mels)
        log_mel = librosa.power_to_db(mel, ref=np.max).T.astype(np.float32)          # (T, n_mels)
        onset_env = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop).astype(np.float32)  # (T,)
        tempo, beats = librosa.beat.beat_track(onset_envelope=onset_env, sr=sr, hop_length=hop)

        save("audio_mel", vid, log_mel)
        save("audio_onset", vid, onset_env)
        save("beats", vid, np.asarray(beats, dtype=np.int32))                   # written last = done marker
        bpm = float(np.atleast_1d(tempo)[0])
        print(f"{cat}/{vid}: {len(log_mel)} frames, {len(beats)} beats (~{bpm:.0f} BPM)")


# ===========================================================================
# 4.6 Alignment + 4.7 Dataset split
# ===========================================================================
CLIPS_DOC = """4.6 Alignment + 4.7 Dataset split: sliding-window clips and meta.csv.

Split (4.7): by video, not by clip, stratified per category (default 70 / 15 / 15,
so 21 videos per category -> 15 / 3 / 3). With --by dancer, all videos of one
dancer go to the same split (AIST has 3 dancers per genre -> one dancer per split).
The split is fixed by --seed and recorded in meta.csv.

Alignment (4.6): motion frame t and audio feature frame t refer to the same time
(both use the same fixed fps), so both are trimmed to the shorter length, then
each video is cut into overlapping windows.

Output:
    2d/clips/<split>/<category>__<video_id>__<start>.npz
        keypoints (W, 17, 2), valid (W, 17), mel (W, 64), onset (W,),
        beats (beat frame indices inside the window), category, category_index,
        video_id, start, end, fps
    2d/meta.csv
        clip_path, audio_feat_path, category, start_frame, end_frame, split, video_id, dancer
    2d/categories.json
        {"<category>": index}, the fixed mapping for the category embedding

Run after the clean (4.4) and audio (4.5) steps:

    python 2d/scripts/pipeline.py clips --window 240 --stride 60
"""


REPO_ROOT = os.path.dirname(ROOT_2D)
SPLITS = ["train", "val", "test"]


def rel(path):
    return os.path.relpath(path, REPO_ROOT)


def allocate(n, ratios):
    """Counts per split that sum to n, each split >= 1 when n >= 3."""
    counts = [int(round(n * r)) for r in ratios]
    counts[0] = n - sum(counts[1:])
    if n >= 3:
        for i in (1, 2):
            if counts[i] == 0:
                counts[i], counts[0] = 1, counts[0] - 1
    return counts


def split_videos(videos, ratios, by, seed):
    """videos: [(category, video_id)] -> {(category, video_id): split}"""
    rng = random.Random(seed)
    by_cat = defaultdict(list)
    for cat, vid in videos:
        by_cat[cat].append(vid)
    splits = {}
    for cat, vids in sorted(by_cat.items()):
        if by == "dancer":
            groups = defaultdict(list)
            for v in vids:
                groups[dancer_id(v) or v].append(v)
            units = [sorted(g) for _, g in sorted(groups.items())]
        else:
            units = [[v] for v in sorted(vids)]
        rng.shuffle(units)
        i = 0
        for split, c in zip(SPLITS, allocate(len(units), ratios)):
            for unit in units[i:i + c]:
                for v in unit:
                    splits[(cat, v)] = split
            i += c
    return splits


def add_clips_args(p):
    p.add_argument("--categories", nargs="*", default=None)
    p.add_argument("--window", type=int, default=240, help="clip length in frames (README: 120-300)")
    p.add_argument("--stride", type=int, default=60)
    p.add_argument("--min-length", type=int, default=120,
                   help="videos shorter than --window but at least this long become one shorter clip")
    p.add_argument("--min-valid", type=float, default=0.5,
                   help="drop clips where less than this fraction of joints was observed")
    p.add_argument("--ratios", nargs=3, type=float, default=[0.7, 0.15, 0.15], metavar=("TRAIN", "VAL", "TEST"))
    p.add_argument("--by", choices=["video", "dancer"], default="video")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--limit", type=int, default=None, help="at most N videos per category")


def run_clips(args):
    # Only videos that have both cleaned keypoints and audio features take part.
    videos, skipped = [], []
    for cat, vid, _ in list_videos(args.categories, limit=getattr(args, "limit", None)):
        need = [clean_keypoint_path(cat, vid)] + [
            feature_path(k, vid) for k in ("audio_mel", "audio_onset", "beats")]
        missing = [rel(x) for x in need if not os.path.exists(x)]
        if missing:
            skipped.append((f"{cat}/{vid}", "missing " + ", ".join(missing)))
        else:
            videos.append((cat, vid))
    if not videos:
        sys.exit("no video has both cleaned keypoints and audio features; run the clean (4.4) and audio (4.5) steps first")

    splits = split_videos(videos, args.ratios, args.by, args.seed)
    cat_index = write_categories_json({c for c, _ in videos})

    for old in glob.glob(os.path.join(CLIPS_DIR, "*", "*.npz")):
        os.remove(old)

    rows = []
    for (cat, vid), split in sorted(splits.items()):
        kp = np.load(clean_keypoint_path(cat, vid))
        mel_path = feature_path("audio_mel", vid)
        mel = np.load(mel_path)
        onset = np.load(feature_path("audio_onset", vid))
        beats = np.load(feature_path("beats", vid))
        T = min(len(kp["keypoints"]), len(mel), len(onset))
        if abs(len(kp["keypoints"]) - len(mel)) > 3:
            print(f"warning {cat}/{vid}: motion {len(kp['keypoints'])} vs audio {len(mel)} frames")
        if T < args.min_length:
            skipped.append((f"{cat}/{vid}", f"too short ({T} frames)"))
            continue
        win = min(args.window, T)
        for s in range(0, T - win + 1, args.stride):
            e = s + win
            valid = kp["valid"][s:e]
            if valid.mean() < args.min_valid:
                continue
            out = os.path.join(CLIPS_DIR, split, f"{cat}__{vid}__{s:06d}.npz")
            os.makedirs(os.path.dirname(out), exist_ok=True)
            b = beats[(beats >= s) & (beats < e)] - s
            np.savez_compressed(out, keypoints=kp["keypoints"][s:e], valid=valid, mel=mel[s:e],
                                onset=onset[s:e], beats=b.astype(np.int32), category=cat,
                                category_index=cat_index[cat], video_id=vid, start=s, end=e, fps=kp["fps"])
            rows.append(dict(clip_path=rel(out), audio_feat_path=rel(mel_path), category=cat,
                             start_frame=s, end_frame=e, split=split, video_id=vid, dancer=dancer_id(vid) or ""))

    with open(META_CSV, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["clip_path", "audio_feat_path", "category", "start_frame",
                                          "end_frame", "split", "video_id", "dancer"])
        w.writeheader()
        w.writerows(rows)

    for cat in sorted({c for c, _ in videos}):
        n = {s: sum(1 for (c, _), v in splits.items() if c == cat and v == s) for s in SPLITS}
        print(f"{cat}: videos {n}")
    for s in SPLITS:
        print(f"{s}: {sum(r['split'] == s for r in rows)} clips")
    if skipped:
        print(f"skipped {len(skipped)} videos, e.g.:")
        for key, why in skipped[:5]:
            print(f"  {key}: {why}")
    print(f"wrote {rel(META_CSV)}")


# ===========================================================================
# Command line
# ===========================================================================
ALL_DOC = """4.4 + 4.5 + 4.6 + 4.7 in one command: clean -> audio -> clips.

With --pose, the pose tool given by --tool (4.3) runs first, so a single
command goes from the videos to the clips:

    python 2d/scripts/pipeline.py all --tool dwpose --categories la_hiphop
    python 2d/scripts/pipeline.py all --tool dwpose --categories la_hiphop --pose --preview

Every option of the clean, audio and clips steps is accepted here as well.
"""


def add_all_args(p):
    add_clean_args(p)
    add_audio_args(p)
    add_clips_args(p)
    p.add_argument("--pose", action="store_true", help="run pose_<tool>.py first (4.3)")
    add_preview_arg(p)


def run_all(args):
    if args.pose:
        script = os.path.join(os.path.dirname(os.path.abspath(__file__)), f"pose_{args.tool}.py")
        cmd = [sys.executable, script]
        if args.categories:
            cmd += ["--categories", *args.categories]
        if args.limit:
            cmd += ["--limit", str(args.limit)]
        if args.preview:
            cmd += ["--preview"] + ([] if args.preview < 0 else [str(args.preview)])
        if args.overwrite:
            cmd += ["--overwrite"]
        print(f"== 4.3 pose ({args.tool})")
        subprocess.run(cmd, check=True)
    print(f"== 4.4 clean ({args.tool})")
    run_clean(args)
    print("== 4.5 audio")
    run_audio(args)
    print("== 4.6 + 4.7 clips")
    run_clips(args)


STEPS = {
    "frames": (FRAMES_DOC, add_frames_args, run_frames),
    "clean": (CLEAN_DOC, add_clean_args, run_clean),
    "audio": (AUDIO_DOC, add_audio_args, run_audio),
    "clips": (CLIPS_DOC, add_clips_args, run_clips),
    "all": (ALL_DOC, add_all_args, run_all),
}


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="step", required=True, metavar="{frames,clean,audio,clips,all}")
    for name, (doc, add_args, _) in STEPS.items():
        sp = sub.add_parser(name, help=doc.splitlines()[0], description=doc, conflict_handler="resolve",
                            formatter_class=argparse.RawDescriptionHelpFormatter)
        add_args(sp)
    args = p.parse_args()
    STEPS[args.step][2](args)


if __name__ == "__main__":
    main()
