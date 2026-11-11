"""Shared paths, skeleton definitions and I/O helpers for the 2D pipeline.

All scripts are meant to be run from the repository root, e.g.
    python 2d/scripts/pose_dwpose.py --categories la_hiphop house
"""
import argparse
import glob
import json
import os
import subprocess

import numpy as np

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT_2D = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
RAW_VIDEO_DIR = os.path.join(ROOT_2D, "raw_video")
FRAMES_DIR = os.path.join(ROOT_2D, "frames")
KEYPOINTS_RAW_DIR = os.path.join(ROOT_2D, "keypoints_raw")   # per-tool output of 4.3
KEYPOINTS_DIR = os.path.join(ROOT_2D, "keypoints")           # cleaned output of 4.4
AUDIO_DIR = os.path.join(ROOT_2D, "audio")
FEATURES_DIR = os.path.join(ROOT_2D, "features")
CLIPS_DIR = os.path.join(ROOT_2D, "clips")
PREVIEW_DIR = os.path.join(ROOT_2D, "previews")
META_CSV = os.path.join(ROOT_2D, "meta.csv")
SPLITS_JSON = os.path.join(ROOT_2D, "splits.json")
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
    return os.path.join(KEYPOINTS_RAW_DIR, tool, category, video_id + ".npz")


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
    p.add_argument("--preview", type=int, default=0,
                   help=f"also write skeleton preview videos for the first N videos to 2d/previews/{tool}/")
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
        if n < args.preview:
            preview = PreviewWriter(os.path.join(PREVIEW_DIR, tool, cat, vid + ".mp4"),
                                    args.fps, w, h, f"{tool}  {cat}/{vid}")
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
