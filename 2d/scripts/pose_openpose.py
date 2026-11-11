"""4.3 Pose estimation with OpenPose.

Two backends:
  --backend official : the official CMU OpenPose (C++, everything on the GPU), BODY_25
                       model. It must be built first (see 2d/scripts/colab_build_openpose.sh on Colab).
  --backend pytorch  : (default) the PyTorch port described below; no build needed, but the
                       post-processing runs on the CPU and is slow on 1080p video.

PyTorch backend: OpenPose BODY-18, PyTorch port.

The official OpenPose is a C++/Caffe build that is slow to compile. This script
uses the PyTorch port of the same body model (from huggingface/controlnet_aux,
weights converted from the original CMU release). Its source files are fetched
at run time into ~/.cache/beat2body/ and the weights are downloaded from the
public Hugging Face repo lllyasviel/Annotators. No account or token is needed.

License note: OpenPose models and code are for academic / non-commercial
research use only (see the OpenPose license).

Install:
    pip install torch scipy opencv-python huggingface_hub

    python 2d/scripts/pose_openpose.py --categories la_hiphop house --preview 2

Output: 2d/keypoints/<category>/<video_id>_openpose.npz
    keypoints (T, 17, 3)  COCO-17 x, y (pixels), heatmap score   <- used by 4.4
    native    (T, 18, 3)  OpenPose BODY-18 (includes the neck joint)
Next step (4.4): python 2d/scripts/pipeline.py clean --tool openpose
"""
import importlib
import os
import sys
import urllib.request

import numpy as np

sys.dont_write_bytecode = True  # keep the repo free of __pycache__ folders
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pipeline import (CONF_THRESHOLD, OPENPOSE18_NAMES, OPENPOSE18_TO_COCO17, PreviewWriter,  # noqa: E402
                      iter_frames, pose_arg_parser, preview_path, probe_video, raw_keypoint_path,
                      run_pose_tool, save_raw_keypoints, select_videos, wants_preview)

TOOL = "openpose"
CACHE = os.path.expanduser("~/.cache/beat2body")
PKG = "openpose_body"
SRC_COMMIT = "b1aac318b15c4e65a21bc516dc236ad4bf593d46"
SRC_URL = ("https://raw.githubusercontent.com/huggingface/controlnet_aux/"
           f"{SRC_COMMIT}/src/controlnet_aux/open_pose/{{f}}")


def load_body_class(scales):
    """Fetch body.py / model.py / util.py once and import them as a small package."""
    pkg_dir = os.path.join(CACHE, PKG)
    os.makedirs(pkg_dir, exist_ok=True)
    for f in ["body.py", "model.py", "util.py"]:
        dst = os.path.join(pkg_dir, f)
        if not os.path.exists(dst):
            urllib.request.urlretrieve(SRC_URL.format(f=f), dst)
    open(os.path.join(pkg_dir, "__init__.py"), "w").close()
    body = os.path.join(pkg_dir, "body.py")
    src = open(body).read()
    # scipy.ndimage.filters is deprecated; inference scale is made configurable.
    src = src.replace("from scipy.ndimage.filters import gaussian_filter", "from scipy.ndimage import gaussian_filter")
    src = src.replace("scale_search = [0.5]", "scale_search = SCALES")
    if "SCALES = " not in src:
        src = src.replace("from . import util", "from . import util\n\nSCALES = [1.0]", 1)
    open(body, "w").write(src)
    sys.path.insert(0, CACHE)
    # util.py and body.py import each other; util must be imported first (as in controlnet_aux).
    importlib.import_module(f"{PKG}.util")
    mod = importlib.import_module(f"{PKG}.body")
    mod.SCALES = list(scales)
    return mod.Body


BODY25_NAMES = [
    "nose", "neck", "right_shoulder", "right_elbow", "right_wrist", "left_shoulder", "left_elbow",
    "left_wrist", "mid_hip", "right_hip", "right_knee", "right_ankle", "left_hip", "left_knee",
    "left_ankle", "right_eye", "left_eye", "right_ear", "left_ear", "left_big_toe", "left_small_toe",
    "left_heel", "right_big_toe", "right_small_toe", "right_heel",
]
BODY25_TO_COCO17 = [0, 16, 15, 18, 17, 5, 2, 6, 3, 7, 4, 12, 9, 13, 10, 14, 11]


def run_official(args):
    """Run the official CMU OpenPose binary (C++, GPU) with the BODY_25 model.

    Each video is first resampled to the fixed fps (same frames as the other tools),
    then openpose.bin writes one JSON file per frame, which is converted to the
    shared output format.
    """
    import glob
    import json
    import shutil
    import subprocess
    import tempfile
    import time

    root = os.path.abspath(args.openpose_dir)
    binary = os.path.join(root, "build", "examples", "openpose", "openpose.bin")
    if not os.path.exists(binary):
        sys.exit(f"official OpenPose not found at {binary}; build it first or use --backend pytorch")
    print(f"[{TOOL}] running the official OpenPose (BODY_25) from {root}")

    videos = select_videos(args)
    print(f"[{TOOL}] {len(videos)} videos")
    for n, (cat, vid, path) in enumerate(videos):
        out = raw_keypoint_path(TOOL, cat, vid)
        if os.path.exists(out) and not args.overwrite:
            print(f"[{TOOL}] skip {cat}/{vid} (exists)")
            continue
        w, h, _ = probe_video(path)
        if args.height:
            w, h = int(round(w * args.height / h / 2)) * 2, args.height
        tmp = tempfile.mkdtemp(prefix="openpose_")
        try:
            clip = os.path.join(tmp, "clip.mp4")
            vf = f"fps={args.fps}" + (f",scale={w}:{h}" if args.height else "")
            subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", path, "-vf", vf, "-an",
                            "-c:v", "libx264", "-preset", "veryfast", "-crf", "12", clip], check=True)
            json_dir = os.path.join(tmp, "json")
            t0 = time.perf_counter()
            subprocess.run([binary, "--video", clip, "--write_json", json_dir, "--display", "0",
                            "--render_pose", "0", "--model_pose", "BODY_25", "--number_people_max", "1",
                            "--net_resolution", args.net_resolution,
                            "--model_folder", os.path.join(root, "models", "")],
                           check=True, cwd=root, stdout=subprocess.DEVNULL)
            dt = time.perf_counter() - t0
            native = []
            for f in sorted(glob.glob(os.path.join(json_dir, "*_keypoints.json"))):
                people = json.load(open(f))["people"]
                if not people:
                    native.append(np.full((25, 3), np.nan, np.float32))
                    continue
                k = np.array(people[0]["pose_keypoints_2d"], np.float32).reshape(25, 3)
                k[k[:, 2] <= 0] = np.nan  # OpenPose writes 0, 0, 0 for joints it did not find
                native.append(k)
            native = np.stack(native)
            coco = native[:, BODY25_TO_COCO17]
            if wants_preview(args, n):
                preview = PreviewWriter(preview_path(TOOL, cat, vid), args.fps, w, h, f"{TOOL}  {cat}/{vid}")
                for i, frame in enumerate(iter_frames(clip, args.fps)):
                    if i < len(coco):
                        preview.write(frame, coco[i], CONF_THRESHOLD[TOOL], (255, 80, 80))
                preview.close()
        finally:
            shutil.rmtree(tmp, ignore_errors=True)
        save_raw_keypoints(out, coco, native, BODY25_NAMES, TOOL, args.fps, w, h, dt)
        found = 100 * (~np.isnan(coco[:, :, 0]).all(1)).mean()
        print(f"[{TOOL}] {cat}/{vid}: {len(coco)} frames, {len(coco) / dt:.1f} fps, person found in {found:.0f}% of frames")


def main():
    p = pose_arg_parser(TOOL, __doc__)
    p.add_argument("--backend", default="pytorch", choices=["pytorch", "official"],
                   help="pytorch: PyTorch port, network on the GPU and post-processing on the CPU; "
                        "official: the official CMU OpenPose binary (C++, GPU), BODY_25 model")
    p.add_argument("--openpose-dir", default=os.environ.get("OPENPOSE_DIR", "/content/openpose"),
                   help="official backend: OpenPose source folder with build/ and models/")
    p.add_argument("--net-resolution", default="-1x368", help="official backend: network input size")
    p.add_argument("--device", default="auto", help="pytorch backend: auto | cuda | cpu")
    p.add_argument("--scales", nargs="+", type=float, default=[1.0],
                   help="inference scales relative to a 368 px tall input (OpenPose default: 1.0; "
                        "multi-scale such as 0.5 1.0 1.5 2.0 is more accurate but slower)")
    args = p.parse_args()
    if args.backend == "official":
        run_official(args)
        return

    import torch
    from huggingface_hub import hf_hub_download

    device = ("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device
    Body = load_body_class(args.scales)
    model = Body(hf_hub_download("lllyasviel/Annotators", "body_pose_model.pth")).to(device)
    print(f"[{TOOL}] running on {device}, scales {args.scales}")

    def make_infer(fps):
        def infer(frame_bgr):
            candidate, subset = model(frame_bgr)  # expects BGR
            if len(subset) == 0:
                return None, None
            best = int(np.argmax(subset[:, 18]))  # person with the highest total score
            native = np.full((18, 3), np.nan, np.float32)
            for j, ci in enumerate(subset[best, :18].astype(int)):
                if ci != -1:
                    native[j] = candidate[ci, :3]
            return native[OPENPOSE18_TO_COCO17], native
        return infer

    run_pose_tool(TOOL, args, make_infer, OPENPOSE18_NAMES, color=(255, 80, 80))


if __name__ == "__main__":
    main()
