"""4.3 Pose estimation with OpenPose (BODY-18, PyTorch port).

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

Output: 2d/keypoints_raw/openpose/<category>/<video_id>.npz
    keypoints (T, 17, 3)  COCO-17 x, y (pixels), heatmap score   <- used by 4.4
    native    (T, 18, 3)  OpenPose BODY-18 (includes the neck joint)
"""
import importlib
import os
import sys
import urllib.request

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import OPENPOSE18_NAMES, OPENPOSE18_TO_COCO17, pose_arg_parser, run_pose_tool  # noqa: E402

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
    mod = importlib.import_module(f"{PKG}.body")
    mod.SCALES = list(scales)
    return mod.Body


def main():
    p = pose_arg_parser(TOOL, __doc__)
    p.add_argument("--device", default="auto", help="auto | cuda | cpu")
    p.add_argument("--scales", nargs="+", type=float, default=[1.0],
                   help="inference scales relative to a 368 px tall input (OpenPose default: 1.0; "
                        "multi-scale such as 0.5 1.0 1.5 2.0 is more accurate but slower)")
    args = p.parse_args()

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
