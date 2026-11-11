"""4.3 Pose estimation with DWPose (whole-body, ONNX Runtime).

Install (GPU, e.g. Colab):
    pip install onnxruntime-gpu huggingface_hub opencv-python
    pip install --no-deps easy-dwpose
CPU only: use `onnxruntime` instead of `onnxruntime-gpu` (much slower).

The two ONNX checkpoints (YOLOX person detector + DWPose-L) are downloaded
automatically from the public Hugging Face repo RedHash/DWPose on first run.
No account or token is needed.

    python 2d/scripts/pose_dwpose.py --categories la_hiphop house --preview 2

Output: 2d/keypoints/<category>/<video_id>_dwpose.npz
    keypoints (T, 17, 3)   COCO-17 x, y (pixels), score   <- used by 4.4
    native    (T, 134, 3)  OpenPose-18 body, 6 feet, 68 face, 21 + 21 hand points
Next step (4.4): python 2d/scripts/pipeline.py clean --tool dwpose
"""
import os
import sys

import numpy as np

sys.dont_write_bytecode = True  # keep the repo free of __pycache__ folders
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pipeline import OPENPOSE18_NAMES, OPENPOSE18_TO_COCO17, pose_arg_parser, run_pose_tool  # noqa: E402

TOOL = "dwpose"
NATIVE_NAMES = (
    OPENPOSE18_NAMES
    + ["left_big_toe", "left_small_toe", "left_heel", "right_big_toe", "right_small_toe", "right_heel"]
    + [f"face_{i}" for i in range(68)]
    + [f"left_hand_{i}" for i in range(21)]
    + [f"right_hand_{i}" for i in range(21)]
)


def build_model(device):
    import onnxruntime as ort
    from easy_dwpose.body_estimation import Wholebody
    from huggingface_hub import hf_hub_download

    if device == "auto":
        device = "cuda" if "CUDAExecutionProvider" in ort.get_available_providers() else "cpu"
    if device.startswith("cuda"):
        # Load the CUDA / cuDNN libraries before creating the sessions; without this
        # onnxruntime can silently fall back to the CPU.
        try:
            import torch  # noqa: F401  (its wheels ship the CUDA / cuDNN libraries)
        except ImportError:
            pass
        if hasattr(ort, "preload_dlls"):
            ort.preload_dlls()
    det = hf_hub_download("RedHash/DWPose", "yolox_l.onnx")
    pose = hf_hub_download("RedHash/DWPose", "dw-ll_ucoco_384.onnx")
    model = Wholebody(model_det=det, model_pose=pose, device=device)
    used = model.session_pose.get_providers()
    actual = "cuda" if "CUDAExecutionProvider" in used else "cpu"
    print(f"[{TOOL}] running on {actual} (onnxruntime providers: {', '.join(used)})")
    if device.startswith("cuda") and actual == "cpu":
        print(f"[{TOOL}] warning: CUDA was requested but is not available in onnxruntime; using the CPU")
    return model


def main():
    p = pose_arg_parser(TOOL, __doc__)
    p.add_argument("--device", default="auto", help="auto | cuda | cuda:N | cpu")
    args = p.parse_args()
    model = build_model(args.device)
    import cv2

    def make_infer(fps):
        def infer(frame_bgr):
            kp, sc = model(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))  # (people, 134, 2), (people, 134)
            if kp is None or len(kp) == 0:
                return None, None
            best = int(np.argmax(sc[:, :18].mean(1)))  # most confident person
            native = np.concatenate([kp[best], sc[best, :, None]], -1).astype(np.float32)
            return native[:18][OPENPOSE18_TO_COCO17], native
        return infer

    run_pose_tool(TOOL, args, make_infer, NATIVE_NAMES, color=(0, 255, 0))


if __name__ == "__main__":
    main()
