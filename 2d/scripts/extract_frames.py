"""4.2 Frame extraction: dump JPEG frames at a fixed frame rate.

Optional. The pose scripts in 4.3 read videos directly at the same fixed frame
rate, so JPEGs are only needed for inspection or for external tools. Note that
1080p JPEGs for every video take a lot of disk space (roughly 20+ GB for all
210 AIST clips); use --categories / --limit / --height to keep it small.

    python 2d/scripts/extract_frames.py --categories la_hiphop --limit 1
    -> 2d/frames/la_hiphop/<video_id>/000001.jpg ...
"""
import argparse
import os
import subprocess
import sys

sys.dont_write_bytecode = True  # keep the repo free of __pycache__ folders
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import DEFAULT_FPS, FRAMES_DIR, list_videos  # noqa: E402


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--categories", nargs="*", default=None)
    p.add_argument("--limit", type=int, default=None, help="at most N videos per category")
    p.add_argument("--fps", type=float, default=DEFAULT_FPS)
    p.add_argument("--height", type=int, default=None, help="resize to this height (default: original)")
    p.add_argument("--quality", type=int, default=3, help="ffmpeg JPEG quality, 2 (best) to 31")
    args = p.parse_args()

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


if __name__ == "__main__":
    main()
