"""4.1 Video collection: download the AIST Dance Video Database clips listed in 2d/data/aist_videos.csv.

The repository already contains these videos under 2d/raw_video/. This script
re-creates them from the public AIST URLs (e.g. on a fresh machine or to add
another category). Read and accept the AIST Dance DB Terms of Use first:
https://aistdancedb.ongaaccel.jp/terms_of_use/

Selection used for the list: refined 2 Mbps videos, advanced choreography (sFM),
front camera (c01), 21 clips per genre.

    python 2d/scripts/download_aist.py                       # all categories
    python 2d/scripts/download_aist.py --categories la_hiphop house
"""
import argparse
import csv
import os
import sys
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import RAW_VIDEO_DIR, ROOT_2D  # noqa: E402

LIST_CSV = os.path.join(ROOT_2D, "data", "aist_videos.csv")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--categories", nargs="*", default=None)
    p.add_argument("--list", default=LIST_CSV)
    args = p.parse_args()

    with open(args.list) as f:
        rows = [r for r in csv.DictReader(f) if not args.categories or r["category"] in args.categories]
    print(f"{len(rows)} videos")
    for r in rows:
        dst = os.path.join(RAW_VIDEO_DIR, r["category"], r["video_id"] + ".mp4")
        if os.path.exists(dst) and os.path.getsize(dst) > 0:
            continue
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        try:
            urllib.request.urlretrieve(r["url"], dst + ".part")
            os.replace(dst + ".part", dst)
            print("downloaded", dst)
        except Exception as e:  # keep going on single failures
            print("FAILED", r["url"], e)


if __name__ == "__main__":
    main()
