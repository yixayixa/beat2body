"""4.7 Dataset split by video (not by clip), stratified per category.

Writes 2d/splits.json  {"<category>/<video_id>": "train" | "val" | "test"}
and    2d/categories.json  {"<category>": index}  (fixed mapping for the embedding).

Default is 70 / 15 / 15 per category. With --by dancer, all videos of one
dancer go to the same split (stricter; AIST has 3 dancers per genre, so this
gives one dancer per split).

    python 2d/scripts/split_videos.py
    python 2d/scripts/split_videos.py --by dancer
"""
import argparse
import json
import os
import random
import sys
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import SPLITS_JSON, dancer_id, list_videos, write_categories_json  # noqa: E402

SPLITS = ["train", "val", "test"]


def allocate(n, ratios):
    """Counts per split that sum to n, each split >= 1 when n >= 3."""
    counts = [int(round(n * r)) for r in ratios]
    counts[0] = n - sum(counts[1:])
    if n >= 3:
        for i in (1, 2):
            if counts[i] == 0:
                counts[i], counts[0] = 1, counts[0] - 1
    return counts


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--categories", nargs="*", default=None)
    p.add_argument("--ratios", nargs=3, type=float, default=[0.7, 0.15, 0.15], metavar=("TRAIN", "VAL", "TEST"))
    p.add_argument("--by", choices=["video", "dancer"], default="video")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    rng = random.Random(args.seed)
    videos = list_videos(args.categories)
    by_cat = defaultdict(list)
    for cat, vid, _ in videos:
        by_cat[cat].append(vid)

    splits = {}
    for cat, vids in sorted(by_cat.items()):
        if args.by == "dancer":
            groups = defaultdict(list)
            for v in vids:
                groups[dancer_id(v) or v].append(v)
            units = [sorted(g) for _, g in sorted(groups.items())]
        else:
            units = [[v] for v in sorted(vids)]
        rng.shuffle(units)
        counts = allocate(len(units), args.ratios)
        i = 0
        for split, c in zip(SPLITS, counts):
            for unit in units[i:i + c]:
                for v in unit:
                    splits[f"{cat}/{v}"] = split
            i += c
        summary = {s: sum(1 for k, v in splits.items() if k.startswith(cat + "/") and v == s) for s in SPLITS}
        print(f"{cat}: {summary}")

    with open(SPLITS_JSON, "w") as f:
        json.dump(dict(sorted(splits.items())), f, indent=2)
    write_categories_json(by_cat.keys())
    print(f"wrote {SPLITS_JSON} ({len(splits)} videos)")


if __name__ == "__main__":
    main()
