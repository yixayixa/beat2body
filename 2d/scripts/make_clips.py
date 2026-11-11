"""4.6 Alignment + 4.7 Dataset split: sliding-window clips and meta.csv.

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

Run after clean_keypoints.py (4.4) and extract_audio_features.py (4.5):

    python 2d/scripts/make_clips.py --window 240 --stride 60
"""
import argparse
import csv
import glob
import os
import random
import sys
from collections import defaultdict

import numpy as np

sys.dont_write_bytecode = True  # keep the repo free of __pycache__ folders
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import (CLIPS_DIR, META_CSV, ROOT_2D, clean_keypoint_path, dancer_id,  # noqa: E402
                    feature_path, list_videos, write_categories_json)

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


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
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
    args = p.parse_args()

    # Only videos that have both cleaned keypoints and audio features take part.
    videos, skipped = [], []
    for cat, vid, _ in list_videos(args.categories):
        need = [clean_keypoint_path(cat, vid)] + [
            feature_path(k, vid) for k in ("audio_mel", "audio_onset", "beats")]
        missing = [rel(x) for x in need if not os.path.exists(x)]
        if missing:
            skipped.append((f"{cat}/{vid}", "missing " + ", ".join(missing)))
        else:
            videos.append((cat, vid))
    if not videos:
        sys.exit("no video has both cleaned keypoints and audio features; run 4.4 and 4.5 first")

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


if __name__ == "__main__":
    main()
