"""4.6 Alignment and sliding-window clips (uses the split from 4.7).

Motion frame t and audio feature frame t already refer to the same time
(both are at the same fixed fps), so alignment only trims both to the shorter
length. Each video is then cut into overlapping windows.

    2d/clips/<split>/<category>__<video_id>__<start>.npz
        keypoints (W, 17, 2), valid (W, 17), mel (W, 64), onset (W,),
        beats (beat frame indices inside the window), category, category_index,
        video_id, start, end, fps
    2d/meta.csv
        clip_path, audio_feat_path, category, start_frame, end_frame, split, video_id, dancer

Run after clean_keypoints.py (4.4), extract_audio_features.py (4.5) and
split_videos.py (4.7):

    python 2d/scripts/make_clips.py --window 240 --stride 60
"""
import argparse
import csv
import glob
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import (CATEGORIES_JSON, CLIPS_DIR, FEATURES_DIR, KEYPOINTS_DIR, META_CSV,  # noqa: E402
                    ROOT_2D, SPLITS_JSON, dancer_id)

REPO_ROOT = os.path.dirname(ROOT_2D)


def rel(path):
    return os.path.relpath(path, REPO_ROOT)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--window", type=int, default=240, help="clip length in frames (README: 120-300)")
    p.add_argument("--stride", type=int, default=60)
    p.add_argument("--min-length", type=int, default=120, help="videos shorter than --window but at least "
                   "this long become a single shorter clip")
    p.add_argument("--min-valid", type=float, default=0.5,
                   help="drop clips where less than this fraction of joints was observed")
    args = p.parse_args()

    if not os.path.exists(SPLITS_JSON):
        sys.exit("run split_videos.py first")
    splits = json.load(open(SPLITS_JSON))
    cat_index = json.load(open(CATEGORIES_JSON))

    for old in glob.glob(os.path.join(CLIPS_DIR, "*", "*.npz")):
        os.remove(old)

    rows, skipped = [], []
    for key, split in splits.items():
        cat, vid = key.split("/")
        kp_path = os.path.join(KEYPOINTS_DIR, cat, vid + ".npz")
        mel_path = os.path.join(FEATURES_DIR, "audio_mel", cat, vid + ".npy")
        onset_path = os.path.join(FEATURES_DIR, "audio_onset", cat, vid + ".npy")
        beats_path = os.path.join(FEATURES_DIR, "beats", cat, vid + ".npy")
        missing = [x for x in (kp_path, mel_path, onset_path, beats_path) if not os.path.exists(x)]
        if missing:
            skipped.append((key, "missing " + ", ".join(rel(m) for m in missing)))
            continue
        kp = np.load(kp_path)
        mel, onset, beats = np.load(mel_path), np.load(onset_path), np.load(beats_path)
        T = min(len(kp["keypoints"]), len(mel), len(onset))
        if abs(len(kp["keypoints"]) - len(mel)) > 3:
            print(f"warning {key}: motion {len(kp['keypoints'])} vs audio {len(mel)} frames")
        if T < args.min_length:
            skipped.append((key, f"too short ({T} frames)"))
            continue
        win = min(args.window, T)
        starts = list(range(0, T - win + 1, args.stride))
        for s in starts:
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

    for s in ("train", "val", "test"):
        print(f"{s}: {sum(r['split'] == s for r in rows)} clips")
    if skipped:
        print(f"skipped {len(skipped)} videos (keypoints or audio features not found), e.g.:")
        for key, why in skipped[:5]:
            print(f"  {key}: {why}")
    print(f"wrote {rel(META_CSV)}")


if __name__ == "__main__":
    main()
