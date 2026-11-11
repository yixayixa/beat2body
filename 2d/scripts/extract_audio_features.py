"""4.5 Audio extraction and features.

For every video:
    2d/audio/<category>/<video_id>.wav                  mono, 22.05 kHz
    2d/features/audio_mel/<category>/<video_id>.npy     (T, 64) log-mel spectrogram (dB)
    2d/features/audio_onset/<category>/<video_id>.npy   (T,)    onset strength
    2d/features/beats/<category>/<video_id>.npy         (num_beats,)  beat frame indices
    2d/features/onsets/<category>/<video_id>.npy        (num_onsets,) onset frame indices
    2d/features/audio_mfcc/<category>/<video_id>.npy    (T, 13) only with --mfcc

hop_length = sample_rate / fps (22050 / 30 = 735), so feature frame t lines up
with motion frame t from 4.3 (both use the same fixed fps).

Install:
    pip install librosa soundfile

    python 2d/scripts/extract_audio_features.py --categories la_hiphop house
"""
import argparse
import os
import subprocess
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import AUDIO_DIR, DEFAULT_FPS, FEATURES_DIR, list_videos  # noqa: E402


def extract_wav(video, wav, sr):
    os.makedirs(os.path.dirname(wav), exist_ok=True)
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", video, "-vn", "-ac", "1", "-ar", str(sr), wav], check=True)


def save(kind, cat, vid, arr):
    path = os.path.join(FEATURES_DIR, kind, cat, vid + ".npy")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.save(path, arr)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--categories", nargs="*", default=None)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--sr", type=int, default=22050)
    p.add_argument("--fps", type=float, default=DEFAULT_FPS)
    p.add_argument("--n-mels", type=int, default=64)
    p.add_argument("--n-fft", type=int, default=2048)
    p.add_argument("--mfcc", action="store_true", help="also save 13 MFCCs")
    p.add_argument("--overwrite", action="store_true")
    args = p.parse_args()

    import librosa

    hop = args.sr / args.fps
    if abs(hop - round(hop)) > 1e-6:
        sys.exit(f"sample rate {args.sr} / fps {args.fps} is not an integer hop length")
    hop = int(round(hop))

    for cat, vid, path in list_videos(args.categories, limit=args.limit):
        wav = os.path.join(AUDIO_DIR, cat, vid + ".wav")
        done = os.path.join(FEATURES_DIR, "beats", cat, vid + ".npy")
        if os.path.exists(done) and not args.overwrite:
            print("skip", f"{cat}/{vid}")
            continue
        extract_wav(path, wav, args.sr)
        y, sr = librosa.load(wav, sr=args.sr, mono=True)

        mel = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=args.n_fft, hop_length=hop, n_mels=args.n_mels)
        log_mel = librosa.power_to_db(mel, ref=np.max).T.astype(np.float32)          # (T, n_mels)
        onset_env = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop).astype(np.float32)  # (T,)
        tempo, beats = librosa.beat.beat_track(onset_envelope=onset_env, sr=sr, hop_length=hop)
        onsets = librosa.onset.onset_detect(onset_envelope=onset_env, sr=sr, hop_length=hop)

        save("audio_mel", cat, vid, log_mel)
        save("audio_onset", cat, vid, onset_env)
        save("onsets", cat, vid, onsets.astype(np.int32))
        if args.mfcc:
            mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13, n_fft=args.n_fft, hop_length=hop).T
            save("audio_mfcc", cat, vid, mfcc.astype(np.float32))
        save("beats", cat, vid, np.asarray(beats, dtype=np.int32))                   # written last = done marker
        bpm = float(np.atleast_1d(tempo)[0])
        print(f"{cat}/{vid}: {len(log_mel)} frames, {len(beats)} beats (~{bpm:.0f} BPM), {len(onsets)} onsets")


if __name__ == "__main__":
    main()
