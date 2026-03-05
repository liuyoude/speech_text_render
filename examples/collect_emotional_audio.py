# -*- coding: utf-8 -*-
"""
Download emotional speech audio samples from ESD dataset for control tag validation.

Source: ESD (Emotional Speech Dataset) demo samples on GitHub
  Repo: https://github.com/HLTSingapore/ESD
  License: MIT (research use)

ESD utterance ID → emotion mapping (350 utterances per emotion):
  000001-000350: Neutral
  000351-000700: Angry
  000701-001050: Happy
  001051-001400: Sad
  001401-001750: Surprise

Usage:
  conda activate speech_text_render
  python examples/collect_emotional_audio.py [--skip-asr]
"""

import os
import sys
import argparse
import time
from pathlib import Path

import requests

from core.audio_aligner import WhisperAligner

BASE_URL = "https://raw.githubusercontent.com/HLTSingapore/ESD/main/audio"

DOWNLOADS = [
    # NOTE: ESD repo demo IDs are counter-intuitive — speakers 0003/0007 speak
    # Mandarin while 0013/0016 speak English (verified via Whisper language detection).
    #
    # --- Chinese (Mandarin) emotional audio, speakers 0003 / 0007 ---
    {"file": "0003_000708.wav", "name": "zh_happy_001",     "emotion": "happy",     "lang": "zh", "speaker": "0003", "dir": "zh_emotion"},
    {"file": "0003_000358.wav", "name": "zh_angry_001",     "emotion": "angry",     "lang": "zh", "speaker": "0003", "dir": "zh_emotion"},
    {"file": "0003_001058.wav", "name": "zh_sad_001",       "emotion": "sad",       "lang": "zh", "speaker": "0003", "dir": "zh_emotion"},
    {"file": "0003_001408.wav", "name": "zh_surprised_001", "emotion": "surprised", "lang": "zh", "speaker": "0003", "dir": "zh_emotion"},
    {"file": "0007_000358.wav", "name": "zh_angry_002",     "emotion": "angry",     "lang": "zh", "speaker": "0007", "dir": "zh_emotion"},
    # --- English emotional audio, speakers 0013 / 0016 ---
    {"file": "0013_000701.wav", "name": "en_happy_001",     "emotion": "happy",     "lang": "en", "speaker": "0013", "dir": "en_emotion"},
    {"file": "0013_000351.wav", "name": "en_angry_001",     "emotion": "angry",     "lang": "en", "speaker": "0013", "dir": "en_emotion"},
    {"file": "0013_001051.wav", "name": "en_sad_001",       "emotion": "sad",       "lang": "en", "speaker": "0013", "dir": "en_emotion"},
    {"file": "0013_001401.wav", "name": "en_surprised_001", "emotion": "surprised", "lang": "en", "speaker": "0013", "dir": "en_emotion"},
    {"file": "0016_000701.wav", "name": "en_happy_002",     "emotion": "happy",     "lang": "en", "speaker": "0016", "dir": "en_emotion"},
]

AUDIOS_ROOT = Path(__file__).resolve().parent / "audios"


def download_file(url: str, save_path: Path, retries: int = 3) -> bool:
    if save_path.exists():
        print(f"  [skip] already exists: {save_path.name}")
        return True
    for attempt in range(1, retries + 1):
        try:
            resp = requests.get(url, timeout=30)
            resp.raise_for_status()
            save_path.write_bytes(resp.content)
            size_kb = len(resp.content) / 1024
            print(f"  [done] {save_path.name}  ({size_kb:.1f} KB)")
            return True
        except Exception as e:
            print(f"  [fail] attempt {attempt}/{retries}: {e}")
            if attempt < retries:
                time.sleep(2 * attempt)
    return False


def transcribe_with_whisper_aligner(aligner: "WhisperAligner", wav_path: str, lang: str) -> str:
    """
    Use the project's WhisperAligner model with explicit language.
    Same transcribe params as WhisperAligner.align() but with language pinned.
    """
    import whisper as _whisper
    audio = _whisper.load_audio(str(wav_path))
    result = aligner.model.transcribe(
        audio,
        language=lang,
        word_timestamps=True,
        temperature=0.0,
        condition_on_previous_text=False,
        fp16=True if aligner.device.type == "cuda" else False,
        hallucination_silence_threshold=0.1,
        beam_size=10,
    )
    return result["text"].strip()


def main():
    parser = argparse.ArgumentParser(description="Download ESD emotional audio samples")
    parser.add_argument("--skip-asr", action="store_true",
                        help="Skip ASR transcript generation, create placeholder .txt files instead")
    parser.add_argument("--device", default="cuda",
                        help="Device for Whisper inference (default: cuda)")
    parser.add_argument("--model-size", default="small",
                        help="Whisper model size (default: small, same as SpeechTextAligner)")
    parser.add_argument("--force-asr", action="store_true",
                        help="Overwrite existing .txt files with new ASR results")
    args = parser.parse_args()

    print("=" * 60)
    print("Collecting emotional speech audio from ESD dataset")
    print(f"Target: {AUDIOS_ROOT}")
    print("=" * 60)

    dirs_needed = {AUDIOS_ROOT / item["dir"] for item in DOWNLOADS}
    for d in dirs_needed:
        d.mkdir(parents=True, exist_ok=True)

    # --- Phase 1: Download ---
    print("\n[1/2] Downloading audio files ...\n")
    downloaded = []
    for item in DOWNLOADS:
        url = f"{BASE_URL}/{item['file']}"
        wav_path = AUDIOS_ROOT / item["dir"] / f"{item['name']}.wav"
        print(f"  {item['name']}  emotion={item['emotion']:<12s} lang={item['lang']}  speaker={item['speaker']}")
        if download_file(url, wav_path):
            downloaded.append((item, wav_path))

    if not downloaded:
        print("\nNo files downloaded. Check network connectivity.")
        sys.exit(1)

    # --- Phase 2: Transcripts ---
    print(f"\n[2/2] Generating transcripts ({len(downloaded)} files) ...\n")

    aligner = None
    if not args.skip_asr:
        try:
            print(f"  Loading WhisperAligner (model={args.model_size}, device={args.device}) ...")
            aligner = WhisperAligner(model_size=args.model_size, device=args.device)
            print("  WhisperAligner ready.\n")
        except Exception as e:
            print(f"  [warn] Could not load WhisperAligner ({e})")
            print("  Falling back to placeholder transcripts.\n")

    for item, wav_path in downloaded:
        txt_path = wav_path.with_suffix(".txt")
        if txt_path.exists() and not args.force_asr:
            print(f"  [skip] {txt_path.name} already exists (use --force-asr to overwrite)")
            continue

        transcript = None
        if aligner:
            try:
                transcript = transcribe_with_whisper_aligner(aligner, wav_path, item["lang"])
            except Exception as e:
                print(f"  [warn] ASR error for {wav_path.name}: {e}")

        if transcript:
            txt_path.write_text(transcript, encoding="utf-8")
            print(f"  [asr]  {txt_path.name}: {transcript}")
        else:
            placeholder = f"[ESD {item['emotion']} speaker={item['speaker']}]"
            txt_path.write_text(placeholder, encoding="utf-8")
            print(f"  [placeholder] {txt_path.name}: {placeholder}")

    # --- Summary ---
    print("\n" + "=" * 60)
    print(f"Result: {len(downloaded)}/{len(DOWNLOADS)} audio files downloaded\n")
    for item, wav_path in downloaded:
        txt_path = wav_path.with_suffix(".txt")
        txt_mark = "T" if txt_path.exists() else "-"
        print(f"  [{txt_mark}] {item['dir']}/{item['name']}.wav  "
              f"emotion={item['emotion']:<12s} lang={item['lang']}")
    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()
