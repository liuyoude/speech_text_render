# -*- coding: utf-8 -*-
"""
Test Qwen3-ASR and Qwen3-ForcedAligner on example audios.
Covers plan tasks 1.3.2 (ASR transcription + WER) and 1.3.3 (forced alignment).
"""
import os
import re
import time
import torch
import tqdm
from qwen_asr import Qwen3ASRModel, Qwen3ForcedAligner
from core.audio_aligner.audio_aligner import plot_alignment, TimeSegment

ASR_MODEL = "Qwen/Qwen3-ASR-0.6B"
ALIGNER_MODEL = "Qwen/Qwen3-ForcedAligner-0.6B"
AUDIO_ROOT = "examples/audios"
DEVICE = "cuda:0"
DTYPE = torch.bfloat16


def get_audio_text_pairs(root_dir):
    pairs = []
    for dirpath, _, filenames in os.walk(root_dir):
        for f in sorted(filenames):
            if f.endswith(".wav"):
                wav = os.path.join(dirpath, f)
                txt = os.path.join(dirpath, f.replace(".wav", ".txt"))
                if os.path.exists(txt):
                    with open(txt, "r", encoding="utf-8") as fh:
                        text = fh.read().strip()
                    subdir = os.path.basename(os.path.normpath(dirpath))
                    pairs.append((wav, text, subdir, f))
    return pairs


def normalize_for_wer(text):
    text = text.lower().strip()
    text = re.sub(r'[^\w\u4e00-\u9fff\s]', '', text, flags=re.UNICODE)
    return text.split()


def compute_wer(ref_words, hyp_words):
    """Word Error Rate via edit distance."""
    n = len(ref_words)
    m = len(hyp_words)
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        dp[i][0] = i
    for j in range(m + 1):
        dp[0][j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = 0 if ref_words[i - 1] == hyp_words[j - 1] else 1
            dp[i][j] = min(dp[i - 1][j] + 1, dp[i][j - 1] + 1, dp[i - 1][j - 1] + cost)
    return dp[n][m] / max(n, 1)


def detect_language(subdir):
    if "en" in subdir and "zh" not in subdir:
        return "English"
    elif "zh" in subdir and "en" not in subdir:
        return "Chinese"
    return None


def test_qwen3_asr(pairs):
    print("=" * 60)
    print("Testing Qwen3-ASR transcription")
    print("=" * 60)

    save_dir = "examples/results/qwen3_asr"
    os.makedirs(save_dir, exist_ok=True)

    model = Qwen3ASRModel.from_pretrained(
        ASR_MODEL,
        dtype=DTYPE,
        device_map=DEVICE,
        max_new_tokens=256,
    )

    result_path = os.path.join(save_dir, "results.txt")
    total_wer = 0.0
    count = 0

    with open(result_path, "w", encoding="utf-8") as fout:
        for wav_path, ref_text, subdir, fname in tqdm.tqdm(pairs, desc="ASR"):
            start = time.time()
            results = model.transcribe(audio=wav_path, language=None)
            elapsed = time.time() - start

            hyp_text = results[0].text
            det_lang = results[0].language

            ref_words = normalize_for_wer(ref_text)
            hyp_words = normalize_for_wer(hyp_text)
            wer = compute_wer(ref_words, hyp_words)
            total_wer += wer
            count += 1

            fout.write(f"file: {wav_path}\n")
            fout.write(f"subdir: {subdir} | detected_lang: {det_lang}\n")
            fout.write(f"ref : {ref_text}\n")
            fout.write(f"hyp : {hyp_text}\n")
            fout.write(f"WER : {wer:.4f} | time: {elapsed:.2f}s\n\n")

        avg_wer = total_wer / max(count, 1)
        fout.write(f"{'=' * 40}\n")
        fout.write(f"Average WER: {avg_wer:.4f} ({count} files)\n")

    print(f"ASR results saved to {result_path}")
    print(f"Average WER: {avg_wer:.4f}")

    del model
    torch.cuda.empty_cache()
    return avg_wer


def test_qwen3_aligner(pairs):
    print("=" * 60)
    print("Testing Qwen3-ForcedAligner")
    print("=" * 60)

    save_dir = "examples/results/qwen3_aligner"
    os.makedirs(save_dir, exist_ok=True)

    aligner = Qwen3ForcedAligner.from_pretrained(
        ALIGNER_MODEL,
        dtype=DTYPE,
        device_map=DEVICE,
    )

    result_path = os.path.join(save_dir, "results.txt")
    total_time = 0.0

    with open(result_path, "w", encoding="utf-8") as fout:
        for wav_path, ref_text, subdir, fname in tqdm.tqdm(pairs, desc="Aligner"):
            lang = detect_language(subdir)
            if lang is None:
                fout.write(f"file: {wav_path} — SKIPPED (mixed lang)\n\n")
                continue

            start = time.time()
            results = aligner.align(audio=wav_path, text=ref_text, language=lang)
            elapsed = time.time() - start
            total_time += elapsed

            fout.write(f"file: {wav_path}\n")
            fout.write(f"subdir: {subdir} | language: {lang}\n")
            fout.write(f"text: {ref_text}\n")
            fout.write(f"time: {elapsed:.2f}s | items: {len(results[0])}\n")
            for item in results[0]:
                fout.write(
                    f"  [{item.start_time:.3f}-{item.end_time:.3f}] {item.text}\n"
                )
            fout.write("\n")

            segments = []
            for item in results[0]:
                segments.append(TimeSegment(
                    start=item.start_time,
                    end=item.end_time,
                    text=item.text,
                    type="word",
                ))
            img_name = f"{subdir}_{fname.replace('.wav', '.png')}"
            img_path = os.path.join(save_dir, img_name)
            try:
                plot_alignment(wav_path, segments, img_path)
            except Exception as e:
                fout.write(f"  plot error: {e}\n\n")

    print(f"Aligner results saved to {result_path}")

    del aligner
    torch.cuda.empty_cache()


if __name__ == "__main__":
    pairs = get_audio_text_pairs(AUDIO_ROOT)
    print(f"Found {len(pairs)} audio-text pairs\n")

    test_qwen3_asr(pairs)
    print()
    test_qwen3_aligner(pairs)
