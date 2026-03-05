# -*- coding: utf-8 -*-
"""
Compare Qwen3 aligner vs Whisper/MFA aligner on all example audios.
Both backends are run on every file. Since no ground-truth timestamps exist,
the two backends are cross-validated against each other.

Metrics:
  - Word-level timestamp deviation (mean/median/max of |start_diff| and |end_diff|)
  - Sentence count agreement
  - Processing speed
"""
import os
import time
import statistics
import torch
import tqdm
from core.audio_aligner import SpeechTextAligner, plot_alignment


AUDIO_ROOT = "examples/audios"
SAVE_DIR = "examples/results/comparison"


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


def detect_language(subdir):
    if "en" in subdir and "zh" not in subdir:
        return "en"
    elif "zh" in subdir and "en" not in subdir:
        return "zh"
    return None


def run_backend(aligner, wav_path, text, lang):
    start = time.time()
    try:
        segments = aligner.align(wav_path, text, lang)
        elapsed = time.time() - start
        return segments, elapsed, None
    except Exception as e:
        elapsed = time.time() - start
        return [], elapsed, str(e)


def format_segments(segments):
    lines = []
    for seg in segments:
        lines.append(f"  [{seg.start:.3f}-{seg.end:.3f}] {seg.text} ({seg.type})")
    return "\n".join(lines)


def compute_word_deviations(q_words, w_words):
    """Align word segments by index and compute timestamp deviations."""
    n = min(len(q_words), len(w_words))
    if n == 0:
        return None

    start_diffs = []
    end_diffs = []
    details = []
    for i in range(n):
        sd = abs(q_words[i].start - w_words[i].start)
        ed = abs(q_words[i].end - w_words[i].end)
        start_diffs.append(sd)
        end_diffs.append(ed)
        details.append({
            "idx": i,
            "q_text": q_words[i].text,
            "w_text": w_words[i].text,
            "q_start": q_words[i].start, "q_end": q_words[i].end,
            "w_start": w_words[i].start, "w_end": w_words[i].end,
            "start_diff": sd, "end_diff": ed,
        })

    return {
        "matched_words": n,
        "q_total": len(q_words),
        "w_total": len(w_words),
        "start_mean": statistics.mean(start_diffs),
        "start_median": statistics.median(start_diffs),
        "start_max": max(start_diffs),
        "end_mean": statistics.mean(end_diffs),
        "end_median": statistics.median(end_diffs),
        "end_max": max(end_diffs),
        "details": details,
    }


def main():
    os.makedirs(SAVE_DIR, exist_ok=True)
    pairs = get_audio_text_pairs(AUDIO_ROOT)
    print(f"Found {len(pairs)} audio-text pairs\n")

    print("Loading Qwen3 backend...")
    qwen3_aligner = SpeechTextAligner(device="cuda:0", aligner_backend="qwen3")
    print("Loading Whisper/MFA backend...")
    wmfa_aligner = SpeechTextAligner(device="cuda:0", aligner_backend="whisper_mfa")

    # Warm up both backends
    lang0 = detect_language(pairs[0][2])
    if lang0:
        qwen3_aligner.align(pairs[0][0], pairs[0][1], lang0)
        wmfa_aligner.align(pairs[0][0], pairs[0][1], lang0)

    report_path = os.path.join(SAVE_DIR, "comparison_report.txt")

    qwen3_times, wmfa_times = [], []
    all_start_diffs, all_end_diffs = [], []
    per_file_stats = []

    with open(report_path, "w", encoding="utf-8") as fout:
        fout.write("=" * 80 + "\n")
        fout.write("  Qwen3 vs Whisper/MFA  —  Alignment Comparison Report\n")
        fout.write("=" * 80 + "\n\n")

        for wav_path, text, subdir, fname in tqdm.tqdm(pairs, desc="Comparing"):
            lang = detect_language(subdir)
            file_key = f"{subdir}_{fname.replace('.wav', '')}"

            fout.write(f"{'─' * 80}\n")
            fout.write(f"[{file_key}]  lang={lang}  file={wav_path}\n")
            fout.write(f"text: {text}\n\n")

            # ── Qwen3 ──
            q_segs, q_time, q_err = run_backend(qwen3_aligner, wav_path, text, lang)
            q_words = [s for s in q_segs if s.type == "word"]
            q_sents = [s for s in q_segs if s.type == "sentence"]
            qwen3_times.append(q_time)

            fout.write(f"  [Qwen3]       time={q_time:.3f}s  words={len(q_words)}  sentences={len(q_sents)}\n")
            if q_err:
                fout.write(f"    ERROR: {q_err}\n")
            else:
                fout.write(format_segments(q_segs) + "\n")

            # ── Whisper/MFA ──
            w_segs, w_time, w_err = run_backend(wmfa_aligner, wav_path, text, lang)
            w_words = [s for s in w_segs if s.type == "word"]
            w_sents = [s for s in w_segs if s.type == "sentence"]
            wmfa_times.append(w_time)

            fout.write(f"\n  [Whisper/MFA] time={w_time:.3f}s  words={len(w_words)}  sentences={len(w_sents)}\n")
            if w_err:
                fout.write(f"    ERROR: {w_err}\n")
            else:
                fout.write(format_segments(w_segs) + "\n")

            # ── Cross-validation ──
            fout.write(f"\n  [Cross-Validation]\n")
            if q_err or w_err:
                fout.write(f"    Skipped (one backend had errors)\n")
            else:
                dev = compute_word_deviations(q_words, w_words)
                if dev is None:
                    fout.write(f"    Skipped (no word segments to compare)\n")
                else:
                    fout.write(f"    word count: Qwen3={dev['q_total']}  Whisper/MFA={dev['w_total']}  matched={dev['matched_words']}\n")
                    fout.write(f"    start_time deviation:  mean={dev['start_mean']:.3f}s  median={dev['start_median']:.3f}s  max={dev['start_max']:.3f}s\n")
                    fout.write(f"    end_time   deviation:  mean={dev['end_mean']:.3f}s  median={dev['end_median']:.3f}s  max={dev['end_max']:.3f}s\n")

                    # Show top-3 largest deviations
                    sorted_by_dev = sorted(dev["details"], key=lambda d: d["start_diff"] + d["end_diff"], reverse=True)
                    fout.write(f"    largest deviations (top-3):\n")
                    for d in sorted_by_dev[:3]:
                        fout.write(f"      #{d['idx']} Q='{d['q_text']}' W='{d['w_text']}' "
                                   f"start_diff={d['start_diff']:.3f}s end_diff={d['end_diff']:.3f}s\n")
                        fout.write(f"        Q:[{d['q_start']:.3f}-{d['q_end']:.3f}]  W:[{d['w_start']:.3f}-{d['w_end']:.3f}]\n")

                    for d in dev["details"]:
                        all_start_diffs.append(d["start_diff"])
                        all_end_diffs.append(d["end_diff"])

                    per_file_stats.append({
                        "file": file_key,
                        "lang": lang,
                        "start_mean": dev["start_mean"],
                        "end_mean": dev["end_mean"],
                        "q_words": dev["q_total"],
                        "w_words": dev["w_total"],
                    })

            fout.write("\n")

            # Save alignment plots for both backends
            for backend_name, segs in [("qwen3", q_segs), ("whisper_mfa", w_segs)]:
                img_dir = os.path.join(SAVE_DIR, backend_name)
                os.makedirs(img_dir, exist_ok=True)
                if segs:
                    try:
                        plot_alignment(wav_path, segs,
                                       os.path.join(img_dir, f"{file_key}.png"))
                    except Exception:
                        pass

        # ══════════════ SUMMARY ══════════════
        fout.write("\n" + "=" * 80 + "\n")
        fout.write("  SUMMARY\n")
        fout.write("=" * 80 + "\n\n")

        avg_q = statistics.mean(qwen3_times) if qwen3_times else 0
        avg_w = statistics.mean(wmfa_times) if wmfa_times else 0
        fout.write(f"Processing Speed ({len(pairs)} files):\n")
        fout.write(f"  Qwen3       : avg={avg_q:.3f}s  total={sum(qwen3_times):.1f}s\n")
        fout.write(f"  Whisper/MFA : avg={avg_w:.3f}s  total={sum(wmfa_times):.1f}s\n")
        if avg_w > 0:
            fout.write(f"  Speedup     : {avg_w / avg_q:.1f}x (Qwen3 is {'faster' if avg_q < avg_w else 'slower'})\n")
        fout.write("\n")

        if all_start_diffs:
            fout.write(f"Overall Word Timestamp Deviation ({len(all_start_diffs)} matched words across {len(per_file_stats)} files):\n")
            fout.write(f"  start_time: mean={statistics.mean(all_start_diffs):.3f}s  median={statistics.median(all_start_diffs):.3f}s  max={max(all_start_diffs):.3f}s\n")
            fout.write(f"  end_time  : mean={statistics.mean(all_end_diffs):.3f}s  median={statistics.median(all_end_diffs):.3f}s  max={max(all_end_diffs):.3f}s\n")
            combined = [(sd + ed) / 2 for sd, ed in zip(all_start_diffs, all_end_diffs)]
            fout.write(f"  combined  : mean={statistics.mean(combined):.3f}s  median={statistics.median(combined):.3f}s\n")

            within_50ms = sum(1 for c in combined if c <= 0.05) / len(combined) * 100
            within_100ms = sum(1 for c in combined if c <= 0.10) / len(combined) * 100
            within_200ms = sum(1 for c in combined if c <= 0.20) / len(combined) * 100
            fout.write(f"\n  Agreement rate:\n")
            fout.write(f"    <=50ms : {within_50ms:.1f}%\n")
            fout.write(f"    <=100ms: {within_100ms:.1f}%\n")
            fout.write(f"    <=200ms: {within_200ms:.1f}%\n")
        else:
            fout.write("No word-level deviations could be computed.\n")

        fout.write(f"\nPer-file Summary:\n")
        fout.write(f"  {'File':<40} {'Lang':<6} {'Q_words':<8} {'W_words':<8} {'Start_dev':<10} {'End_dev':<10}\n")
        fout.write(f"  {'─' * 82}\n")
        for ps in per_file_stats:
            fout.write(f"  {ps['file']:<40} {str(ps['lang']):<6} {ps['q_words']:<8} {ps['w_words']:<8} "
                       f"{ps['start_mean']:.3f}s    {ps['end_mean']:.3f}s\n")

        fout.write(f"\n{'=' * 80}\n")

    print(f"\nReport saved to {report_path}")
    print(f"Qwen3 avg: {avg_q:.3f}s  |  Whisper/MFA avg: {avg_w:.3f}s")
    if all_start_diffs:
        print(f"Overall deviation: start={statistics.mean(all_start_diffs):.3f}s  end={statistics.mean(all_end_diffs):.3f}s")


if __name__ == "__main__":
    main()
