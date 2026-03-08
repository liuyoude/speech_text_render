# -*- coding: utf-8 -*-
"""
Population calibrator — scan datasets to compute per-feature mean/std
using Welford's online algorithm.  Outputs population_stats.json consumed
by downstream z-score extractors.

date: 20250306
author: liuyoude
"""
import json
import logging
import math
import os
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional

import librosa
import numpy as np
import pyloudnorm
import yaml

from core.audio_aligner.audio_aligner import SpeechTextAligner, TimeSegment
from core.utils.audio_utils import calculate_perceptual_energy

logger = logging.getLogger(__name__)

CLAUSE_FEATURES = ["speed", "energy", "loudness_lufs", "duration"]
WORD_FEATURES = ["word_duration", "word_energy"]
ALL_FEATURES = CLAUSE_FEATURES + WORD_FEATURES

PAUSE_THRESHOLD = 0.2
MIN_LUFS_DURATION = 0.4


@dataclass
class WelfordAccumulator:
    """Incremental mean/variance using Welford's online algorithm."""
    n: int = 0
    mean: float = 0.0
    m2: float = 0.0

    def update(self, value: float):
        self.n += 1
        delta = value - self.mean
        self.mean += delta / self.n
        delta2 = value - self.mean
        self.m2 += delta * delta2

    @property
    def std(self) -> float:
        if self.n < 2:
            return 0.0
        return math.sqrt(self.m2 / (self.n - 1))

    def to_dict(self) -> dict:
        return {"n": self.n, "mean": round(self.mean, 6), "std": round(self.std, 6)}

    @classmethod
    def from_dict(cls, d: dict) -> "WelfordAccumulator":
        acc = cls()
        acc.n = d.get("n", 0)
        acc.mean = d.get("mean", 0.0)
        std = d.get("std", 0.0)
        acc.m2 = (std ** 2) * max(acc.n - 1, 0)
        return acc


class PopulationCalibrator:
    """Scan audio datasets and accumulate population statistics for z-score normalisation."""

    def __init__(self, dataset_config_path: str, device: str = "cuda"):
        with open(dataset_config_path, "r", encoding="utf-8") as f:
            self.config = yaml.safe_load(f)
        self.device = device
        self._accumulators: Dict[str, Dict[str, WelfordAccumulator]] = {}
        self._aligner: Optional[SpeechTextAligner] = None

        self._energy_frame_ms = 30
        self._energy_hop_ms = 10

    @property
    def aligner(self) -> SpeechTextAligner:
        if self._aligner is None:
            self._aligner = SpeechTextAligner(device=self.device, time_fix=True)
        return self._aligner

    def _get_group(self, lang: str, style: str) -> Dict[str, WelfordAccumulator]:
        key = f"{lang}_{style}"
        if key not in self._accumulators:
            self._accumulators[key] = {feat: WelfordAccumulator() for feat in ALL_FEATURES}
        return self._accumulators[key]

    def calibrate(self) -> dict:
        """Main entry: scan all datasets listed in config."""
        datasets = self.config.get("datasets", [])
        for ds in datasets:
            name = ds["name"]
            path = ds["path"]
            lang = ds["lang"]
            style = ds["style"]
            text_ext = ds.get("text_ext", ".txt")
            logger.info("Calibrating dataset %s (%s)", name, path)

            wav_files = sorted(
                f for f in os.listdir(path) if f.lower().endswith(".wav")
            )
            for wav_name in wav_files:
                audio_path = os.path.join(path, wav_name)
                text_path = os.path.join(path, wav_name.rsplit(".", 1)[0] + text_ext)
                if not os.path.isfile(text_path):
                    logger.warning("Text file missing for %s, skipping", audio_path)
                    continue
                with open(text_path, "r", encoding="utf-8") as f:
                    text = f.read().strip()
                try:
                    self.calibrate_single(audio_path, text, lang, style)
                    logger.info("  [OK] %s", wav_name)
                except Exception:
                    logger.exception("  [FAIL] %s", wav_name)
        return self.get_stats()

    def calibrate_single(self, audio_path: str, text: str, lang: str, style: str):
        """Process one audio file: align, extract raw features, update accumulators."""
        segments = self.aligner.align(audio_path, text, lang=lang)
        y, sr = librosa.load(audio_path, sr=None, mono=True)

        frame_length = self._nearest_pow2(int(self._energy_frame_ms * sr / 1000))
        energy_hop = max(1, int(self._energy_hop_ms * sr / 1000))
        energy_frames = calculate_perceptual_energy(y, sr, frame_length, energy_hop)

        meter = pyloudnorm.Meter(sr, block_size=0.4)

        group = self._get_group(lang, style)
        self._extract_clause_features(segments, y, sr, energy_frames, energy_hop, meter, group)
        self._extract_word_features(segments, y, sr, energy_frames, energy_hop, group)

    def _extract_clause_features(
        self, segments, y, sr, energy_frames, energy_hop, meter, group
    ):
        clause_words: List[TimeSegment] = []
        for seg in segments:
            if seg.type == "word":
                clause_words.append(seg)
            elif seg.type == "clause":
                if not clause_words:
                    continue
                self._update_clause(clause_words, seg, y, sr, energy_frames, energy_hop, meter, group)
                clause_words = []
        if clause_words:
            self._update_clause_from_words(clause_words, y, sr, energy_frames, energy_hop, meter, group)

    def _update_clause(self, words, clause_seg, y, sr, energy_frames, energy_hop, meter, group):
        start, end = clause_seg.start, clause_seg.end
        duration = end - start
        if duration <= 0:
            return

        group["duration"].update(duration)

        eff_dur = self._effective_duration(words)
        if eff_dur > 0:
            speed = len(words) / eff_dur
            group["speed"].update(speed)

        e = self._segment_mean(energy_frames, sr, energy_hop, start, end)
        if e is not None:
            group["energy"].update(e)

        lufs = self._segment_lufs(y, sr, meter, start, end)
        if lufs is not None:
            group["loudness_lufs"].update(lufs)

    def _update_clause_from_words(self, words, y, sr, energy_frames, energy_hop, meter, group):
        """Fallback when no explicit clause segment follows the last group of words."""
        start = words[0].start
        end = words[-1].end
        dummy_clause = TimeSegment(start=start, end=end, text="", type="clause")
        self._update_clause(words, dummy_clause, y, sr, energy_frames, energy_hop, meter, group)

    def _extract_word_features(self, segments, y, sr, energy_frames, energy_hop, group):
        for seg in segments:
            if seg.type != "word":
                continue
            dur = seg.end - seg.start
            if dur <= 0:
                continue
            group["word_duration"].update(dur)

            e = self._segment_mean(energy_frames, sr, energy_hop, seg.start, seg.end)
            if e is not None:
                group["word_energy"].update(e)

    def _effective_duration(self, words: List[TimeSegment]) -> float:
        """Clause duration minus internal pauses longer than PAUSE_THRESHOLD."""
        if not words:
            return 0.0
        total = words[-1].end - words[0].start
        for i in range(1, len(words)):
            gap = words[i].start - words[i - 1].end
            if gap > PAUSE_THRESHOLD:
                total -= gap
        return max(total, 0.0)

    def _segment_mean(self, frames, sr, hop, start, end):
        rate = sr / hop
        s = int(start * rate)
        e = int(end * rate)
        seg = frames[s:e]
        if len(seg) == 0:
            return None
        return float(np.mean(seg))

    def _segment_lufs(self, y, sr, meter, start, end):
        s = int(start * sr)
        e = int(end * sr)
        seg = y[s:e]
        if len(seg) / sr < MIN_LUFS_DURATION:
            return None
        try:
            lufs = meter.integrated_loudness(seg)
            if np.isinf(lufs) or np.isnan(lufs):
                return None
            return float(lufs)
        except Exception:
            return None

    @staticmethod
    def _nearest_pow2(n):
        return 2 ** int(np.log2(n) + 0.5)

    def save(self, output_path: str):
        """Serialise current statistics to JSON."""
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(self.get_stats(), f, indent=2, ensure_ascii=False)
        logger.info("Saved population stats to %s", output_path)

    def load(self, stats_path: str):
        """Load previously saved stats for incremental update."""
        with open(stats_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        for group_key, features in data.items():
            if group_key not in self._accumulators:
                self._accumulators[group_key] = {}
            for feat_name, feat_data in features.items():
                self._accumulators[group_key][feat_name] = WelfordAccumulator.from_dict(feat_data)
        logger.info("Loaded population stats from %s", stats_path)

    def get_stats(self) -> dict:
        result = {}
        for group_key, accs in sorted(self._accumulators.items()):
            result[group_key] = {name: acc.to_dict() for name, acc in accs.items()}
        return result
