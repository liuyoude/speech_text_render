# -*- coding: utf-8 -*-
"""
base class for all feature extractors
date: 20250417
author: liuyoude
"""
import json
import logging
import os
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple

import numpy as np

from core.audio_aligner.audio_aligner import TimeSegment

logger = logging.getLogger(__name__)

_DEFAULT_STATS_PATH = os.path.join(
    os.path.dirname(os.path.dirname(__file__)),
    "default_population_stats.json",
)

_DEFAULT_LABELS = ("xlow", "low", "normal", "high", "xhigh")

_MIN_VOICED_FRAMES = 5
_MIN_SPEAKER_STD_HZ = 10.0


class BaseExtractor(ABC):
    def __init__(self, config: Dict):
        self.config = config
        stats_path = config.get("population_stats_path", _DEFAULT_STATS_PATH)
        self._population_stats = self._load_population_stats(stats_path)
        self._style = config.get("style", "read")
        self._number_control = config.get("number_control", False)
        self._shared_context: Dict = {}

    # ------------------------------------------------------------------
    # population stats & z-score helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _load_population_stats(path: str) -> Optional[Dict]:
        if not os.path.isfile(path):
            logger.warning("Population stats file not found: %s", path)
            return None
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        data.pop("_comment", None)
        return data

    def _get_lang_style(self, lang: str) -> str:
        lang_prefix = "zh" if lang in ("zh", "chinese") else "en"
        return f"{lang_prefix}_{self._style}"

    _Z_CLAMP = 3.0  # output z-scores clamped to [-3, 3]

    def _zscore(self, value: float, feature_name: str, lang_style: str) -> Optional[float]:
        """Compute population z-score: (value - mean) / std."""
        if self._population_stats is None:
            return None
        group = self._population_stats.get(lang_style)
        if group is None:
            return None
        feat = group.get(feature_name)
        if feat is None:
            return None
        std = feat.get("std", 0.0)
        if std <= 0:
            return None
        mean = feat.get("mean", 0.0)
        return (value - mean) / std

    @staticmethod
    def _should_annotate(z: float, threshold: float = 0.5) -> bool:
        """Return True when |z| exceeds *threshold* (default 0.5)."""
        return abs(z) > threshold

    @classmethod
    def _clamp_z(cls, z: float) -> float:
        """Clamp z-score to [-_Z_CLAMP, _Z_CLAMP] for output."""
        return max(-cls._Z_CLAMP, min(cls._Z_CLAMP, z))

    def _format_z(self, z: float, labels: Tuple[str, ...] = None):
        """Return clamped z-score (number_control=True) or label (False)."""
        if self._number_control:
            return round(self._clamp_z(z), 2)
        return self._z_to_label(z, labels)

    @staticmethod
    def _z_to_label(z: float, labels: Tuple[str, ...] = None) -> str:
        """Map z-score to a discrete 5-level label.

        labels is a 5-tuple: (very_low, low, normal, high, very_high).
        Subclass extractors pass their own names, e.g.
          SpeedExtractor:  ("xslow",  "slow", "normal", "fast", "xfast")
          VolumeExtractor: ("whisper", "soft", "normal", "loud", "shout")
        """
        if labels is None:
            labels = _DEFAULT_LABELS
        if z < -1.5:
            return labels[0]
        if z < -0.5:
            return labels[1]
        if z <= 0.5:
            return labels[2]
        if z <= 1.5:
            return labels[3]
        return labels[4]

    # ------------------------------------------------------------------
    # segment grouping helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _group_by_clause(segments: List[TimeSegment]):
        """Group word segments by their enclosing clause.

        Returns a list of ``(clause_segment, [word_indices])`` tuples.
        """
        clauses = []
        current_word_indices: List[int] = []
        for idx, seg in enumerate(segments):
            if seg.type == "word":
                current_word_indices.append(idx)
            elif seg.type == "clause":
                clauses.append((seg, current_word_indices))
                current_word_indices = []
        return clauses

    # ------------------------------------------------------------------
    # shared feature caching helpers
    # ------------------------------------------------------------------

    def _load_audio_shared(self, audio_path: str) -> Tuple[np.ndarray, int]:
        """Load audio waveform, reusing shared context cache."""
        key = f"audio:{audio_path}"
        if key in self._shared_context:
            return self._shared_context[key]
        import librosa
        y, sr = librosa.load(audio_path, sr=None, mono=True)
        self._shared_context[key] = (y, sr)
        return y, sr

    def _get_f0_shared(
        self, audio_path: str, device: str = "cpu", hop_ms: float = 10,
    ) -> Tuple[np.ndarray, int]:
        """Extract F0 sequence via FCPE, reusing shared context cache."""
        key = f"f0:{audio_path}"
        if key in self._shared_context:
            return self._shared_context[key]
        from core.utils.audio_utils import extract_f0
        y, sr = self._load_audio_shared(audio_path)
        f0, hop = extract_f0(y, sr, hop_ms=hop_ms, device=device)
        self._shared_context[key] = (f0, hop)
        return f0, hop

    def _get_speaker_baseline_shared(
        self, audio_path: str, f0: np.ndarray,
    ) -> Tuple[Optional[float], Optional[float]]:
        """Compute per-file speaker F0 baseline (mean, std)."""
        key = f"speaker_baseline:{audio_path}"
        if key in self._shared_context:
            return self._shared_context[key]
        voiced = f0[f0 > 0]
        if len(voiced) < _MIN_VOICED_FRAMES:
            logger.warning("Too few voiced frames in %s, skipping pitch", audio_path)
            result = (None, None)
        else:
            mean = float(np.mean(voiced))
            std = max(float(np.std(voiced, ddof=1)), _MIN_SPEAKER_STD_HZ)
            result = (mean, std)
        self._shared_context[key] = result
        return result

    @staticmethod
    def _get_voiced_f0(
        f0: np.ndarray, sr: int, hop: int, start: float, end: float,
    ) -> np.ndarray:
        """Slice F0 array by time range and return only voiced frames."""
        rate = sr / hop
        s = int(start * rate)
        e = int(end * rate)
        seg = f0[s:e]
        return seg[seg > 0]

    # ------------------------------------------------------------------
    # abstract interface
    # ------------------------------------------------------------------

    @abstractmethod
    def load_model(self) -> None:
        """load pretrained model or method"""
        pass

    @abstractmethod
    def extract(self, audio_path: str, time_segments: List[TimeSegment], lang: str = None) -> List[Dict]:
        """
        Extract features from audio and optional text.

        Args:
            time_segments (List[TimeSegment]): List of time segments for alignment.

        Returns:
            Dictionary containing extracted features and controls
        """
        pass


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)

    class _DummyExtractor(BaseExtractor):
        def load_model(self):
            pass
        def extract(self, audio_path, time_segments, lang=None):
            return []

    ext = _DummyExtractor({"style": "read"})
    assert ext._population_stats is not None, "Failed to load default population stats"

    # zh_read speed: mean=4.88, std=0.45
    z = ext._zscore(4.88, "speed", "zh_read")
    assert z is not None and abs(z) < 0.01, f"Expected z ~= 0.0 for mean value, got {z}"

    z_high = ext._zscore(5.8, "speed", "zh_read")
    assert z_high is not None and z_high > 1.5, f"Expected z > 1.5, got {z_high}"

    assert not ext._should_annotate(0.3), "_should_annotate(0.3) should be False"
    assert not ext._should_annotate(-0.5), "_should_annotate(-0.5) should be False"
    assert ext._should_annotate(0.8), "_should_annotate(0.8) should be True"
    assert ext._should_annotate(-1.0), "_should_annotate(-1.0) should be True"
    assert not ext._should_annotate(0.8, threshold=1.0), "threshold=1.0 should reject 0.8"
    assert ext._should_annotate(1.2, threshold=1.0), "threshold=1.0 should accept 1.2"

    assert ext._clamp_z(5.5) == 3.0, "_clamp_z(5.5) should be 3.0"
    assert ext._clamp_z(-4.0) == -3.0, "_clamp_z(-4.0) should be -3.0"
    assert ext._clamp_z(1.5) == 1.5, "_clamp_z(1.5) should be 1.5"

    assert ext._z_to_label(-2.0) == "xlow"
    assert ext._z_to_label(-1.0) == "low"
    assert ext._z_to_label(0.0) == "normal"
    assert ext._z_to_label(1.0) == "high"
    assert ext._z_to_label(1.8) == "xhigh"

    speed_labels = ("xslow", "slow", "normal", "fast", "xfast")
    assert ext._z_to_label(-2.0, speed_labels) == "xslow"
    assert ext._z_to_label(1.8, speed_labels) == "xfast"

    vol_labels = ("whisper", "soft", "normal", "loud", "shout")
    assert ext._z_to_label(-0.8, vol_labels) == "soft"
    assert ext._z_to_label(1.2, vol_labels) == "loud"

    assert ext._zscore(999.0, "nonexistent", "zh_read") is None
    assert ext._zscore(999.0, "speed", "xx_unknown") is None

    bad_ext = _DummyExtractor({"population_stats_path": "/nonexistent.json"})
    assert bad_ext._population_stats is None
    assert bad_ext._zscore(4.88, "speed", "zh_read") is None

    assert ext._get_lang_style("zh") == "zh_read"
    assert ext._get_lang_style("chinese") == "zh_read"
    assert ext._get_lang_style("en") == "en_read"
    assert ext._get_lang_style("english") == "en_read"

    print("All tests passed.")