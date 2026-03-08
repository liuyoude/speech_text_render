# -*- coding: utf-8 -*-
"""
pitch control extractor — speaker-relative z-score
date: 20250308
author: liuyoude
"""
import logging
from typing import Dict, List

import numpy as np

from core.feature_extractor.base_extractor import (
    BaseExtractor, TimeSegment, _MIN_VOICED_FRAMES,
)

logger = logging.getLogger(__name__)

_PITCH_LABELS = ("xlow", "low", "normal", "high", "xhigh")


class PitchExtractor(BaseExtractor):
    def __init__(self, config: Dict):
        super().__init__(config)
        self.type = "pitch"
        self.device = config.get("device", "cpu")
        self._hop_ms = config.get("hop_ms", 10)

    def load_model(self) -> None:
        pass

    def extract(
        self,
        audio_path: str,
        time_segments: List[TimeSegment],
        lang: str = None,
    ) -> List[Dict]:
        f0, f0_hop = self._get_f0_shared(audio_path, device=self.device, hop_ms=self._hop_ms)
        speaker_mean, speaker_std = self._get_speaker_baseline_shared(audio_path, f0)
        if speaker_mean is None:
            return []

        y, sr = self._load_audio_shared(audio_path)
        clause_groups = self._group_by_clause(time_segments)
        pitch_controls: List[Dict] = []

        for clause_seg, word_indices in clause_groups:
            if not word_indices:
                continue

            voiced = self._get_voiced_f0(f0, sr, f0_hop, clause_seg.start, clause_seg.end)
            if len(voiced) < _MIN_VOICED_FRAMES:
                continue

            clause_f0_mean = float(np.mean(voiced))
            z = (clause_f0_mean - speaker_mean) / speaker_std

            if self._should_annotate(z):
                pitch_controls.append({
                    "type": self.type,
                    "value": self._format_z(z, _PITCH_LABELS),
                    "pos": word_indices[0],
                    "info": (
                        f"f0={clause_f0_mean:.1f}Hz, "
                        f"speaker={speaker_mean:.1f}±{speaker_std:.1f}Hz, "
                        f"z={z:.2f}"
                    ),
                })

        return pitch_controls
