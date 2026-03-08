# -*- coding: utf-8 -*-
"""
speed control extractor
date: 20250621
author: liuyoude
"""
from typing import Dict, List
from core.feature_extractor.base_extractor import BaseExtractor, TimeSegment

PAUSE_THRESHOLD = 0.2  # gaps longer than this are subtracted from duration

_SPEED_LABELS = ("xslow", "slow", "normal", "fast", "xfast")


class SpeedExtractor(BaseExtractor):
    def __init__(self, config):
        super().__init__(config)
        self.type = "speed"

    def load_model(self) -> None:
        pass

    def extract(self, audio_path: str, time_segments: List[TimeSegment], lang: str = None) -> List[Dict]:
        lang_style = self._get_lang_style(lang)
        speed_controls: List[Dict] = []
        clause_groups = self._group_by_clause(time_segments)

        for clause_seg, word_indices in clause_groups:
            if not word_indices:
                continue

            word_count = len(word_indices)
            eff_dur = self._effective_duration(time_segments, clause_seg, word_indices)
            if eff_dur <= 0:
                continue

            speed = word_count / eff_dur
            z = self._zscore(speed, "speed", lang_style)
            if z is not None and self._should_annotate(z):
                speed_controls.append({
                    "type": self.type,
                    "value": self._format_z(z, _SPEED_LABELS),
                    "pos": word_indices[0],
                    "info": f"speed={speed:.2f}word/s, z={z:.2f}",
                })

        return speed_controls

    @staticmethod
    def _effective_duration(time_segments, clause_seg, word_indices):
        """Clause duration minus large inter-word gaps (> PAUSE_THRESHOLD)."""
        total = clause_seg.end - clause_seg.start
        pause_deduction = 0.0
        for i in range(1, len(word_indices)):
            prev = time_segments[word_indices[i - 1]]
            curr = time_segments[word_indices[i]]
            gap = curr.start - prev.end
            if gap > PAUSE_THRESHOLD:
                pause_deduction += gap
        return total - pause_deduction
