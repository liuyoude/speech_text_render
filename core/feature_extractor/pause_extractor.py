# -*- coding: utf-8 -*-
"""
pause control extractor
date: 20250514
author: liuyoude
"""
import numpy as np
from typing import Dict, List
from core.feature_extractor.base_extractor import BaseExtractor, TimeSegment


class PauseExtractor(BaseExtractor):
    def __init__(self, config):
        super().__init__(config)
        self.type = "break"
        self.default_pause_levels = {
            "none": 0.2,
            "short": 0.5,
            "medium": 0.9,
            "long": 1.5,
            "xlong": np.float64("inf"),
        }
        self.pause_levels = self.config.get("pause_levels", self.default_pause_levels)
        self.number_control = self.config.get("number_control", False)
        self.pause_types = list(self.pause_levels.keys())

    def load_model(self) -> None:
        pass

    def extract(self, audio_path: str, time_segments: List[TimeSegment], lang: str = None) -> List[Dict]:
        pause_controls: List[Dict] = []
        word_to_clause = self._build_word_to_clause_map(time_segments)
        prev_seg = None
        prev_idx = None

        for seg_idx, curr_seg in enumerate(time_segments):
            if curr_seg.type != "word":
                continue
            if prev_seg is not None:
                gap = curr_seg.start - prev_seg.end
                pause_type = self._get_pause_level(gap, self.pause_levels)
                if pause_type != self.pause_types[0]:
                    is_inter = word_to_clause.get(prev_idx) != word_to_clause.get(seg_idx)
                    loc = "inter_clause" if is_inter else "intra_clause"
                    pause_controls.append({
                        "type": self.type,
                        "value": round(gap, 1) if self.number_control else pause_type,
                        "pos": seg_idx,
                        "info": f"duration={gap:.2f}s, {loc}",
                    })
            prev_seg = curr_seg
            prev_idx = seg_idx

        return pause_controls

    def _get_pause_level(self, gap: float, pause_levels: Dict) -> str:
        for level, threshold in pause_levels.items():
            if gap < threshold:
                return level
        return self.pause_types[-1]

    @staticmethod
    def _build_word_to_clause_map(time_segments: List[TimeSegment]) -> Dict[int, int]:
        """Map each word segment index to its clause group index."""
        mapping: Dict[int, int] = {}
        clause_id = 0
        for idx, seg in enumerate(time_segments):
            if seg.type == "word":
                mapping[idx] = clause_id
            elif seg.type == "clause":
                clause_id += 1
        return mapping
