# -*- coding: utf-8 -*-
"""
pause control extractor
date: 20250514
author: liuyoude
"""
import os
import sys
import numpy as np
from typing import Dict, Optional, List, Tuple
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from core.feature_extractor.base_extractor import BaseExtractor, TimeSegment

class PauseExtractor(BaseExtractor):
    def __init__(self, config):
        super().__init__(config)
        # default config
        self.type = 'break'
        self.default_number_control = False
        self.default_pause_levels = {
            'none': 0.2, # 0.1
            'short': 0.5, # 0.3
            'medium': 0.9, # 0.8
            'long': 1.5,
            'xlong': np.float64('inf'),
        }
        self.pause_levels = self.config.get('pause_levels', self.default_pause_levels)
        self.number_control = self.config.get('number_control', self.default_number_control)
        self.pause_types = list(self.pause_levels.keys())

    def load_model(self) -> None:
        pass
        
    def extract(self, audio_path: str, time_segments: List[TimeSegment], lang: str=None) -> List[Dict]:
        """
        extract pause control
        """
        pause_controls = []
        prev_seg, curr_seg = None, None
        # seg_idx = 0
        for seg_idx, curr_seg in enumerate(time_segments):
            if curr_seg.type == 'word':
                if prev_seg is not None:
                    gap = curr_seg.start - prev_seg.end
                    pause_type = self._get_pause_level(gap, self.pause_levels)
                    # filter none pause type
                    if pause_type != self.pause_types[0]:
                        pause_controls.append({
                            "type": self.type,
                            "value": round(gap, 1) if self.number_control else pause_type, 
                            "pos": seg_idx,
                            "info": f'duration={gap:.2f}s',
                            # "dur": round(gap, 3)
                        })
                prev_seg = curr_seg
                # seg_idx += 1
        return pause_controls
    
    def _get_pause_level(self, gap: float, pause_levels: Dict) -> str:
        for level, threshold in pause_levels.items():
            if gap < threshold:
                return level
        return self.pause_types[-1]
