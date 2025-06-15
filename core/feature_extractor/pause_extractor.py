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
        self.ptypes = ['none', 'micro', 'normal', 'emphatic', 'long']
        self.thresholds = [0.05, 0.3, 0.8, 1.5, np.float64('inf')]
        self.default_gap_thresholds = {}
        for ptype, threshold in zip(self.ptypes, self.thresholds):
            self.default_gap_thresholds[ptype] = threshold

    def load_model(self) -> None:
        pass
        
    def extract(self, audio_path: str, time_segments: List[TimeSegment]) -> List[Dict]:
        """
        input:   
            - audio_path: audio path (for raw waveform analysis)
            - time_segments: aligned time segments list
        return:
            - [{type: 'break', value: ptype, pos: 0, dur: 0.5}, ...] 
                - type: feature type (break)
                - value: pause type (short or long)
                - pos: word position of pause in time segments list
                - dur: duration of pause in seconds
        """
        pauses = []
        prev_seg, curr_seg = None, None
        seg_idx = 0
        for curr_seg in time_segments:
            if curr_seg.type == 'word':
                if prev_seg is None:
                    prev_seg = curr_seg
                    seg_idx += 1
                    continue
                gap = curr_seg.start - prev_seg.end
                if gap < self.config.get(self.ptypes[0], self.default_gap_thresholds[self.ptypes[0]]):
                    prev_seg = curr_seg
                    seg_idx += 1                    
                    continue
                for i_ptype in range(1, len(self.ptypes)):
                    if gap < self.config.get(self.ptypes[i_ptype], self.default_gap_thresholds[self.ptypes[i_ptype]]):
                        ptype = self.ptypes[i_ptype]
                        break
                pauses.append({
                    "type": self.type,
                    "value": ptype, 
                    "pos": seg_idx,
                    "dur": round(gap, 3)
                })
                prev_seg = curr_seg
                seg_idx += 1
        return pauses
