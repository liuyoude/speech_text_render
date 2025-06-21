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
        self.default_pause_levels = {
            'none': 0.1,
            'short': 0.3,
            'medium': 0.8,
            'long': 1.5,
            'xlong': np.float64('inf'),
        }
        self.pause_levels = self.config.get('pause_levels', self.default_pause_levels)
        self.ptypes = list(self.pause_levels.keys())
        self.thresholds = list(self.pause_levels.values())


    def load_model(self) -> None:
        pass
        
    def extract(self, audio_path: str, time_segments: List[TimeSegment], lang: str=None) -> List[Dict]:
        """
        input:   
            - audio_path: audio path (for raw waveform analysis)
            - time_segments: aligned time segments list
        return:
            - [{type: 'break', value: ptype, pos: 0, dur: 0.5}, ...] 
                - type: feature type (break)
                - value: pause type (short or long)
                - pos: word position of pause in time segments list
        """
        pause_controls = []
        prev_seg, curr_seg = None, None
        seg_idx = 0
        for curr_seg in time_segments:
            if curr_seg.type == 'word':
                if prev_seg is not None:
                    gap = curr_seg.start - prev_seg.end
                    ptype = None
                    for i_ptype in range(0, len(self.ptypes)):
                        if gap < self.thresholds[i_ptype]:
                            ptype = self.ptypes[i_ptype]
                            break
                    if ptype is not None and ptype != self.ptypes[0]:
                        pause_controls.append({
                            "type": self.type,
                            "value": ptype, 
                            "pos": seg_idx,
                            "info": f'duration={gap:.2f}s',
                            # "dur": round(gap, 3)
                        })
                prev_seg = curr_seg
                seg_idx += 1
        return pause_controls
