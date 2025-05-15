# -*- coding: utf-8 -*-
"""
pause control extractor
date: 20250514
author: liuyoude
"""
from typing import Dict, Optional, List, Tuple
from core.feature_extractor.base_extractor import BaseExtractor, TimeSegment
from core.audio_aligner.audio_aligner import SpeechTextAligner

class PauseExtractor(BaseExtractor):
    def __init__(self, config):
        super().__init__(config)
        # default config
        self.default_gap_thresholds = {
            'silence': 0.05,
            'short_pause': 0.5,
            'long_pause': 1.0
        }
        self.type = 'break'
        self.ptype_short = 'short'
        self.ptype_long = 'long' 
        
    def extract(self, audio_path: str, time_segments: List[TimeSegment]) -> List[Dict]:
        """
        input:   
            - audio_path: audio path (for raw waveform analysis)
            - time_segments: aligned time segments list
        output:
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
                    continue
                gap = curr_seg.start - prev_seg.end
                if gap < self.config.get('silence', self.default_gap_thresholds['silence']):
                    continue
                if gap >= self.config.get('long_pause', self.default_gap_thresholds['long_pause']):
                    ptype = self.ptype_long
                elif gap > self.config.get('short_pause', self.default_gap_thresholds['short_pause']):
                    ptype = self.ptype_short
                else:
                    ptype = self.ptype_short
                pauses.append({
                    "type": self.type,
                    "value": ptype, 
                    "pos": seg_idx,
                    "dur": round(gap, 3)
                })
                prev_seg = curr_seg
                seg_idx += 1
        return pauses
