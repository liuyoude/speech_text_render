# -*- coding: utf-8 -*-
"""
speed control extractor
date: 20250621
author: liuyoude
"""
import os
import sys
import numpy as np
from typing import Dict, Optional, List, Tuple
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from core.feature_extractor.base_extractor import BaseExtractor, TimeSegment
from core.audio_aligner import TextNormalizer 

class SpeedExtractor(BaseExtractor):
    def __init__(self, config):
        super().__init__(config)
        self.type = 'speed'
        self.default_number_control = True
        # speed config
        self.default_chinese_speed_levels = {
            'xslow': (1.5, 2.5),     # 极慢 1.5-2.5字/秒 (朗诵、诗歌)
            'slow': (2.5, 3.3),      # 慢速 2.5-3.3字/秒 (情感表达、教学)
            'normal': (3.3, 4.2),    # 正常 3.3-4.2字/秒 (日常对话)
            'fast': (4.2, 5.0),      # 快速 4.2-5.0字/秒 (新闻播报)
            'xfast': (5.0, 6.0),     # 极快 5.0-6.0字/秒 (体育解说、辩论)
        }
        self.default_english_speed_levels = {
            'xslow': (2.0, 3.0),     # 极慢 2.0-3.0词/秒 (诗歌朗诵)
            'slow': (3.0, 3.8),      # 慢速 3.0-3.8词/秒 (教学、演讲)
            'normal': (3.8, 4.5),    # 正常 3.8-4.5词/秒 (日常对话)
            'fast': (4.5, 5.5),      # 快速 4.5-5.5词/秒 (新闻播报)
            'xfast': (5.5, 6.5),     # 极快 5.5-6.5词/秒 (辩论、解说)
        }
        self.default_reference_speed = {
            'chinese': 3.8,  # 中文参考语速
            'english': 4.2,  # 英文参考语速
        }        
        self.chinese_speed_levels = self.config.get('chinese_speed_levels', self.default_chinese_speed_levels)
        self.english_speed_levels = self.config.get('english_speed_levels', self.default_english_speed_levels)
        self.reference_speed = self.config.get('reference_speed', self.default_reference_speed)
        self.number_control = self.config.get('number_control', self.default_number_control)
        self.text_normalizer = TextNormalizer()

    def load_model(self) -> None:
        # 语速提取无需额外模型
        pass
        
    def extract(self, audio_path: str, time_segments: List[TimeSegment], lang: str) -> List[Dict]:
        """
        extract speed control
        """
        if lang in ['chinese', 'zh']:
            speed_levels = self.chinese_speed_levels
            ref_speed = self.reference_speed['chinese']
        elif lang in ['english', 'en']:
            speed_levels = self.english_speed_levels
            ref_speed = self.reference_speed['english']
        else:
            raise ValueError(f"Unsupported language: {lang}")
        speed_controls = []
        pos = 0
        pos_location_flag = True
        word_count = 0
        for seg_idx, seg in enumerate(time_segments):
            if seg.type == 'sentence':
                pos_location_flag = True
                duration = seg.end - seg.start
                # word_count = len(self.text_normalizer.normalize(seg.text).split())
                speed = word_count / duration if duration > 0 else 0
                speed_ratio = round(speed / ref_speed, 1)
                if self.number_control:
                    if speed_ratio < 0.9 or speed_ratio > 1.1:
                        speed_controls.append({
                            "type": self.type,
                            "value": speed_ratio,
                            "pos": pos,
                            "info": f'speed={speed:.2f}word/s',
                        })
                else:
                    speed_level = self._get_speed_level(speed, speed_levels)
                    if speed_level not in ['normal', 'unknown']:
                        speed_controls.append({
                            "type": self.type,
                            "value": speed_level,
                            "pos": pos,
                            "info": f'speed={speed:.2f}word/s',
                        })
                # pos += word_count
                word_count = 0
            elif seg.type == 'word':
                word_count += 1
                if pos_location_flag:
                    pos = seg_idx
                    pos_location_flag = False

        return speed_controls

    def _get_speed_level(self, speed: float, speed_levels: Dict) -> str:
        for level, (min_val, max_val) in speed_levels.items():
            if min_val <= speed < max_val:
                return level
        return 'unknown'