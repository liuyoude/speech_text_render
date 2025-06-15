# -*- coding: utf-8 -*-
"""
control generator using extractor
date: 20250514
author: liuyoude
"""
import os
import sys
from typing import Dict, Optional, List, Tuple
# print(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core.feature_extractor.base_extractor import BaseExtractor, TimeSegment
from core.audio_aligner.audio_aligner import SpeechTextAligner, plot_alignment
from core.feature_extractor.pause_extractor import PauseExtractor

class ControlGenerator:
    def __init__(self, config):
        self.config = config
        self.extractors = {}  # 存储不同类型的特征提取器

    def add_extractor(self, extractor: BaseExtractor|List[BaseExtractor]):
        """添加特征提取器"""
        if isinstance(extractor, list):
            for e in extractor:
                self.extractors[e.type] = e
        else:
            self.extractors[extractor.type] = extractor

    def generate(self, audio_path: str, time_segments: List[TimeSegment]) -> List[Dict]:
        """
        生成控制文本
        input:
            - audio_path: 音频路径
            - text: 文本
        return:
            - [{type: 'break', value: ptype, pos: 0, dur: 0.5},...]
                - type: 特征类型 (break)
                - value: 控制类型 (short or long)
                - pos: 控制位置 (word position in time segments list)
                - dur: 控制持续时间 (seconds)
        """
        controls = {}
        for extractor in self.extractors.values():
            extractor_res = extractor.extract(audio_path, time_segments)
            for control in extractor_res:
                if control['pos'] not in controls:
                    controls[control['pos']] = f'[{control["type"]}={control["value"]}]'
                else:
                    controls[control['pos']] += f'[{control["type"]}={control["value"]}]'
                    controls[control['pos']].replace('][', ',')
        return controls
    
class ControlBuilder:
    def __init__(self, config):
        self.config = config
        self.aligner = SpeechTextAligner()  # 音频对齐器
        self.generator = ControlGenerator(config)

    def build(self, audio_path: str, text: str) -> str:
        """
        构建控制文本
        input:
            - audio_path: 音频路径
            - text: 文本
        return:
            - control text
        """
        self.audio_path = audio_path
        self.time_segments = self.aligner.align(audio_path, text)
        controls = self.generator.generate(audio_path, self.time_segments)
        for seg_idx, control in zip(controls.keys(), controls.values()):
            self.time_segments[seg_idx].text = control + self.time_segments[seg_idx].text
        control_text = ' '.join([seg.text for seg in self.time_segments if seg.type == 'word'])
        return control_text
    
    def add_extractor(self, extractor: BaseExtractor|List[BaseExtractor]):
        """添加特征提取器"""
        self.generator.add_extractor(extractor)

    def plot(self) -> None:
        plot_alignment(self.audio_path, self.time_segments)
        
    
if __name__ == "__main__":
    config = {
    }
    builder = ControlBuilder(config)
    builder.add_extractor(PauseExtractor(config))
    file_path_en = r"examples/audios/en/LJ001-0005.wav"
    ori_text_en = "the invention of movable metal letters in the middle of the fifteenth century may justly be considered as the invention of the art of printing."
    control_text = builder.build(file_path_en, ori_text_en)
    print(control_text)
    builder.plot()
    for seg in builder.time_segments:
        print(f"[{seg.start:.2f}-{seg.end:.2f}] {seg.text} (type: {seg.type})")