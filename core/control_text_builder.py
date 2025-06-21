# -*- coding: utf-8 -*-
"""
control generator using extractor
date: 20250514
author: liuyoude
"""
import os
import sys
from typing import Dict, Optional, List, Tuple, Union
# print(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core.feature_extractor.base_extractor import BaseExtractor, TimeSegment
from core.audio_aligner.audio_aligner import SpeechTextAligner, plot_alignment
from core.feature_extractor import PauseExtractor, SpeedExtractor

def get_audio_text_path_list(root_dir):
    ext = '.wav'
    audio_path_list = []
    text_path_list = []
    for root, dirs, files in os.walk(root_dir):
        for file in files:
            if file.endswith(ext):
                audio_path_list.append(os.path.join(root, file))
                text_path_list.append(os.path.join(root, file.replace(ext, '.txt')))
    return audio_path_list, text_path_list

class ControlGenerator:
    def __init__(self, config):
        self.config = config
        self.extractors = {}

    def add_extractor(self, extractor: Union[BaseExtractor, List[BaseExtractor]]):
        """add feature extractor"""
        if isinstance(extractor, list):
            for e in extractor:
                self.extractors[e.type] = e
        else:
            self.extractors[extractor.type] = extractor

    def generate(self, audio_path: str, time_segments: List[TimeSegment], lang: str=None) -> List[Dict]:
        """
        generate control flag and text
        """
        controls = {}
        for extractor in self.extractors.values():
            extracts = extractor.extract(audio_path, time_segments, lang=lang)
            for extract in extracts:
                print(f'type: {extract["type"]}=={extract["value"]}, info: {extract["info"]}')
                if extract['pos'] not in controls:
                    controls[extract['pos']] = f'[{extract["type"]}={extract["value"]}]'
                else:
                    controls[extract['pos']] += f'[{extract["type"]}={extract["value"]}]'
                    controls[extract['pos']].replace('][', ',')
        return controls
    
class ControlBuilder:
    def __init__(self, config):
        self.config = config
        self.aligner = SpeechTextAligner()  # 音频对齐器
        self.control_generator = ControlGenerator(config)

    def build(self, audio_path: str, text: str, lang: str=None) -> str:
        """
        generate text with control
        """
        self.audio_path = audio_path
        self.time_segments = self.aligner.align(audio_path, text, lang=lang)
        controls = self.control_generator.generate(audio_path, self.time_segments, lang=self.aligner.lang)
        for seg_idx, control in zip(controls.keys(), controls.values()):
            self.time_segments[seg_idx].text = control + self.time_segments[seg_idx].text
        control_text = ' '.join([seg.text for seg in self.time_segments if seg.type == 'word'])
        return control_text
    
    def add_extractor(self, extractor: Union[BaseExtractor, List[BaseExtractor]]) -> None:
        self.control_generator.add_extractor(extractor)

    def plot(self, save_path: str = None) -> None:
        plot_alignment(self.audio_path, self.time_segments, save_path)

    def test(self, audio_path: str, text: str) -> None:
        """
        测试控制文本生成
        input:
            - audio_path: 音频路径
            - text: 文本
        """
        control_text = builder.build(audio_path, text)
        print(control_text)
        self.plot()        
        
    
if __name__ == "__main__":
    config = {
        "speed_extractor": {
            "use_ratio": False,
        }
    }
    builder = ControlBuilder(config)
    extractors = [
        PauseExtractor(config),
        SpeedExtractor(config['speed_extractor']),
    ]
    builder.add_extractor(extractors)

    # audio_path_list, text_path_list = get_audio_text_path_list(r"examples/audios")
    # for audio_path, text_path in zip(audio_path_list, text_path_list):
    #     with open(text_path, 'r', encoding='utf-8') as f:
    #         ori_text_en = f.read()
    #     builder.test(audio_path, ori_text_en)

    file_path_en = r"examples/audios/en/LJ001-0005.wav"
    ori_text_en = "the invention of movable metal letters in the middle of the fifteenth century may justly be considered as the invention of the art of printing."
    builder.test(file_path_en, ori_text_en)
    # for seg in builder.time_segments:
    #     print(f"[{seg.start:.2f}-{seg.end:.2f}] {seg.text} (type: {seg.type})")   