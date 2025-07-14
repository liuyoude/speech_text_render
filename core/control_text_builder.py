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
from core.feature_extractor import PauseExtractor, SpeedExtractor, VolumeExtractor

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
        # merge control text in the same pos
        for pos in controls:
            controls[pos] = f'[{controls[pos][1:-1].replace("][", ",")}]'
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

    def test(self, audio_path: str, text: str, lang: str=None, plot: bool=False) -> None:
        control_text = builder.build(audio_path, text)
        print(control_text)
        if plot:
            self.plot()        
        
    
if __name__ == "__main__":
    config = {
        "pause_extractor": {
            "number_control": False,
        },
        "speed_extractor": {
            "number_control": True,
        },
        "volume_extractor": {
            "number_control": False,
        },
    }
    builder = ControlBuilder(config)
    extractors = [
        PauseExtractor(config['pause_extractor']),
        SpeedExtractor(config['speed_extractor']),
        VolumeExtractor(config['volume_extractor']),
    ]
    builder.add_extractor(extractors)

    # audio_path_list, text_path_list = get_audio_text_path_list(r"examples/audios")
    # for audio_path, text_path in zip(audio_path_list, text_path_list):
    #     with open(text_path, 'r', encoding='utf-8') as f:
    #         ori_text_en = f.read()
    #     builder.test(audio_path, ori_text_en, plot=False)

    file_path_en = r"examples/audios/en/LJ001-0001.wav"
    ori_text_en = "Printing, in the only sense with which we are at present concerned, differs from most if not from all the arts and crafts represented in the Exhibition?"
    file_path_zh = r"examples/audios/zh/D4_752.wav"
    ori_text_zh = "他们走到四马路一家茶室铺里，二九说要买鱘鱼，他给买了，又给转儿买了饼干。"
    # builder.test(file_path_en, ori_text_en)
    builder.test(file_path_zh, ori_text_zh)
    for seg in builder.time_segments:
        print(f"[{seg.start:.2f}-{seg.end:.2f}] {seg.text} (type: {seg.type})")
    builder.plot(save_path=False)