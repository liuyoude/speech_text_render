# -*- coding: utf-8 -*-
"""
control generator using extractor
date: 20250514
author: liuyoude
"""
import logging
import os
from collections import defaultdict
from typing import Dict, List, Union

from core.feature_extractor.base_extractor import BaseExtractor, TimeSegment

logger = logging.getLogger(__name__)
from core.audio_aligner.audio_aligner import SpeechTextAligner, plot_alignment
from core.feature_extractor import (
    PauseExtractor, SpeedExtractor, VolumeExtractor, EmotionExtractor,
    PitchExtractor, EmphasisExtractor, StyleExtractor,
)

_CONTROL_PRIORITY = {
    "style": 0,
    "emotion": 1,
    "speed": 2,
    "volume": 3,
    "pitch": 4,
    "break": 5,
    "emphasis": 6,
}


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
        self.max_controls_per_clause = config.get("max_controls_per_clause", None)

    def add_extractor(self, extractor: Union[BaseExtractor, List[BaseExtractor]]):
        """add feature extractor"""
        if isinstance(extractor, list):
            for e in extractor:
                self.extractors[e.type] = e
        else:
            self.extractors[extractor.type] = extractor

    # ------------------------------------------------------------------
    # public API
    # ------------------------------------------------------------------

    def generate(self, audio_path: str, time_segments: List[TimeSegment],
                 lang: str = None) -> Dict[int, str]:
        """Return formatted controls: ``{pos: "[type=val][type=val]..."}``."""
        raw = self._extract_all(audio_path, time_segments, lang)
        raw = self._apply_density_limit(raw, time_segments)
        return self._format_controls(raw)

    def extract_raw(self, audio_path: str, time_segments: List[TimeSegment],
                    lang: str = None) -> List[Dict]:
        """Return raw control dicts after density limiting."""
        raw = self._extract_all(audio_path, time_segments, lang)
        return self._apply_density_limit(raw, time_segments)

    # ------------------------------------------------------------------
    # internal helpers
    # ------------------------------------------------------------------

    def _extract_all(self, audio_path: str, time_segments: List[TimeSegment],
                     lang: str = None) -> List[Dict]:
        """Collect raw control dicts from every registered extractor."""
        all_controls: List[Dict] = []
        shared_ctx: Dict = {}
        for extractor in self.extractors.values():
            extractor._shared_context = shared_ctx
            extracts = extractor.extract(audio_path, time_segments, lang=lang)
            for ctrl in extracts:
                logger.debug("type: %s==%s, info: %s",
                             ctrl["type"], ctrl["value"], ctrl["info"])
            all_controls.extend(extracts)
        return all_controls

    def _apply_density_limit(self, controls: List[Dict],
                             time_segments: List[TimeSegment]) -> List[Dict]:
        if self.max_controls_per_clause is None or not controls:
            return controls
        word_to_clause = self._build_word_to_clause_map(time_segments)
        clause_buckets: Dict[int, List[Dict]] = defaultdict(list)
        for ctrl in controls:
            clause_id = word_to_clause.get(ctrl["pos"], -1)
            clause_buckets[clause_id].append(ctrl)
        kept: List[Dict] = []
        for ctrls in clause_buckets.values():
            ctrls.sort(key=lambda c: _CONTROL_PRIORITY.get(c["type"], 99))
            kept.extend(ctrls[:self.max_controls_per_clause])
        return kept

    @staticmethod
    def _format_controls(controls: List[Dict]) -> Dict[int, str]:
        """Group by pos, sort by priority, render each as ``[type=value]``."""
        by_pos: Dict[int, List[Dict]] = defaultdict(list)
        for ctrl in controls:
            by_pos[ctrl["pos"]].append(ctrl)
        result: Dict[int, str] = {}
        for pos, ctrls in by_pos.items():
            ctrls.sort(key=lambda c: _CONTROL_PRIORITY.get(c["type"], 99))
            result[pos] = "".join(_format_single_control(c) for c in ctrls)
        return result

    @staticmethod
    def _build_word_to_clause_map(time_segments: List[TimeSegment]) -> Dict[int, int]:
        mapping: Dict[int, int] = {}
        clause_id = 0
        for idx, seg in enumerate(time_segments):
            if seg.type == "word":
                mapping[idx] = clause_id
            elif seg.type == "clause":
                clause_id += 1
        return mapping


def _format_single_control(ctrl: Dict) -> str:
    """Render one control dict as a bracketed tag.

    - Normal:   ``[type=value]``
    - Emphasis label mode: ``[emphasis]``  (type equals value)
    """
    ctype, value = ctrl["type"], ctrl["value"]
    if ctype == "emphasis" and value == "emphasis":
        return "[emphasis]"
    return f"[{ctype}={value}]"


class ControlBuilder:
    def __init__(self, config):
        self.config = config
        device = config.get('device', 'cuda')
        self.aligner = SpeechTextAligner(device=device)
        self.control_generator = ControlGenerator(config)

    def build(self, audio_path: str, text: str, lang: str = None) -> str:
        """Generate text with control tags inserted at target positions."""
        self.audio_path = audio_path
        self.time_segments = self.aligner.align(audio_path, text, lang=lang)
        controls = self.control_generator.generate(
            audio_path, self.time_segments, lang=self.aligner.lang,
        )
        for seg_idx, control in controls.items():
            self.time_segments[seg_idx].text = control + self.time_segments[seg_idx].text
        control_text = ' '.join(
            seg.text for seg in self.time_segments if seg.type == 'word'
        )
        return control_text

    def add_extractor(self, extractor: Union[BaseExtractor, List[BaseExtractor]]) -> None:
        self.control_generator.add_extractor(extractor)

    def plot(self, save_path: str = None) -> None:
        plot_alignment(self.audio_path, self.time_segments, save_path)

    def test(self, audio_path: str, text: str, lang: str = None, plot: bool = False) -> None:
        control_text = self.build(audio_path, text)
        logger.info(control_text)
        if plot:
            self.plot()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run control text pipeline")
    parser.add_argument("--number", action="store_true",
                        help="Use z-score continuous values (training mode)")
    parser.add_argument("--max-per-clause", type=int, default=None,
                        help="Max control annotations per clause")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    number_control = args.number
    extractor_cfgs = {
        "pause": {"number_control": number_control},
        "speed": {"number_control": number_control},
        "volume": {"number_control": number_control},
        "pitch": {"device": "cuda", "number_control": number_control},
        "emphasis": {"device": "cuda", "number_control": number_control},
        "emotion": {"device": "cuda"},
        "style": {"style": "read"},
    }
    config = dict(extractor_cfgs)
    if args.max_per_clause is not None:
        config["max_controls_per_clause"] = args.max_per_clause

    builder = ControlBuilder(config)
    extractors = [
        StyleExtractor(extractor_cfgs["style"]),
        PauseExtractor(extractor_cfgs["pause"]),
        SpeedExtractor(extractor_cfgs["speed"]),
        VolumeExtractor(extractor_cfgs["volume"]),
        PitchExtractor(extractor_cfgs["pitch"]),
        EmphasisExtractor(extractor_cfgs["emphasis"]),
        EmotionExtractor(extractor_cfgs["emotion"]),
    ]
    builder.add_extractor(extractors)

    mode = "number (training)" if number_control else "label (inference)"
    logger.info("Mode: %s, max_controls_per_clause: %s", mode, args.max_per_clause)

    audio_path_list, text_path_list = get_audio_text_path_list(r"examples/audios")
    for audio_path, text_path in zip(audio_path_list, text_path_list):
        with open(text_path, "r", encoding="utf-8") as f:
            text = f.read()
        builder.test(audio_path, text, plot=False)