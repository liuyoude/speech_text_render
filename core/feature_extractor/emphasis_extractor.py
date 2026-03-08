# -*- coding: utf-8 -*-
"""
emphasis control extractor — composite z-score (energy + duration + F0)
date: 20250308
author: liuyoude
"""
import logging
from typing import Dict, List

import numpy as np

from core.feature_extractor.base_extractor import (
    BaseExtractor, TimeSegment, _MIN_VOICED_FRAMES,
)
from core.utils.audio_utils import calculate_perceptual_energy

logger = logging.getLogger(__name__)

_MIN_WORD_VOICED_FRAMES = 3


class EmphasisExtractor(BaseExtractor):
    def __init__(self, config: Dict):
        super().__init__(config)
        self.type = "emphasis"
        self.device = config.get("device", "cpu")
        self._hop_ms = config.get("hop_ms", 10)
        self._w_energy = config.get("w_energy", 0.4)
        self._w_duration = config.get("w_duration", 0.3)
        self._w_pitch = config.get("w_pitch", 0.3)
        self._threshold = config.get("emphasis_threshold", 1.0)
        self._target_frame_ms = 30
        self._target_hop_ms = 10

    def load_model(self) -> None:
        pass

    def extract(
        self,
        audio_path: str,
        time_segments: List[TimeSegment],
        lang: str = None,
    ) -> List[Dict]:
        y, sr = self._load_audio_shared(audio_path)
        f0, f0_hop = self._get_f0_shared(
            audio_path, device=self.device, hop_ms=self._hop_ms,
        )
        speaker_mean, speaker_std = self._get_speaker_baseline_shared(audio_path, f0)

        frame_length = int(self._target_frame_ms * sr / 1000)
        frame_length = 2 ** int(np.log2(frame_length) + 0.5)
        hop_length = max(1, int(self._target_hop_ms * sr / 1000))
        energy_frames = calculate_perceptual_energy(y, sr, frame_length, hop_length)

        lang_style = self._get_lang_style(lang)
        emphasis_controls: List[Dict] = []

        for seg_idx, seg in enumerate(time_segments):
            if seg.type != "word":
                continue

            z_components: List[float] = []
            weights: List[float] = []
            info_parts: List[str] = []

            word_energy = self._segment_energy(
                energy_frames, sr, hop_length, seg.start, seg.end,
            )
            z_energy = self._zscore(word_energy, "word_energy", lang_style)
            if z_energy is not None:
                z_components.append(z_energy)
                weights.append(self._w_energy)
                info_parts.append(f"e={z_energy:.2f}")

            dur = seg.end - seg.start
            z_dur = self._zscore(dur, "word_duration", lang_style)
            if z_dur is not None:
                z_components.append(z_dur)
                weights.append(self._w_duration)
                info_parts.append(f"d={z_dur:.2f}")

            if speaker_mean is not None:
                voiced = self._get_voiced_f0(f0, sr, f0_hop, seg.start, seg.end)
                if len(voiced) >= _MIN_WORD_VOICED_FRAMES:
                    word_f0 = float(np.mean(voiced))
                    z_pitch = (word_f0 - speaker_mean) / speaker_std
                    z_components.append(z_pitch)
                    weights.append(self._w_pitch)
                    info_parts.append(f"p={z_pitch:.2f}")

            if not z_components:
                continue

            w_sum = sum(weights)
            composite_z = sum(w * z for w, z in zip(weights, z_components)) / w_sum

            if composite_z > self._threshold:
                value = round(self._clamp_z(composite_z), 2) if self._number_control else "emphasis"
                emphasis_controls.append({
                    "type": self.type,
                    "value": value,
                    "pos": seg_idx,
                    "info": (
                        f"[{seg.text}] composite_z={composite_z:.2f} "
                        f"({', '.join(info_parts)})"
                    ),
                })

        return emphasis_controls

    @staticmethod
    def _segment_energy(energy_frames, sr, hop_length, start, end):
        frame_rate = sr / hop_length
        s = int(start * frame_rate)
        e = int(end * frame_rate)
        seg = energy_frames[s:e]
        return float(np.mean(seg)) if len(seg) > 0 else 0.0
