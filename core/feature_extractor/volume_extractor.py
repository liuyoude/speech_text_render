# -*- coding: utf-8 -*-
"""
volume control extractor
date: 20250704
author: liuyoude
"""
import numpy as np
from typing import Dict, List
from core.feature_extractor.base_extractor import BaseExtractor, TimeSegment
from core.utils.audio_utils import calculate_perceptual_energy

_VOLUME_LABELS = ("whisper", "soft", "normal", "loud", "shout")


class VolumeExtractor(BaseExtractor):
    def __init__(self, config):
        super().__init__(config)
        self.type = "volume"
        self.target_frame_ms = 30
        self.target_hop_ms = 10

    def load_model(self) -> None:
        pass

    def extract(self, audio_path: str, time_segments: List[TimeSegment], lang: str = None) -> List[Dict]:
        y, sr = self._load_audio_shared(audio_path)
        frame_length, hop_length = self._calculate_frame_params(sr)
        energy_frames = calculate_perceptual_energy(y, sr, frame_length, hop_length)

        lang_style = self._get_lang_style(lang)
        volume_controls: List[Dict] = []
        clause_groups = self._group_by_clause(time_segments)

        for clause_seg, word_indices in clause_groups:
            if not word_indices:
                continue

            clause_rms = self._get_segment_rms(energy_frames, sr, hop_length,
                                               clause_seg.start, clause_seg.end)
            clause_energy = float(np.mean(clause_rms)) if len(clause_rms) > 0 else 0.0

            first_word_idx = word_indices[0]

            # --- clause-level: population z-score → type="volume" ---
            z = self._zscore(clause_energy, "energy", lang_style)
            if z is not None and self._should_annotate(z):
                volume_controls.append({
                    "type": "volume",
                    "value": self._format_z(z, _VOLUME_LABELS),
                    "pos": first_word_idx,
                    "info": (f"[{clause_seg.text}] "
                             f"energy={clause_energy:.4f}, z={z:.2f}"),
                })

        return volume_controls

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------

    def _get_segment_rms(self, full_rms, sr, hop_length, start_time, end_time):
        frame_rate = sr / hop_length
        start_frame = int(start_time * frame_rate)
        end_frame = int(end_time * frame_rate)
        return full_rms[start_frame:end_frame]

    def _calculate_frame_params(self, sr):
        frame_length = int(self.target_frame_ms * sr / 1000)
        hop_length = int(self.target_hop_ms * sr / 1000)
        frame_length = 2 ** int(np.log2(frame_length) + 0.5)
        hop_length = max(1, hop_length)
        return frame_length, hop_length
