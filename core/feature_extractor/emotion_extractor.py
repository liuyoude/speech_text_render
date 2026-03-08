# -*- coding: utf-8 -*-
"""
emotion control extractor
date: 20250621
author: liuyoude
"""
import logging
import numpy as np
from typing import Dict, Optional, List
from core.feature_extractor.base_extractor import BaseExtractor, TimeSegment

logger = logging.getLogger(__name__)

_EMOTION2VEC_LABELS = [
    "angry", "disgusted", "fearful", "happy",
    "neutral", "other", "sad", "surprised", "unknow",
]


class EmotionExtractor(BaseExtractor):
    def __init__(self, config):
        super().__init__(config)
        self.type = "emotion"
        self.emotion_labels = config.get("emotion_labels", _EMOTION2VEC_LABELS)
        self.min_confidence = config.get("min_confidence", 0.3)
        self.max_emotions = config.get("max_emotions", 2)
        self.device = config.get("device", "cuda")
        self.sr = config.get("sr", 16000)
        self.model_loaded = False

    def load_model(self) -> None:
        try:
            from funasr import AutoModel
            model_id = "iic/emotion2vec_plus_large"
            self.model = AutoModel(
                model=model_id,
                hub="ms",
                disable_update=True,
                device=self.device,
            )
            self.model_loaded = True
        except ImportError:
            raise RuntimeError("funasr package not installed. "
                               "Please install with: pip install funasr")
        except Exception as e:
            raise RuntimeError(f"Failed to load emotion2vec model: {e}")

    def extract(
        self,
        audio_path: str,
        time_segments: List[TimeSegment],
        lang: str = None,
    ) -> List[Dict]:
        if not self.model_loaded:
            self.load_model()

        audio = self._load_audio(audio_path)
        return self._extract_sentence_level(audio, time_segments)

    def _extract_sentence_level(
        self,
        audio: np.ndarray,
        time_segments: List[TimeSegment],
    ) -> List[Dict]:
        emotion_controls: List[Dict] = []

        start_idx = None
        for seg_idx, segment in enumerate(time_segments):
            if segment.type in ("word", "clause"):
                if segment.type == "word" and start_idx is None:
                    start_idx = seg_idx
                continue

            if segment.type == "sentence":
                start_sample = int(segment.start * self.sr)
                end_sample = int(segment.end * self.sr)
                seg_audio = audio[start_sample:end_sample]

                emotion_output = self.model.generate(
                    seg_audio,
                    output_dir=None,
                    granularity="utterance",
                    extract_embedding=False,
                )

                emotion_str = self._parse_emotion_output(emotion_output[0])

                if emotion_str:
                    emotion_controls.append({
                        "type": self.type,
                        "value": emotion_str,
                        "pos": start_idx,
                        "info": f"[{segment.text}] emo={emotion_str}",
                    })
                start_idx = None

        return emotion_controls

    def _parse_emotion_output(self, output: Dict) -> Optional[str]:
        """Pick top-k emotions by confidence, skip neutral, return '+'-joined string."""
        scored = sorted(
            zip(self.emotion_labels, output["scores"]),
            key=lambda x: x[1],
            reverse=True,
        )

        emotions: List[str] = []
        for emotion, confidence in scored:
            if emotion in ("neutral", "other", "unknow"):
                continue
            if confidence < self.min_confidence:
                continue
            emotions.append(emotion)
            if len(emotions) >= self.max_emotions:
                break

        return "+".join(emotions) if emotions else None

    def _load_audio(self, audio_path: str) -> np.ndarray:
        import librosa
        y, _ = librosa.load(audio_path, sr=self.sr, mono=True)
        return y
