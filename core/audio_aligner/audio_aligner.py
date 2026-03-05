# -*- coding: utf-8 -*-
"""
align audio and text, get time segments for each word/sentence
date: 20250422
author: liuyoude
"""
import logging
import os
import re
from abc import ABC, abstractmethod
from typing import Optional, List
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)
import torch
import librosa
import numpy as np
from core.audio_aligner.text_normalizer import TextNormalizer, SENTENCE_END_CHARS, is_punctuation
from core.utils.audio_utils import calculate_perceptual_energy

@dataclass
class TimeSegment:
    start: float                # start time in seconds
    end: float                  # end time in seconds
    text: str                   # corresponding text (word/sentence)
    type: str = "word"          # segment type (word/sentence/paragraph)

class BaseAligner(ABC):
    @abstractmethod
    def align(self, audio_path: str, text: Optional[str] = None) -> List[TimeSegment]:
        raise NotImplementedError

class AlignmentError(Exception):
    pass

class SegmentFixer:
    "fix time for time segments, remove silence"
    def __init__(self, frame_ms=20, hop_ms=10, silence_threshold=0.002):
        self.frame_ms = frame_ms # ms
        self.hop_ms = hop_ms # ms
        self.silence_threshold = silence_threshold
    
    def fix_segments(self, audio_path: str, time_segments: List[TimeSegment]):
        """修正时间段，去除可能存在的静音片段"""
        audio, sr = librosa.load(audio_path, sr=None, mono=True)
        frame_length, hop_length = self._calculate_frame_params(sr)
        full_energy = calculate_perceptual_energy(audio, sr, frame_length, hop_length)
        first_word_seg, last_word_seg = None, None
        first_sentence_seg, last_sentence_seg = None, None
        for seg in time_segments:
            if seg.type == 'word':
                seg = self._fix_word_segment(full_energy, sr, seg, hop_length, self.silence_threshold)
                if first_word_seg is None:
                    first_word_seg = seg
                last_word_seg = seg
            elif seg.type == "sentence":
                seg.start = first_word_seg.start
                seg.end = last_word_seg.end
                first_word_seg = None
                if first_sentence_seg is None:
                    first_sentence_seg = seg
                last_sentence_seg = seg
            elif seg.type == "paragraph":
                seg.start = first_sentence_seg.start
                seg.end = last_sentence_seg.end
                first_sentence_seg = None                
    
    def _calculate_frame_params(self, sr):
        """根据采样率计算合适的帧参数"""
        frame_length = int(self.frame_ms * sr / 1000)
        hop_length = int(self.hop_ms * sr / 1000)
        frame_length = 2 ** int(np.log2(frame_length) + 0.5)
        hop_length = max(1, hop_length)
        return frame_length, hop_length
    
    def _fix_word_segment(self, full_energy, sr, time_segment, hop_length, silence_threshold):
        """提取指定时间段内的能量值，并过滤静音片段"""
        frame_rate = sr / hop_length
        start_frame = int(time_segment.start * frame_rate)
        end_frame = int(time_segment.end * frame_rate)
        segment_energy = full_energy[start_frame:end_frame]
        
        if len(segment_energy) == 0:
            return time_segment
        
        local_threshold = max(
            np.percentile(segment_energy, 10),
            silence_threshold
        )
        
        non_silent_mask = segment_energy > local_threshold
        non_silent_indices = np.where(non_silent_mask)[0]
        
        if len(non_silent_indices) == 0:
            return time_segment

        start_idx = non_silent_indices[0]
        end_idx = non_silent_indices[-1]

        time_segment.start = (start_frame + start_idx) / frame_rate
        time_segment.end = (start_frame + end_idx) / frame_rate

        return time_segment


LANG_MAP = {
    "zh": "Chinese", "chinese": "Chinese",
    "en": "English", "english": "English",
    "yue": "Cantonese", "cantonese": "Cantonese",
    "ja": "Japanese", "japanese": "Japanese",
    "ko": "Korean", "korean": "Korean",
    "fr": "French", "french": "French",
    "de": "German", "german": "German",
    "es": "Spanish", "spanish": "Spanish",
    "pt": "Portuguese", "portuguese": "Portuguese",
    "ru": "Russian", "russian": "Russian",
    "it": "Italian", "italian": "Italian",
}


class Qwen3Aligner(BaseAligner):
    """Forced alignment using Qwen3-ForcedAligner. Requires text and language."""

    def __init__(self, model_name="Qwen/Qwen3-ForcedAligner-0.6B",
                 device="cuda:0", time_fix=True):
        from qwen_asr import Qwen3ForcedAligner
        self.aligner = Qwen3ForcedAligner.from_pretrained(
            model_name, dtype=torch.bfloat16, device_map=device,
        )
        self.text_normalizer = TextNormalizer()
        self.time_fix = time_fix
        self.segment_fixer = SegmentFixer(frame_ms=20, hop_ms=10, silence_threshold=0.002)

    def align(self, audio_path: str, text: str = None, lang: str = "zh") -> List[TimeSegment]:
        if text is None:
            raise AlignmentError("Qwen3Aligner requires text for forced alignment")

        qwen_lang = LANG_MAP.get(lang.lower())
        if qwen_lang is None:
            raise ValueError(f"Unsupported language for Qwen3 aligner: {lang}")

        results = self.aligner.align(audio=audio_path, text=text, language=qwen_lang)
        items = results[0]

        original_words = self.text_normalizer.normalize(text).split()
        segments = self._items_to_segments(items, original_words, text)

        if self.time_fix:
            self.segment_fixer.fix_segments(audio_path, segments)
        return segments

    def _items_to_segments(self, items, original_words, original_text):
        """Convert ForcedAlignItem list to List[TimeSegment] with word/sentence/paragraph."""
        clean_items = [it for it in items]
        clean_orig = [re.sub(r'[^\w\u4e00-\u9fff]', '', w, flags=re.UNICODE).lower()
                      for w in original_words]

        item_idx = 0
        word_segments = []
        for orig_idx, orig_word in enumerate(original_words):
            clean_word = clean_orig[orig_idx]
            if not clean_word:
                if word_segments:
                    word_segments[-1].text += orig_word
                continue

            is_chinese = any('\u4e00' <= c <= '\u9fff' for c in clean_word)
            if is_chinese:
                n_chars = len(clean_word)
                if item_idx + n_chars > len(clean_items):
                    n_chars = len(clean_items) - item_idx
                if n_chars <= 0:
                    continue
                start_t = clean_items[item_idx].start_time
                end_t = clean_items[item_idx + n_chars - 1].end_time
                item_idx += n_chars
                word_segments.append(TimeSegment(
                    start=start_t, end=end_t, text=orig_word, type="word",
                ))
            else:
                if item_idx < len(clean_items):
                    it = clean_items[item_idx]
                    item_idx += 1
                    word_segments.append(TimeSegment(
                        start=it.start_time, end=it.end_time,
                        text=orig_word, type="word",
                    ))

        segments = []
        current_sentence = []
        for ws in word_segments:
            segments.append(ws)
            current_sentence.append(ws)
            if any(c in SENTENCE_END_CHARS for c in ws.text):
                segments.append(self._create_sentence_segment(current_sentence))
                current_sentence = []
        if current_sentence:
            segments.append(self._create_sentence_segment(current_sentence))

        paragraph_segments = [TimeSegment(
            start=word_segments[0].start if word_segments else 0.0,
            end=word_segments[-1].end if word_segments else 0.0,
            text=original_text,
            type="paragraph",
        )]
        return segments + paragraph_segments

    def _create_sentence_segment(self, word_segments: List[TimeSegment]) -> TimeSegment:
        text = ' '.join([ws.text for ws in word_segments])
        text = self.text_normalizer.denormalize(text)
        return TimeSegment(
            start=word_segments[0].start,
            end=word_segments[-1].end,
            text=text,
            type="sentence",
        )


class Qwen3ASR:
    """Wrapper around Qwen3ASRModel for transcription and language detection."""

    def __init__(self, model_name="Qwen/Qwen3-ASR-0.6B", device="cuda:0"):
        from qwen_asr import Qwen3ASRModel
        self.model = Qwen3ASRModel.from_pretrained(
            model_name, dtype=torch.bfloat16, device_map=device,
            max_new_tokens=256,
        )

    def transcribe(self, audio_path: str, language: str = None):
        """Return (text, detected_language)."""
        qwen_lang = None
        if language is not None:
            qwen_lang = LANG_MAP.get(language.lower(), language)
        results = self.model.transcribe(audio=audio_path, language=qwen_lang)
        r = results[0]
        lang_code = r.language.lower() if r.language else None
        return r.text, lang_code

    def detect_language(self, audio_path: str) -> str:
        """Return ISO language code (e.g. 'zh', 'en')."""
        _, lang = self.transcribe(audio_path)
        return lang


class SpeechTextAligner(BaseAligner):
    """Unified aligner using Qwen3 backend for ASR and forced alignment."""

    def __init__(self, device="cpu", time_fix=True):
        self.device = device
        self.time_fix = time_fix
        self.time_segments = []
        self.lang = None

        self._qwen3_aligner = None
        self._qwen3_asr = None

    @property
    def qwen3_aligner(self):
        if self._qwen3_aligner is None:
            self._qwen3_aligner = Qwen3Aligner(device=self.device, time_fix=self.time_fix)
        return self._qwen3_aligner

    @property
    def qwen3_asr(self):
        if self._qwen3_asr is None:
            self._qwen3_asr = Qwen3ASR(device=self.device)
        return self._qwen3_asr

    def align(self, audio_path: str, text: str = None, lang: str = None) -> List[TimeSegment]:
        if text is None or lang is None:
            transcribed_text, detected_lang = self.qwen3_asr.transcribe(audio_path)
            if text is None:
                text = transcribed_text
            if lang is None:
                lang = detected_lang
        self.lang = lang.lower()
        self.time_segments = self.qwen3_aligner.align(audio_path, text, self.lang)
        return self.time_segments

    def plot(self, audio_path: str, save_path: str = None):
        if len(self.time_segments) == 0:
            raise ValueError("No alignment result, please align first")
        plot_alignment(audio_path, self.time_segments, save_path)

    def test(self, audio_path: str, text: str, lang: str):
        self.align(audio_path, text, lang)
        for seg in self.time_segments:
            logger.debug("[%.2f-%.2f] %s (type: %s)", seg.start, seg.end, seg.text, seg.type)
        self.plot(audio_path)
    
def plot_alignment(audio_path, segments, save_path=None):
    import matplotlib.pyplot as plt
    import numpy as np
    import librosa
    import matplotlib.font_manager as fm
    zh_fonts = ['SimHei', 'WenQuanYi Zen Hei', 'Noto Sans CJK SC', 'AR PL UMing CN', 'DejaVu Sans']
    available = {f.name for f in fm.fontManager.ttflist}
    plt.rcParams['font.sans-serif'] = [f for f in zh_fonts if f in available] or ['DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False

    audio, sr = librosa.load(audio_path, sr=None, mono=True)
    plt.figure(figsize=(15, 6))
    ax1 = plt.subplot(2, 1, 1)
    ax2 = plt.subplot(2, 1, 2, sharex=ax1)

    duration = len(audio) / sr
    time_axis = np.linspace(0, duration, len(audio))

    ax1.plot(time_axis, audio)
    ax1.set_ylabel("Amplitude")


    n_fft, hop_length, n_mels = 1024, 512, 128
    S = librosa.feature.melspectrogram(y=audio, sr=sr, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels)
    S_db = librosa.amplitude_to_db(S, ref=np.max)
    img = librosa.display.specshow(S_db, sr=sr, x_axis='time', 
                          y_axis='mel', ax=ax2,
                          hop_length=hop_length,
                          cmap='viridis')
    ax2.set(title='Log-Power Spectrogram')
    
    colors = {'word': 'red', 'sentence': 'blue', 'segment': 'green'}
    for seg in segments:
        if seg.type == 'word':
            ax1.axvspan(seg.start, seg.end, alpha=0.2, color=colors['word'])
            ax2.axvspan(seg.start, seg.end, alpha=0.1, color='white')
            ax1.text(seg.start, 0, seg.text, rotation=45)
            ax2.text(seg.start, 4096, seg.text, rotation=45, color='white')
        elif seg.type == 'sentence':
            ax1.axvline(seg.start, color=colors['sentence'], linestyle='--')
            ax1.axvline(seg.end, color=colors['sentence'], linestyle='--')
            ax2.axvline(seg.start, color=colors['sentence'], linestyle='--')
            ax2.axvline(seg.end, color=colors['sentence'], linestyle='--')
            ax1.text(seg.start + 0.1, -0.5, seg.text, rotation=0)
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()

if __name__ == "__main__":
    file_path_en = r"examples/audios/en/LJ001-0005.wav"
    ori_text_en = "the invention of movable metal letters in the middle of the fifteenth century may justly be considered as the invention of the art of printing."

    file_path_zh = r"examples/audios/zh/D4_750.wav"
    ori_text_zh = "苏北军的一些爱国将士，马战山、李渡、唐巨武、苏炳爱、邓铁梅等也奋起抗战。"

    speech_text_aligner = SpeechTextAligner(device='cuda', time_fix=True)

    segments_zh = speech_text_aligner.align(file_path_zh, ori_text_zh, lang="zh")
    for seg in segments_zh:
        print(f"[{seg.start:.2f}-{seg.end:.2f}] {seg.text} (type: {seg.type})")
    plot_alignment(file_path_zh, segments_zh)

    segments_en = speech_text_aligner.align(file_path_en, ori_text_en, lang="en")
    for seg in segments_en:
        print(f"[{seg.start:.2f}-{seg.end:.2f}] {seg.text} (type: {seg.type})")
    plot_alignment(file_path_en, segments_en)
