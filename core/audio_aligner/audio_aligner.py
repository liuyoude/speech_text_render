# -*- coding: utf-8 -*-
"""
align audio and text, get time segments for each word/sentence
date: 20250422
author: liuyoude
"""
import subprocess
import tempfile
import os
from abc import ABC, abstractmethod
from typing import Dict, Optional, List
from dataclasses import dataclass, field
from textgrid import TextGrid
from pypinyin import pinyin, Style, lazy_pinyin

@dataclass
class TimeSegment:
    start: float                # start time in seconds
    end: float                  # end time in seconds
    text: str                   # corresponding text (word/sentence)
    type: str = "word"          # segment type (word/sentence)
    # type: str = field(          # segment type (word/sentence)
    #     default="word",
    #     metadata={"valid_values": ["word", "sentence"]}
    # )

# sentence end char sets
SENTENCE_END_CHARS = {
    '。', '！', '？', '…',  # chinese punctuation
    '.', '!', '?', ':', ';'  # english punctuation
}

class BaseAligner(ABC):
    @abstractmethod
    def align(self, audio_path: str, text: Optional[str] = None) -> List[TimeSegment]:
        raise NotImplementedError

class AlignmentError(Exception):
    pass
    
class WhisperAligner(BaseAligner):
    """align audio and text when no text is provided"""
    def __init__(self, model_size="medium", device="cpu"):
        import whisper
        self.model = whisper.load_model(model_size, device=device)
    
    def align(self, audio_path, text=None):
        result = self.model.transcribe(audio_path, 
                                       word_timestamps=True,
                                       temperature=0.0,
                                       condition_on_previous_text=False,
                                    #    logprob_threshold=-0.5,
                                    #    no_speech_threshold=0.8,
                                    #    hallucination_silence_threshold=0.1
                                       )
        print(result["text"])
        return self._convert_format(result["segments"])

    def _convert_format(self, whisper_segments):
        segments = []
        for seg in whisper_segments:
            # word level segments
            for word in seg["words"]:
                segments.append(TimeSegment(
                    start=word["start"],
                    end=word["end"],
                    text=word["word"].strip(),
                    type="word",
                ))
            # sentence level segments
            segments.append(TimeSegment(
                start=seg["start"],
                end=seg["end"],
                text=seg["text"],
                type="sentence",
            ))
        return segments

class MFAAligner(BaseAligner):
    """ align audio and text using MFA """
    def __init__(self, lang: str = "english"):
        self.lang = lang
        if lang == "english" or lang == "en":
            self.acoustic_model = "english_mfa"
            self.dictionary = "english_mfa"
        elif lang == "chinese" or lang == "zh":
            self.acoustic_model = 'core/audio_aligner/mfa_model/mandarin_acoustic'
            self.dictionary = 'core/audio_aligner/mfa_model/mandarin_pinyin.dict'
        else:
            raise ValueError(f"Unsupported language: {lang}")
        
        self._check_mfa_installed()
    
    def _check_mfa_installed(self):
        try:
            subprocess.run(["mfa", "--help"], check=True, capture_output=True)
        except FileNotFoundError:
            raise FileNotFoundError("MFA is not installed. Please install MFA first: conda install -c conda-forge montreal-forced-aligner.")

    def align(self, audio_path: str, text: str) -> List[TimeSegment]:
        if not os.path.exists(audio_path):
            raise FileNotFoundError(f"[ERROR] audio file {audio_path} not found")
        file_name = os.path.basename(audio_path).split('.')[0]
        with tempfile.TemporaryDirectory() as tmp_dir:
            # Step 1: prepare input files
            text_path = self._prepare_inputs(tmp_dir, text)
            # Step 2: run MFA alignment
            self._run_mfa_alignment(audio_path, text_path, tmp_dir)
            # Step 3: parse TextGrid results
            word_segments = self._parse_textgrid(os.path.join(tmp_dir, f"{file_name}.TextGrid"))
            # Step 4: merge word segments to sentence segments
            return self._merge_to_sentences(word_segments, original_text=text)

    def _prepare_inputs(self, tmp_dir: str, text: str) -> str:
        if self.lang == "chinese" or self.lang == "zh":
            # For Chinese, use pinyin as the text input for MFA
            text = self._zh_to_pinyin(text)
        text_path = os.path.join(tmp_dir, "transcript.txt")
        with open(text_path, 'w', encoding='utf-8') as f:
            f.write(text)
        return text_path
    
    def _zh_to_pinyin(self, text: str) -> str:
        """Convert Chinese text to pinyin with tone numbers"""
        return ' '.join(
            lazy_pinyin(
                text,
                style=Style.TONE3,  # use number to represent tone, e.g. zhong1
                neutral_tone_with_five=True, # use 5 to represent neutral tone
            )   
        )

    def _run_mfa_alignment(self, audio_path: str, text_path: str, tmp_dir: str):
        """MFA cmd: mfa align_one --clean --overwrite audio_path text_path dictionary acoustic_model tmp_dir --beam 50"""
        cmd = [
            "mfa", "align_one", "--clean", "--overwrite",
            audio_path,
            text_path,
            self.dictionary,
            self.acoustic_model,
            tmp_dir,
            "--beam", "50"
        ]
        try:
            subprocess.run(cmd, check=True, capture_output=True)
        except subprocess.CalledProcessError as e:
            raise AlignmentError(f"[ERROR] MFA align error: {e}")

    def _parse_textgrid(self, tg_path: str) -> List[TimeSegment]:
        if not os.path.exists(tg_path):
            raise FileNotFoundError(f"[ERROR] MFA align result {tg_path} not found")

        tg = TextGrid.fromFile(tg_path)
        word_tier = next(t for t in tg.tiers if t.name == "words")
        
        return [
            TimeSegment(
                start=interval.minTime,
                end=interval.maxTime,
                text=interval.mark.strip(),
                type="word"
            )
            for interval in word_tier.intervals if (interval.mark.strip() and interval.mark.strip() != "<eps>")
        ]

    def _merge_to_sentences(self, word_segments: List[TimeSegment], original_text: str) -> List[TimeSegment]:
        # 重建原始文本结构
        sentences = []
        current_sentence = []
        if self.lang == "chinese" or self.lang == "zh":
            original_words = self._zh_to_pinyin(original_text).split()
        else:
            original_words = original_text.split()
        # merge punctuation marks to words
        processed_words = []
        for word in original_words:
            if processed_words and all(c in SENTENCE_END_CHARS for c in word):
                processed_words[-1] += word
            else:
                processed_words.append(word)
        original_words = processed_words
        # check word segments and original text match
        if len(word_segments) != len(original_words):
            raise AlignmentError("[ERROR] MFA align result and original text do not match")
        # split sentences based on punctuation marks
        for word_seg, orig_word in zip(word_segments, original_words):
            current_sentence.append(word_seg)
            # check end of word match original text
            if any(c in SENTENCE_END_CHARS for c in orig_word):
                sentences.append(self._create_sentence_segment(current_sentence))
                current_sentence = []
        if current_sentence:
            sentences.append(self._create_sentence_segment(current_sentence))
        return word_segments + sentences

    def _create_sentence_segment(self, word_segments: List[TimeSegment]) -> TimeSegment:
        return TimeSegment(
            start=word_segments[0].start,
            end=word_segments[-1].end,
            text=' '.join([ws.text for ws in word_segments]),
            type="sentence",
        )
    
def plot_alignment(audio_path, segments):
    import matplotlib.pyplot as plt
    import numpy as np
    import librosa
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 指定全局中文字体
    plt.rcParams['axes.unicode_minus'] = False    # 解决负号显示为方块的问题

    audio, sr = librosa.load(audio_path, sr=None, mono=True)
    plt.figure(figsize=(15, 3))
    
    plt.plot(np.linspace(0, len(audio)/sr, len(audio)), audio)
    
    # tag segments
    colors = {'word': 'red', 'sentence': 'blue', 'segment': 'green'}
    for seg in segments:
        if seg.type == 'word':
            plt.axvspan(seg.start, seg.end, alpha=0.2, color=colors[seg.type])
            plt.text(seg.start, 0, seg.text, rotation=45)
        else:
            plt.axvline(seg.start, color=colors[seg.type], linestyle='--')
            plt.axvline(seg.end, color=colors[seg.type], linestyle='--')
            plt.text(seg.start + 0.1, -0.5, seg.text, rotation=0)
    
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.show()

if __name__ == "__main__":
    # model = whisper.load_model("turbo") # turbo tiny
    file_path = "D:\Project\TTS\\test.wav"
    # file_path = r"D:\Project\TTS\speech_text_render\core\mfa_test\audio\LJ037_0171_test.wav"
    aligner = WhisperAligner(model_size="turbo", device="cpu")
    segments = aligner.align(file_path)
    for seg in segments:
        print(f"[{seg.start:.2f}-{seg.end:.2f}] {seg.text} (type: {seg.type})") 
    plot_alignment(file_path, segments)

    # file_path = r"D:\Project\TTS\speech_text_render\core\mfa_test\audio\LJ037_0171_test.wav"
    # ori_text = "The examination and testimony of the experts enabled the Commission to conclude that five shots may have been fired."
    # file_path1 = "D:\Project\TTS\\test.wav"
    # ori_text1 = "现在是测试VCIPER的音频。它能分辨出这个分段吗？真的吗？可以吗？嗯？"
    # aligner = MFAAligner(lang="english")
    # aligner_zh = MFAAligner(lang="chinese")
    # # segments = aligner._parse_textgrid("D:\Project\TTS\speech_text_render\core\LJ037_0171_test.TextGrid")
    # # ori_text = "The examination and testimony of the experts enabled the Commission to conclude that five shots may have been fired."
    # # segments = aligner._merge_to_sentences(segments, original_text=ori_text)

    # segments_en = aligner.align(file_path, text="The examination and testimony of the experts enabled the Commission to conclude that five shots may have been fired.")
    # segments_zh = aligner_zh.align(file_path1, text=ori_text1)
    # for seg in segments_en:
    #     print(f"[{seg.start:.2f}-{seg.end:.2f}] {seg.text} (type: {seg.type})")
    # for seg in segments_zh:
    #     print(f"[{seg.start:.2f}-{seg.end:.2f}] {seg.text} (type: {seg.type})")
    # plot_alignment(file_path, segments_en)
    # plot_alignment(file_path1, segments_zh)