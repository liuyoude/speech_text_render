# -*- coding: utf-8 -*-
"""
align audio and text, get time segments for each word/sentence
date: 20250422
author: liuyoude
"""
import os
import sys
import subprocess
import tempfile
from abc import ABC, abstractmethod
from typing import Dict, Optional, List
from dataclasses import dataclass, field
from textgrid import TextGrid
from pypinyin import pinyin, Style, lazy_pinyin
import whisper
import torch
import librosa
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from core.audio_aligner.text_normalizer import TextNormalizer, SENTENCE_END_CHARS

@dataclass
class TimeSegment:
    start: float                # start time in seconds
    end: float                  # end time in seconds
    text: str                   # corresponding text (word/sentence)
    type: str = "word"          # segment type (word/sentence/paragraph)
    # type: str = field(          # segment type (word/sentence)
    #     default="word",
    #     metadata={"valid_values": ["word", "sentence"]}
    # )

class BaseAligner(ABC):
    @abstractmethod
    def align(self, audio_path: str, text: Optional[str] = None) -> List[TimeSegment]:
        raise NotImplementedError

class AlignmentError(Exception):
    pass
    
class WhisperAligner(BaseAligner):
    """align audio and text when no text is provided"""
    def __init__(self, model_size="medium", device="cpu"):
        self.device = torch.device(device)
        self.model = whisper.load_model(model_size, 
                                        device=self.device,
                                        in_memory=True).eval()
        self.text_normalizer = TextNormalizer()
        if self.device.type == "cuda":
            torch.backends.cudnn.benchmark = True
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            self.model = torch.compile(self.model) 
    
    def align(self, audio_path, text=None):
        audio = whisper.load_audio(audio_path)
        result = self.model.transcribe(audio,
                                       word_timestamps=True,
                                       temperature=0.0,
                                       condition_on_previous_text=False,
                                       fp16=True if self.device.type == "cuda" else False,
                                       initial_prompt="中文情况下的中文文字请使用简体中文",
                                    #    logprob_threshold=-0.5,
                                    #    no_speech_threshold=0.8,
                                    #    hallucination_silence_threshold=0.1
                                        )
        # print(result["text"])
        segments = self._convert_format(result)
        if text:
            # correct Error in whisper segments using original text
            try:
                segments = self._correct_text_Errors(text, segments)
            except AlignmentError as e:
                print(e)
                print("Skip Error correction")
        return segments

    def _convert_format(self, whisper_result):
        """convert whisper format to TimeSegment format"""
        whisper_segments = whisper_result['segments']
        segments = []
        # paragraph level segments
        segments.append(TimeSegment(
            start=whisper_segments[0]["start"],
            end=whisper_segments[-1]["end"],
            text=whisper_result["text"],
            type="paragraph", 
        ))
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
    
    def _correct_text_Errors(self, original_text: str, 
                            whisper_segments: List[TimeSegment]) -> List[TimeSegment]:
        """modify Error in whisper segments using original text"""
        original_normalized = self.text_normalizer.normalize(original_text)
        whisper_normalized = self.text_normalizer.normalize(whisper_segments[0].text)
        # TODO: need better alignment method for Error correction
        # 基础校验
        if len(original_normalized.split()) == len(whisper_normalized.split()):
            original_words = original_normalized.split()
            # replace Error in whisper segments using original text
            word_idx = 0
            sentence = []
            for seg_idx, seg in enumerate(whisper_segments):
                if seg.type == "word":
                    # consider muiltiple words predicted by whisper in word type segment
                    skip_len = len(self.text_normalizer.normalize(seg.text).split())
                    text = self.text_normalizer.denormalize(' '.join(original_words[word_idx: word_idx + skip_len]))
                    whisper_segments[seg_idx].text = text
                    sentence.append(text)
                    word_idx += skip_len
                elif seg.type == "sentence":
                    whisper_segments[seg_idx].text = self.text_normalizer.denormalize(' '.join(sentence))
                    sentence = []
            whisper_segments[0].text = original_text            
            return whisper_segments
            
        raise AlignmentError(f"[Error] input text: <{original_text}> and speech content length: <{whisper_segments[0].text}> do not match")

class MFAAligner(BaseAligner):
    """ align audio and text using MFA """
    def __init__(self, lang: str = "english"):
        if lang == "english" or lang == "en":
            self.lang = "english"
            self.acoustic_model = "english_mfa"
            self.dictionary = "english_mfa"
        elif lang == "chinese" or lang == "zh":
            self.lang = "chinese"
            self.acoustic_model = 'core/audio_aligner/mfa_model/mandarin_acoustic'
            self.dictionary = 'core/audio_aligner/mfa_model/mandarin_pinyin.dict'
        else:
            raise ValueError(f"Unsupported language: {lang}")
        
        self.text_normalizer = TextNormalizer()
        # self._check_mfa_installed()
    
    def _check_mfa_installed(self):
        try:
            subprocess.run(["mfa", "--help"], check=True, capture_output=True)
        except FileNotFoundError:
            raise FileNotFoundError("[Error] MFA is not installed. Please install MFA first: conda install -c conda-forge montreal-forced-aligner.")

    def align(self, audio_path: str, text: str) -> List[TimeSegment]:
        if not os.path.exists(audio_path):
            raise FileNotFoundError(f"[Error] audio file <{audio_path}> not found")
        file_name = os.path.basename(audio_path).split('.')[0]
        with tempfile.TemporaryDirectory() as tmp_dir:
            # Step 1: prepare input files
            audio_path, text_path = self._prepare_inputs(tmp_dir, audio_path, text)
            # Step 2: run MFA alignment
            self._run_mfa_alignment(audio_path, text_path, tmp_dir)
            # Step 3: parse TextGrid results
            word_segments = self._parse_textgrid(os.path.join(tmp_dir, f"{file_name}.TextGrid"))
            # Step 4: merge word segments to sentence segments
            return self._merge_to_sentences(word_segments, original_text=text)

    def _prepare_inputs(self, tmp_dir: str, audio_path: str, text: str) -> list[str]:
        # resample audio to 16kHz if needed
        # audio, sr = librosa.load(audio_path, sr=16000, mono=True)
        # audio_path = os.path.join(tmp_dir, "audio.wav")
        # sf.write(audio_path, audio, sr, format='WAV')
        if self.lang == "chinese" or self.lang == "zh":
            # For Chinese, use pinyin as the text input for MFA
            text = self._zh_to_pinyin(text)
        text_path = os.path.join(tmp_dir, "transcript.txt")
        with open(text_path, 'w', encoding='utf-8') as f:
            f.write(text)
        return audio_path, text_path
    
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
            raise AlignmentError(f"[Error] MFA align Error: {e}")

    def _parse_textgrid(self, tg_path: str) -> List[TimeSegment]:
        if not os.path.exists(tg_path):
            raise FileNotFoundError(f"[Error] MFA align result {tg_path} not found")

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
        current_sentence = []
        original_words = self.text_normalizer.normalize(original_text).split()
        # check word segments and original text match
        if len(word_segments) != len(original_words):
            raise AlignmentError("[Error] MFA align result and original text do not match")
        # split sentences based on punctuation marks
        segments = [TimeSegment(
            start=word_segments[0].start,
            end=word_segments[-1].end,
            text=original_text,
            type="paragraph",
        )]
        for idx, (word_seg, orig_word) in enumerate(zip(word_segments, original_words)):
            word_seg.text = orig_word
            segments.append(word_seg)
            current_sentence.append(word_seg)
            # check end of word match original text
            if any(c in SENTENCE_END_CHARS for c in orig_word):
                segments.append(self._create_sentence_segment(current_sentence))
                current_sentence = []
        if current_sentence:
            segments.append(self._create_sentence_segment(current_sentence))
        return segments

    def _create_sentence_segment(self, word_segments: List[TimeSegment]) -> TimeSegment:
        text = ' '.join([ws.text for ws in word_segments])
        text = self.text_normalizer.denormalize(text)
        return TimeSegment(
            start=word_segments[0].start,
            end=word_segments[-1].end,
            text=text,
            type="sentence",
        )
    
class SpeechTextAligner:
    """ align audio and text using MFA """
    def __init__(self, device="cpu"):
        self.whisper_aligner = WhisperAligner(model_size="turbo", device=device)
        self.mfa_aligner_en = MFAAligner(lang="english")
        self.mfa_aligner_zh = MFAAligner(lang="chinese")

    def align(self, audio_path: str, text: str = None, lang: str = None) -> List[TimeSegment]:
        if text is None:
            return self.whisper_aligner.align(audio_path)
        else:
            if lang == "english" or lang == "en":
                return self.mfa_aligner_en.align(audio_path, text)
            elif lang == "chinese" or lang == "zh":
                return self.mfa_aligner_zh.align(audio_path, text)
            elif lang is None:
                raise ValueError("Please specify the language of the text")
            else:
                raise ValueError(f"Unsupported language: {lang}")
    
def plot_alignment(audio_path, segments, save_path=None):
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
        elif seg.type == 'sentence':
            plt.axvline(seg.start, color=colors[seg.type], linestyle='--')
            plt.axvline(seg.end, color=colors[seg.type], linestyle='--')
            plt.text(seg.start + 0.1, -0.5, seg.text, rotation=0)
    
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    if save_path is not None:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()

if __name__ == "__main__":
    file_path_en = r"examples/audios/en/LJ001-0001.wav"
    ori_text_en = "Printing, in the only sense with which we are at present concerned, differs from most if not from all the arts and crafts represented in the Exhibition?"
    file_path_zh = r"examples/audios/zh/D4_750.wav"
    ori_text_zh = "苏北军的一些爱国将士，马战山、李渡、唐巨武、苏炳爱、邓铁梅等也奋起抗战。"

    _, sr_zh = librosa.load(file_path_zh, sr=None, mono=True)
    _, sr_en = librosa.load(file_path_en, sr=None, mono=True)
    print(f"sr_zh: {sr_zh}, sr_en: {sr_en}")

    whisper_aligner = WhisperAligner(model_size="medium", device="cpu")
    # # no text provided, use whisper to align audio
    # segments = whisper_aligner.align(file_path_en)
    # for seg in segments:
    #     print(f"[{seg.start:.2f}-{seg.end:.2f}] {seg.text} (type: {seg.type})") 
    # plot_alignment(file_path_en, segments)    

    # segments = whisper_aligner.align(file_path_zh)
    # for seg in segments:
    #     print(f"[{seg.start:.2f}-{seg.end:.2f}] {seg.text} (type: {seg.type})") 
    # plot_alignment(file_path_zh, segments) 

    # text provided for Error correction
    # segments = whisper_aligner.align(file_path_en, text=ori_text_en)
    # for seg in segments:
    #     print(f"[{seg.start:.2f}-{seg.end:.2f}] {seg.text} (type: {seg.type})")
    # plot_alignment(file_path_en, segments)

    # segments = whisper_aligner.align(file_path_zh, text=ori_text_zh)
    # for seg in segments:
    #     print(f"[{seg.start:.2f}-{seg.end:.2f}] {seg.text} (type: {seg.type})")
    # plot_alignment(file_path_zh, segments)

    mfa_aligner_en = MFAAligner(lang="english")
    mfa_aligner_zh = MFAAligner(lang="chinese")

    # segments_zh = mfa_aligner_zh.align(file_path_zh, text=ori_text_zh)
    # for seg in segments_zh:
    #     print(f"[{seg.start:.2f}-{seg.end:.2f}] {seg.text} (type: {seg.type})")
    # plot_alignment(file_path_zh, segments_zh)

    segments_en = mfa_aligner_en.align(file_path_en, text=ori_text_en)
    for seg in segments_en:
        print(f"[{seg.start:.2f}-{seg.end:.2f}] {seg.text} (type: {seg.type})")
    plot_alignment(file_path_en, segments_en)    