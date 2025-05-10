# -*- coding: utf-8 -*-
"""
align audio and text, get time segments for each word/sentence
date: 20250422
author: liuyoude
"""
import os
import sys
import re
import subprocess
import tempfile
from abc import ABC, abstractmethod
from typing import Dict, Optional, List, Tuple
from dataclasses import dataclass, field
from textgrid import TextGrid
from pypinyin import pinyin, Style, lazy_pinyin
import whisper
import torch
import librosa
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from core.audio_aligner.text_normalizer import TextNormalizer, SENTENCE_END_CHARS, is_punctuation

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
        # only support 16k sample rate, 30s max length
        if len(audio) > 16000 * 30:
            raise AlignmentError(f"[Error] audio file: {audio_path} is too long")
        result = self.model.transcribe(audio,
                                       word_timestamps=True,
                                       temperature=0.0,
                                       condition_on_previous_text=False,
                                       fp16=True if self.device.type == "cuda" else False,
                                    #    initial_prompt="中文情况下的中文文字请使用简体中文",
                                    #    logprob_threshold=-0.5,
                                    #    no_speech_threshold=0.1,
                                       hallucination_silence_threshold=0.1
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

    def detect_language(self, audio_path):
        audio = whisper.load_audio(audio_path)
        audio = whisper.pad_or_trim(audio)
        mel = whisper.log_mel_spectrogram(audio).to(self.model.device)
        _, probs = self.model.detect_language(mel)
        return max(probs, key=probs.get)

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
                text = word["word"].strip()
                # chinese may have multiple words in one segment
                is_chinese = any('\u4e00' <= char <= '\u9fff' for char in text)
                if is_chinese and len(text) > 1:
                    duration = word["end"] - word["start"]
                    char_duration = duration / len(text)
                    for i, char in enumerate(text):
                        if is_punctuation(char):
                            # merge punctuation with previous word
                            segments[-1].text += char
                            segments[-1].end += char_duration
                        else:
                            segments.append(TimeSegment(
                                start=word["start"] + i*char_duration,
                                end=word["start"] + (i+1)*char_duration,
                                text=char,
                                type="word"
                            ))
                else:
                    segments.append(TimeSegment(
                        start=word["start"],
                        end=word["end"],
                        text=text,
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
        # record original text with punctuation
        original_words = self.text_normalizer.normalize(original_text).split()
        # get predicted words (keep original punctuation)
        word_segments = [seg for seg in whisper_segments if seg.type == "word"]
        # without punctuation for dynamic programming alignment
        predicted_words = [re.sub(r'[^\w]', '', w.text, flags=re.UNICODE).lower() for w in word_segments]
        
        if len(original_words) == len(predicted_words):
            # directly replace Error in whisper segments using original text 
            # for the case that the number of words is the same
            word_idx = 0
            sentence = []
            for seg_idx, seg in enumerate(whisper_segments):
                if seg.type == "word":
                    text = original_words[word_idx]
                    whisper_segments[seg_idx].text = text
                    sentence.append(text)
                    word_idx += 1
                elif seg.type == "sentence":
                    whisper_segments[seg_idx].text = self.text_normalizer.denormalize(' '.join(sentence))
                    sentence = []
            whisper_segments[0].text = original_text             
        else:
            # dynamic programming alignment 
            # for the case that the number of words is different
            aligned_pairs = list(self._dynamic_align_words(original_words, predicted_words))
            if len(aligned_pairs) == 0:
                raise AlignmentError(f"[Error] alignment failed, original text: {original_text}, predicted text: {whisper_segments[0].text}")
            # restruct time segments using original text with punctuation
            word_idx = 0
            skip_seg = 0
            sentence = []
            whisper_segments[0].text = original_text
            new_whisper_segments = [whisper_segments[0]]               
            for seg in whisper_segments:
                if seg.type == "word":
                    if skip_seg > 0:
                        skip_seg -= 1
                        continue
                    # get original word (with punctuation)
                    orig_idx, (pred_start, pred_end) = aligned_pairs[word_idx]
                    merged_time_segments = word_segments[pred_start:pred_end]
                    skip_seg = pred_end - pred_start - 1
                    # merge time segments
                    seg.text = original_words[orig_idx]
                    seg.start = min(ts.start for ts in merged_time_segments)
                    seg.end = max(ts.end for ts in merged_time_segments)
                    word_idx += 1
                    sentence.append(seg.text)
                    new_whisper_segments.append(seg)
                elif seg.type == "sentence":
                    seg.text = self.text_normalizer.denormalize(' '.join(sentence))
                    sentence = []
                    new_whisper_segments.append(seg)
            whisper_segments = new_whisper_segments
        return whisper_segments

    def _dynamic_align_words(self, original_words: List[str], 
                             predicted_words: List[str],
                             max_merge: int = 5) -> List[Tuple[int, Tuple[int, int]]]:
        """align words using dynamic programming"""
        # preprocess: remove all punctuation in words (keep characters inside words)
        clean_original = [re.sub(r'[^\w\u4e00-\u9fff]', '', w).lower() for w in original_words]
        clean_predicted = [re.sub(r'[^\w\u4e00-\u9fff]', '', w).lower() for w in predicted_words]
        len_p, len_o = len(clean_predicted), len(clean_original)
        # initialize DP table
        dp = [[float('inf')] * (len_p + 1) for _ in range(len_o + 1)]
        path = [[[] for _ in range(len_p + 1)] for _ in range(len_o + 1)]
        dp[0][0] = 0
        # fill DP table
        for i in range(1, len(clean_original)+1):
            for j in range(1, len(clean_predicted)+1):
                # single word match
                if clean_predicted[j-1] == clean_original[i-1]:
                    if dp[i-1][j-1] < dp[i][j]:
                        dp[i][j] = dp[i-1][j-1]
                        path[i][j] = path[i-1][j-1] + [(i-1, j-1, j)] 
                    continue               
                # adaptive merging strategy
                # skip chinese word
                if any('\u4e00' <= c <= '\u9fff' for c in clean_original[i-1]):
                    continue
                for k in range(1, min(j, max_merge)+1):
                    merged = ''.join(clean_predicted[j-k:j])
                    original_word = clean_original[i-1]
                    containment_cost = 0 if merged in original_word else 1
                    length_ratio = abs(len(merged) - len(original_word)) / len(original_word)
                    common_chars = len(set(merged) & set(original_word))
                    char_match_ratio = common_chars / max(len(set(merged)), len(set(original_word)), 1)
                    cost = 0.4 * containment_cost + 0.3 * length_ratio + 0.3 * (1 - char_match_ratio)
                    if dp[i-1][j-k] + cost < dp[i][j]:
                        dp[i][j] = dp[i-1][j-k] + cost
                        path[i][j] = path[i-1][j-k] + [(i-1, j-k, j)]

        if len(clean_original) == 0 and len(clean_predicted) > 0:
            return [(0, (0, len(clean_predicted)))]
        # traceback path to get alignment results
        alignment = []
        i, j = len(clean_original), len(clean_predicted)
        while i > 0 and j > 0:
            step = path[i][j][-1]
            orig_idx = step[0]
            pred_start = step[1]
            pred_end = step[2]
            alignment.append( (orig_idx, (pred_start, pred_end)) )
            i, j = orig_idx, pred_start       
        return reversed(alignment)

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
            # self.acoustic_model = 'core/audio_aligner/mfa_model/aishell3_model'
            # self.dictionary = 'core/audio_aligner/mfa_model/simple.dict'
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
            "--beam", "100"
        ]
        try:
            subprocess.run(cmd, check=True, capture_output=True)
        except subprocess.CalledProcessError as e:
            raise AlignmentError(f"[Error] MFA align Error: {e.stderr.decode('utf-8')}")

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
        self.whisper_aligner = WhisperAligner(model_size="small", device=device)
        self.mfa_aligner_en = MFAAligner(lang="english")
        self.mfa_aligner_zh = MFAAligner(lang="chinese")

    def align(self, audio_path: str, text: str = None, lang: str = None) -> List[TimeSegment]:
        if text is None:
            return self.whisper_aligner.align(audio_path)
        else:
            if lang is None:
                lang = self.whisper_aligner.detect_language(audio_path)
                print(f"[Info] detected language: {lang}")
            if lang == "english" or lang == "en":
                return self.mfa_aligner_en.align(audio_path, text)
            elif lang == "chinese" or lang == "zh":
                # return self.mfa_aligner_zh.align(audio_path, text)
                return self.whisper_aligner.align(audio_path, text)
            else:
                raise ValueError(f"Unsupported language: {lang}")
    
def plot_alignment(audio_path, segments, save_path=None):
    import matplotlib.pyplot as plt
    import numpy as np
    import librosa
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 指定全局中文字体
    plt.rcParams['axes.unicode_minus'] = False    # 解决负号显示为方块的问题

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
    # 绘制对数功率谱（时间对齐）
    img = librosa.display.specshow(S_db, sr=sr, x_axis='time', 
                          y_axis='mel', ax=ax2,
                          hop_length=hop_length,
                          cmap='viridis')
    # plt.colorbar(img, ax=ax2, format="%+2.0f dB")
    ax2.set(title='Log-Power Spectrogram')
    
    # tag segments
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
    # file_path_en = r"examples/audios/en/LJ001-0001.wav"
    # ori_text_en = "Printing, in the only sense with which we are at present concerned, differs from most if not from all the arts and crafts represented in the Exhibition?"

    file_path_en = r"examples/audios/en/LJ001-0005.wav"
    ori_text_en = "the invention of movable metal letters in the middle of the fifteenth century may justly be considered as the invention of the art of printing."

    # file_path_zh = r"examples/audios/zh/D4_750.wav"
    # ori_text_zh = "苏北军的一些爱国将士，马战山、李渡、唐巨武、苏炳爱、邓铁梅等也奋起抗战。"

    # file_path_zh = r"examples/audios/zh/D4_754.wav"
    # ori_text_zh = "由太原市南郊区寇庄村农民投资数百万元建设的平阳集贸市场，因管理等诸多方面的原因已停业一年。"

    file_path_zh = r"examples/audios/zh_en/zh_en_test_0001.wav"
    ori_text_zh = "现在是测试VCIPER的音频。它能分辨出这个分段吗？真的吗？可以吗？嗯？"

    _, sr_zh = librosa.load(file_path_zh, sr=None, mono=True)
    _, sr_en = librosa.load(file_path_en, sr=None, mono=True)
    print(f"sr_zh: {sr_zh}, sr_en: {sr_en}")

    whisper_aligner = WhisperAligner(model_size="small", device="cpu")

    # text_normalizer = TextNormalizer()
    # ori_text = "the invention of movable metal letters in the middle of the fifteenth century may justly be considered as the invention of the art of printing."
    # pred_text = "the invention of movable metal letters in the middle of the 15th century may justly be considered as the invention of the art of printing."
    # ori_text_norm = text_normalizer.normalize(ori_text).split()
    # pred_text_norm = text_normalizer.normalize(pred_text).split()
    ori_text_norm = ['现', '在', '是', '测', '试', 'vciper', '的', '音', '频', '它', '能', '分', '辨', '出', '这', '个', '分', '段', '吗', '真', '的', '吗', '可', '以', '吗', '嗯']
    pred_text_norm = ['现', '在', '是', '测', '试', 'v', 'is', 'ible', '的', '音', '频', '它', '能', '分', '辨', '出', '这', '个', '分', '段', '吗', '真', '的', '吗', '可', '以', '吗', '嗯']
    # print(ori_text_norm)
    # print(pred_text_norm)
    for res in whisper_aligner._dynamic_align_words(ori_text_norm, pred_text_norm):
        print(ori_text_norm[res[0]], pred_text_norm[res[1][0]:res[1][1]])

    # no text provided, use whisper to align audio
    # segments = whisper_aligner.align(file_path_en, text=ori_text_en)
    # for seg in segments:
    #     print(f"[{seg.start:.2f}-{seg.end:.2f}] {seg.text} (type: {seg.type})") 
    # plot_alignment(file_path_en, segments)    

    # segments = whisper_aligner.align(file_path_zh, text=ori_text_zh)
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

    # segments_en = mfa_aligner_en.align(file_path_en, text=ori_text_en)
    # for seg in segments_en:
    #     print(f"[{seg.start:.2f}-{seg.end:.2f}] {seg.text} (type: {seg.type})")
    # plot_alignment(file_path_en, segments_en)    
