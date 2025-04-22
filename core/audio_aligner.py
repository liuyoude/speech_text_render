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

class BaseAligner(ABC):
    @abstractmethod
    def align(self, audio_path: str, text: Optional[str] = None) -> List[TimeSegment]:
        raise NotImplementedError
    
class WhisperAligner(BaseAligner):
    """align audio and text when no text is provided"""
    def __init__(self, model_size="medium", device="cpu"):
        import whisper
        self.model = whisper.load_model(model_size, device=device)
    
    def align(self, audio_path, text=None):
        result = self.model.transcribe(audio_path, word_timestamps=True)
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

# class MFAAligner(BaseAligner):
#     class AlignmentError(Exception):
#         pass
#     def __init__(self, lang="english", acoustic_model="english", dictionary="english"):
#         """
#         :param lang: 语言标识符
#         :param acoustic_model: MFA声学模型名称
#         :param dictionary: MFA发音词典名称
#         """
#         self.lang = lang
#         self.acoustic_model = acoustic_model
#         self.dictionary = dictionary

#     def align(self, audio_path: str, text: str) -> List[TimeSegment]:
#         """执行完整对齐流程，返回包含词级和句级的分段数据"""
#         with tempfile.TemporaryDirectory() as tmp_dir:
#             # Step 1: 准备输入文件
#             text_path = self._prepare_inputs(tmp_dir, audio_path, text)
            
#             # Step 2: 执行MFA对齐
#             self._run_mfa_alignment(tmp_dir, text_path)
            
#             # Step 3: 解析TextGrid结果
#             word_segments = self._parse_textgrid(os.path.join(tmp_dir, "aligned.TextGrid"))
            
#             # Step 4: 合并词语为句子
#             return self._merge_to_sentences(word_segments, original_text=text)

#     def _prepare_inputs(self, tmp_dir: str, audio_path: str, text: str) -> str:
#         """创建临时文本文件并返回路径"""
#         text_path = os.path.join(tmp_dir, "transcript.txt")
#         with open(text_path, 'w', encoding='utf-8') as f:
#             f.write(text)
#         return text_path

#     def _run_mfa_alignment(self, tmp_dir: str, text_path: str):
#         """执行MFA命令行对齐"""
#         cmd = [
#             "mfa", "align",
#             "--clean",  # 自动清理临时文件
#             "--overwrite",
#             audio_path,
#             text_path,
#             self.dictionary,
#             self.acoustic_model,
#             tmp_dir
#         ]
#         try:
#             subprocess.run(cmd, check=True, capture_output=True)
#         except subprocess.CalledProcessError as e:
#             raise AlignmentError(f"MFA对齐失败: {e.stderr.decode()}") from e

#     def _parse_textgrid(self, tg_path: str) -> List[TimeSegment]:
#         """解析TextGrid文件获取词级对齐"""
#         if not os.path.exists(tg_path):
#             raise FileNotFoundError(f"对齐结果文件 {tg_path} 不存在")

#         tg = TextGrid.fromFile(tg_path)
#         word_tier = next(t for t in tg.tiers if t.name == "words")
        
#         return [
#             TimeSegment(
#                 start=interval.minTime,
#                 end=interval.maxTime,
#                 text=interval.mark.strip(),
#                 type="word"
#             )
#             for interval in word_tier.intervals if interval.mark.strip()
#         ]

#     def _merge_to_sentences(self, word_segments: List[TimeSegment], original_text: str) -> List[TimeSegment]:
#         """将词级对齐结果合并为句子级"""
#         # 重建原始文本结构
#         sentences = []
#         current_sentence = []
#         original_words = original_text.split()
        
#         # 确保词序匹配
#         if len(word_segments) != len(original_words):
#             raise AlignmentError("MFA对齐结果与输入文本词数不匹配")
        
#         # 基于标点分句
#         sentence_end_chars = {'。', '!', '?', '.', '！', '？'}
#         for word_seg, orig_word in zip(word_segments, original_words):
#             current_sentence.append(word_seg)
            
#             # 检测句子结束
#             if any(c in sentence_end_chars for c in orig_word):
#                 sentences.append(self._create_sentence_segment(current_sentence))
#                 current_sentence = []
        
#         # 处理剩余词语
#         if current_sentence:
#             sentences.append(self._create_sentence_segment(current_sentence))
        
#         # 合并词级和句级结果
#         return word_segments + sentences

#     def _create_sentence_segment(self, word_segments: List[TimeSegment]) -> TimeSegment:
#         """从词级分段创建句子级分段"""
#         return TimeSegment(
#             start=word_segments[0].start,
#             end=word_segments[-1].end,
#             text=' '.join([ws.text for ws in word_segments]),
#             type="sentence",
#             confidence=min(ws.confidence for ws in word_segments)
#         )
    
def plot_alignment(audio_path, segments):
    import matplotlib.pyplot as plt
    import numpy as np
    import librosa
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 指定全局中文字体
    plt.rcParams['axes.unicode_minus'] = False    # 解决负号显示为方块的问题

    audio, sr = librosa.load(audio_path, sr=None, mono=True)
    plt.figure(figsize=(15, 3))
    
    # 绘制波形
    plt.plot(np.linspace(0, len(audio)/sr, len(audio)), audio)
    
    # 标注对齐段
    colors = {'word': 'red', 'sentence': 'blue', 'segment': 'green'}
    for seg in segments:
        if seg.type == 'word':
            plt.axvspan(seg.start, seg.end, alpha=0.2, color=colors[seg.type])
            plt.text(seg.start + 0.1, 0, seg.text, rotation=45)
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
    aligner = WhisperAligner(model_size="tiny", device="cpu")
    segments = aligner.align(file_path)
    for seg in segments:
        print(f"[{seg.start:.2f}-{seg.end:.2f}] {seg.text} (type: {seg.type})") 
    plot_alignment(file_path, segments)
