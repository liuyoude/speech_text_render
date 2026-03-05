# -*- coding: utf-8 -*-
"""
volume control extractor
date: 20250704
author: liuyoude
"""
import librosa
import numpy as np
from typing import Dict, Optional, List, Tuple
from core.feature_extractor.base_extractor import BaseExtractor, TimeSegment
from core.utils.audio_utils import calculate_perceptual_energy

class VolumeExtractor(BaseExtractor):
    def __init__(self, config):
        super().__init__(config)
        self.type = 'volume'
        # 默认配置
        # self.default_volume_levels = {
        #     'whisper': (0.0, 0.4),    # 轻声/耳语
        #     'soft': (0.4, 0.7),       # 柔和说话
        #     'normal': (0.7, 1.3),     # 正常音量
        #     'strong': (1.3, 1.8),     # 强调/重读
        #     'shout': (1.8, float('inf'))  # 喊叫/愤怒
        # }
        # 字级配置
        self.default_word_volume_levels = {
            'whisper': (0.0, 0.2),    # < 60% 平均音量
            'normal': (0.2, 3),     # 60%-140% 视为正常范围
            'strong': (3, 5),     # 140%-200%
            'shout': (5, float('inf'))
        }
        self.default_word_duration_threshold = 0.5
        # 句级配置
        self.default_sentence_volume_levels = {
            'whisper': (0.0, 0.8),    # < 60% 平均音量
            'normal': (0.8, 1.25),     # 60%-140% 视为正常范围
            'strong': (1.25, 2.5),     # 140%-200%
            'shout': (2.5, float('inf'))
        }
        self.default_sentence_duration_threshold = 1.0
        # self.default_word_threshold = 0.3  # 字级别相对音量变化阈值
        # self.default_sentence_threshold = 0.2  # 句级别相对音量变化阈值
        self.target_frame_ms = 30  # 目标帧长度30ms
        self.target_hop_ms = 10    # 目标跳步长度10ms
        
        # 应用配置
        self.number_control = self.config.get('number_control', False)
        self.word_volume_levels = self.config.get('word_volume_levels', self.default_word_volume_levels)
        self.word_duration_threshold = self.config.get('word_duration_threshold', self.default_word_duration_threshold)
        self.sentence_volume_levels = self.config.get('sentence_volume_levels', self.default_sentence_volume_levels)
        self.sentence_duration_threshold = self.config.get('sentence_duration_threshold', self.default_sentence_duration_threshold)
        # self.word_threshold = self.config.get('word_threshold', self.default_word_threshold)
        # self.sentence_threshold = self.config.get('sentence_threshold', self.default_sentence_threshold)
        
        # 初始化librosa缓存
        self.audio_cache = {}
        
    def load_model(self) -> None:
        # 无需额外模型加载
        pass
        
    def extract(self, audio_path: str, time_segments: List[TimeSegment], lang: str = None) -> List[Dict]:
        """
        提取相对音量控制符
        基于上下文动态计算音量级别
        """
        # 加载音频并计算RMS能量
        y, sr = self._load_audio(audio_path)
        frame_length, hop_length = self._calculate_frame_params(sr)
        # rms = self._calculate_rms(y, sr, frame_length, hop_length)
        rms = calculate_perceptual_energy(y, sr, frame_length, hop_length)

        # self.plot_volume_analysis(audio_path, rms, sr, hop_length, time_segments)
        
        volume_controls = []
        
        # 计算全局平均音量作为基准
        # global_avg_vol = np.mean(rms)
        
        # 按句子分组处理
        sentence_rmss = []
        sentence_groups = self._group_by_sentence(time_segments)
        for sentence_idx, (sentence_seg, word_segments) in enumerate(sentence_groups):
            sentence_rms = self._get_segment_rms(rms, sr, hop_length, sentence_seg.start, sentence_seg.end)
            sentence_rmss.append(sentence_rms)
        global_avg_vol = np.mean(np.concatenate(sentence_rmss))
        seg_idx = 0
        for sentence_idx, (sentence_seg, word_segments) in enumerate(sentence_groups):
            if sentence_seg.end - sentence_seg.start < self.sentence_duration_threshold:
                seg_idx += len(word_segments) + 1
                continue
            # 计算句子平均音量
            # sentence_rms = self._get_segment_rms(rms, sr, hop_length, sentence_seg.start, sentence_seg.end)
            sentence_rms = sentence_rmss[sentence_idx]
            sentence_avg_vol = np.mean(sentence_rms) if len(sentence_rms) > 0 else global_avg_vol
            
            # 确定句子级别音量类型
            sentence_vol_ratio = sentence_avg_vol / global_avg_vol
            sentence_vol_type = self._get_volume_level(sentence_vol_ratio, self.sentence_volume_levels)
            
            # 添加句子级别控制符（如果明显不同于正常）
            # if abs(sentence_vol_ratio - 1.0) > self.sentence_threshold:
            if sentence_vol_type != 'normal':
                volume_controls.append({
                    "type": self.type,
                    "value": round(sentence_vol_ratio, 2) if self.number_control else sentence_vol_type,
                    "pos": seg_idx,
                    "info": f"[{sentence_seg.text}]relative volume ratio={sentence_vol_ratio:.2f}",
                })

            # 处理句子内的单词级别音量
            for word_seg in word_segments:
                if word_seg.end - word_seg.start < self.word_duration_threshold:
                    seg_idx += 1
                    continue
                # 计算单词级别音量
                word_rms = self._get_segment_rms(rms, sr, hop_length,word_seg.start, word_seg.end)
                word_avg_vol = np.mean(word_rms) if len(word_rms) > 0 else sentence_avg_vol

                # 计算相对于句子平均音量的比例
                word_vol_ratio = word_avg_vol / sentence_avg_vol
                # word_vol_ratio = word_avg_vol / global_avg_vol
                word_vol_type = self._get_volume_level(word_vol_ratio, self.word_volume_levels)
                
                # 仅当音量明显变化时才添加控制符
                # if abs(word_vol_ratio - 1.0) > self.word_threshold:
                if word_vol_type != 'normal':
                    volume_controls.append({
                        "type": self.type,
                        "value": round(word_vol_ratio, 2) if self.number_control else word_vol_type,
                        "pos": seg_idx,
                        "info": f"[{word_seg.text}] duration={(word_seg.end-word_seg.start):.3f}s, relative volume ratio={word_vol_ratio:.2f}",
                    })
                seg_idx += 1
            # skip sentence after the last word
            seg_idx += 1
        
        return volume_controls
    
    def plot_volume_analysis(self, audio_path: str, perceptual_energy: np.ndarray, 
                            sr: int, hop_length: int, time_segments: List[TimeSegment]):
        """可视化音频波形与能量曲线，标注句子/单词分段"""
        import matplotlib.pyplot as plt
        import librosa.display
        
        y, _ = self._load_audio(audio_path)
        plt.figure(figsize=(15, 8))
        
        # 波形图
        ax1 = plt.subplot(2, 1, 1)
        librosa.display.waveshow(y, sr=sr, alpha=0.6)
        plt.title('Audio Waveform with Segmentation')
        plt.ylabel('Amplitude')
        
        # 能量曲线图
        ax2 = plt.subplot(2, 1, 2, sharex=ax1)
        times = librosa.times_like(perceptual_energy, sr=sr, hop_length=hop_length)
        plt.plot(times, perceptual_energy, color='r', alpha=0.8, label='Perceptual Energy')
        plt.title('Energy Curve with Volume Levels')
        plt.xlabel('Time (s)')
        plt.ylabel('Energy')
        plt.legend()
        
        # 标注时间分段
        colors = {'sentence': 'blue', 'word': 'red'}
        for seg in time_segments:
            alpha = 0.2 if seg.type == 'sentence' else 0.1
            # 在波形图上标注
            ax1.axvspan(seg.start, seg.end, alpha=alpha, color=colors.get(seg.type, 'gray'))
            # 在能量图上标注
            ax2.axvline(seg.start, color='green', linestyle='--', alpha=0.5)
            ax2.axvline(seg.end, color='purple', linestyle='--', alpha=0.5)
            
            # 标注能量值
            seg_energy = self._get_segment_rms(perceptual_energy, sr, hop_length, seg.start, seg.end)
            avg_energy = np.mean(seg_energy) if len(seg_energy) > 0 else 0
            ax2.text(seg.start, avg_energy, 
                    f"{seg.text}:{avg_energy:.2f}", 
                    color='black', fontsize=8)

        plt.tight_layout()
        plt.show()
    
    def _load_audio(self, audio_path):
        """缓存音频加载结果避免重复IO"""
        if audio_path not in self.audio_cache:
            y, sr = librosa.load(audio_path, sr=None, mono=True)
            self.audio_cache[audio_path] = (y, sr)
        return self.audio_cache[audio_path]
    
    def _calculate_rms(self, y, sr, frame_length=2048, hop_length=512):
        """计算RMS能量"""
        return librosa.feature.rms(
            y=y, 
            frame_length=frame_length, 
            hop_length=hop_length
        )[0]

    def _get_segment_rms(self, full_rms, sr, hop_length, start_time, end_time):
        """提取指定时间段内的RMS值"""
        frame_rate = sr / hop_length  # hop_length=512时的帧率
        start_frame = int(start_time * frame_rate)
        end_frame = int(end_time * frame_rate)
        return full_rms[start_frame:end_frame]
    
    def _get_volume_level(self, ratio: float, volume_levels: dict) -> str:
        """根据音量比例确定音量级别"""
        for level, (low, high) in volume_levels.items():
            if low <= ratio < high:
                return level
        return list(volume_levels.keys())[-1]  # 默认返回最高级别
    
    def _group_by_sentence(self, segments):
        """将时间分段按句子分组"""
        sentences = []
        current_words = []
        
        for seg in segments:
            if seg.type == 'word':
                current_words.append(seg)
            elif seg.type == 'sentence':
                sentences.append((seg, current_words))
                current_words = []

        return sentences
    
    def _calculate_frame_params(self, sr):
        """根据采样率计算合适的帧参数"""
        # 计算目标帧长度（采样点数）
        frame_length = int(self.target_frame_ms * sr / 1000)
        hop_length = int(self.target_hop_ms * sr / 1000)
        
        # 确保帧长度是2的幂（优化FFT计算）
        frame_length = self._nearest_power_of_two(frame_length)
        
        # 确保跳步长度至少为1
        hop_length = max(1, hop_length)
    
        return frame_length, hop_length
    
    def _nearest_power_of_two(self, n):
        """找到最接近的2的幂"""
        return 2 ** int(np.log2(n) + 0.5)
    
if __name__ == '__main__':
    import torchcrepe
    import matplotlib.pyplot as plt
    import torch

    # audio_path = r"examples/audios/en/LJ001-0001.wav"
    audio_path = r"examples/audios/zh/normal.wav"
    # Load audio
    audio, sr = torchcrepe.load.audio(audio_path)
    audio = audio[0:1, :]
    duration = audio.shape[1] / sr

    n_fft, hop_length, n_mels = 512, 256, 64
    S = librosa.feature.melspectrogram(y=audio[0].numpy(), sr=sr, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels)
    S_db = librosa.amplitude_to_db(S, ref=np.max)

    # Here we'll use a 5 millisecond hop length
    # hop_length = int(sr / 200.)

    # Provide a sensible frequency range for your domain (upper limit is 2006 Hz)
    # This would be a reasonable range for speech
    fmin = 50
    fmax = 550

    # Select a model capacity--one of "tiny" or "full"
    model = 'tiny'

    # Choose a device to use for inference
    device = 'cuda:0'

    # Pick a batch size that doesn't cause memory errors on your gpu
    batch_size = 2048

    # Compute pitch using first gpu
    pitch = torchcrepe.predict(audio,
                            sr,
                            hop_length,
                            fmin,
                            fmax,
                            model,
                            batch_size=batch_size,
                            device=device) 
    ax1 = plt.subplot(1, 1, 1)
    # ax2 = plt.subplot(2, 1, 2)
    time_axis = np.linspace(0, duration, pitch.shape[1])
    ax1.plot(time_axis,pitch[0])

    # 绘制对数功率谱（时间对齐）
    img = librosa.display.specshow(S_db, sr=sr, x_axis='time',
                          y_axis='mel', ax=ax1,
                          hop_length=hop_length,
                          cmap='viridis')
    # plt.plot(pitch[0])
    plt.show()
    
