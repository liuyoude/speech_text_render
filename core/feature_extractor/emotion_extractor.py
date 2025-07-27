"""
emotion control extractor
date: 20250621
author: liuyoude
"""
import os
import sys
import numpy as np
from typing import Dict, Optional, List, Tuple
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from core.feature_extractor.base_extractor import BaseExtractor, TimeSegment

class EmotionExtractor(BaseExtractor):
    def __init__(self, config):
        super().__init__(config)
        # 默认配置
        self.type = 'emotion'
        self.default_emotion_labels = [
            'angry', 'disgusted', 'fearful', 'happy', 
            'neutral', 'other', 'sad', 'surprised', 'unknow',
        ] # emotion2vec labels
        self.default_intensity_levels = {
            'none': 0.3,
            'low': 0.5,
            'medium': 0.9,
            'high': 1.0,
        }
        
        # 从配置中获取或使用默认值
        self.emotion_labels = config.get('emotion_labels', self.default_emotion_labels)
        self.intensity_levels = config.get('intensity_levels', self.default_intensity_levels)
        self.min_confidence = config.get('min_confidence', 0.3)
        self.sentence_level = config.get('sentence_level', True)
        self.multi_emotion = config.get('multi_emotion', True)
        self.number_control = config.get('number_control', False)
        self.device = config.get('device', 'cuda')
        self.sr = config.get('sr', 16000) # sample rate
        
        self.sentence_level = "True" # not support for word level
        # 模型初始化标志
        self.model_loaded = False

    def load_model(self) -> None:
        """加载Emotion2Vec模型"""
        try:
            # 使用动态导入避免强依赖
            from funasr import AutoModel
            # model="iic/emotion2vec_base"
            # model="iic/emotion2vec_base_finetuned"
            # model="iic/emotion2vec_plus_seed"
            # model="iic/emotion2vec_plus_base"
            model_id = "iic/emotion2vec_plus_large"            
            self.model = AutoModel(
                model=model_id,
                hub="ms",  # "ms" or "modelscope" for China mainland users; "hf" or "huggingface" for other overseas users
                disable_update=True,
                device=self.device,
            )
            self.model_loaded = True
        except ImportError:
            raise RuntimeError("emotion2vec package not installed. "
                               "Please install with: pip install emotion2vec")
        except Exception as e:
            raise RuntimeError(f"Failed to load emotion2vec model: {str(e)}")

    def extract(
        self, 
        audio_path: str, 
        time_segments: List[TimeSegment], 
        lang: str = None
    ) -> List[Dict]:
        """
        提取情感控制标签
        
        参数:
            audio_path: 音频文件路径
            time_segments: 时间分段列表
            lang: 可选语言代码
            
        返回:
            情感控制标签列表
        """
        if not self.model_loaded:
            self.load_model()
            
        # 读取音频文件
        audio = self._load_audio(audio_path)
        
        # 根据粒度处理
        if self.sentence_level:
            return self._extract_sentence_level(audio, time_segments)
        else:
            return self._extract_word_level(audio, time_segments)
    
    def _extract_sentence_level(
        self, 
        audio: np.ndarray,
        time_segments: List[TimeSegment]
    ) -> List[Dict]:
        """句子级情感提取"""
        emotion_controls = []
        
        # 提取所有句子段
        # sentence_segments = [seg for seg in time_segments if seg.type == 'sentence']
        start_idx = None
        for seg_idx, segment in enumerate(time_segments):
            if segment.type == 'word':
                if start_idx is None:
                    start_idx = seg_idx
                continue
            elif segment.type == 'sentence':
                # 提取句子音频
                start_sample = int(segment.start * self.sr)
                end_sample = int(segment.end * self.sr)
                seg_audio = audio[start_sample: end_sample]
                
                # 情感预测
                emotion_output = self.model.generate(seg_audio,
                                                    output_dir=None,
                                                    granularity="utterance",
                                                    extract_embedding=False,
                                                    )
                # print(emotion_output)
                
                # 解析结果
                emotion_str = self._parse_emotion_output(emotion_output[0])
                
                # 创建控制标签
                if emotion_str:
                    emotion_controls.append({
                        "type": self.type,
                        "value": emotion_str,
                        "pos": start_idx,
                        "info": f"[{segment.text}] emo={emotion_str}"
                    })
                start_idx = None
        
        return emotion_controls
    
    def _extract_word_level(
        self, 
        audio: np.ndarray,
        time_segments: List[TimeSegment]
    ) -> List[Dict]:
        """词级情感提取（适用于强调词）现有模型emotion2vec不支持词级情感预测"""
        emotion_controls = []
        
        for seg_idx, segment in enumerate(time_segments):
            if segment.type == 'word':
                # 提取词音频 - 增加上下文窗口
                context = 0.1  # 前后100ms上下文
                start = max(0, segment.start - context)
                end = segment.end + context
                start_sample = int(start * self.sr)
                end_sample = int(end * self.sr)
                seg_audio = audio[start_sample:end_sample]
                
                # 情感预测
                emotion_output = self.model.generate(seg_audio,
                                                    output_dir=None,
                                                    granularity="utterance",
                                                    extract_embedding=False,
                                                    )
                
                # 仅当置信度足够高时记录
                # if emotion_output['confidence'] > self.min_confidence:
                if True:
                    emotion, intensity, confidence = self._parse_emotion_output(emotion_output[0])
                    
                    # # 跳过中性情绪
                    # if emotion != 'neutral':
                    #     emotion_controls.append({
                    #         "type": self.type,
                    #         "value": emotion,
                    #         "intensity": intensity,
                    #         "pos": seg_idx,
                    #         "scope": "word",
                    #         "info": f"confidence={emotion_output['confidence']:.2f}"
                    #     })
                if emotion and intensity:
                    emotion_controls.append({
                        "type": self.type,
                        "value": f'{confidence:.2f}{emotion}',
                        "pos": seg_idx,
                        "info": f"[{segment.text}] emo={emotion}, intensity={intensity}, confidence={confidence:.2f}"
                    })
        
        return emotion_controls
    
    def _parse_emotion_output(self, output: Dict) -> str:
        """解析模型输出为情感标签和强度"""
        if self.multi_emotion:
            emotions = None
            for idx, score in enumerate(output['scores']):
                emotion = self.emotion_labels[idx]
                confidence = score
                if (score < self.min_confidence) or emotion == 'neutral':
                    continue
                if self.number_control:
                    if emotions is None:
                        emotions = f'{confidence:.2f}{emotion}'
                    else:
                        emotions += f',{confidence:.2f}{emotion}'
                else:
                    intensity = None
                    for level, threshold in self.intensity_levels.items():
                        if confidence < threshold:
                            intensity = level
                            break
                    if (intensity is None):
                        continue
                    if emotions is None:
                        emotions = f'{emotion}' if intensity == 'medium' else f'{intensity}_{emotion}'
                    else:
                        emotions += f',{emotion}' if intensity == 'medium' else f',{intensity}_{emotion}'
            if emotions is None:
                return None
            return emotions
        else:
            # 获取主导情感
            dominant_idx = np.argmax(output['scores'])
            emotion = self.emotion_labels[dominant_idx]
            confidence = output['scores'][dominant_idx]

            if (confidence < self.min_confidence) or emotion == 'neutral':
                return None

            if self.number_control:
                return f'{confidence:.2f}{emotion}'
            
            # 确定强度级别
            intensity = None
            for level, threshold in self.intensity_levels.items():
                if confidence < threshold:
                    intensity = level
                    break
            if intensity is None:
                return None
            return f'{emotion}' if intensity == 'medium' else f'{intensity}_{emotion}'
    
    def _load_audio(self, audio_path: str) -> np.ndarray:
        """加载音频并重采样至模型所需格式"""
        import librosa
        y, sr = librosa.load(audio_path, sr=self.sr, mono=True)
        return y

if __name__ == '__main__':
    from funasr import AutoModel
    import librosa

    # model="iic/emotion2vec_base"
    # model="iic/emotion2vec_base_finetuned"
    # model="iic/emotion2vec_plus_seed"
    # model="iic/emotion2vec_plus_base"
    model_id = "iic/emotion2vec_plus_large"

    model = AutoModel(
        model=model_id,
        hub="ms",  # "ms" or "modelscope" for China mainland users; "hf" or "huggingface" for other overseas users
        disable_update=True,
        device='cuda',
    )

    wav_file = f"{model.model_path}/example/test.wav"
    y, _ = librosa.load(wav_file, sr=None, mono=True)
    rec_result = model.generate(y, output_dir=None, granularity="utterance", extract_embedding=False)
    print(rec_result)