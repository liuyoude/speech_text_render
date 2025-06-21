# -*- coding: utf-8 -*-
"""
base class for all feature extractors
date: 20250417
author: liuyoude
"""
from abc import ABC, abstractmethod
from typing import Dict, Optional, List
from core.audio_aligner.audio_aligner import TimeSegment

class BaseExtractor(ABC):
    def __init__(self, config: Dict):
        self.config = config
    
    @abstractmethod
    def load_model(self) -> None:
        """load pretrained model or method"""
        pass
    
    @abstractmethod
    def extract(self, audio_path: str, time_segments: List[TimeSegment], lang: str=None) -> Dict:
        """
        Extract features from audio and optional text.
        
        Args:
            time_segments (List[TimeSegment]): List of time segments for alignment.
            
            
        Returns:
            Dictionary containing extracted features and controls
        """
        pass