# -*- coding: utf-8 -*-
"""
base class for all feature extractors
date: 20250417
author: liuyoude
"""
from abc import ABC, abstractmethod
from typing import Dict, Optional

class BaseExtractor(ABC):
    def __init__(self, config: Dict):
        self.config = config
    
    @abstractmethod
    def load_model(self) -> None:
        """load pretrained model or method"""
        pass
    
    @abstractmethod
    def extract(self, audio_path: str, text: Optional[str] = None) -> Dict:
        """
        Extract features from audio and optional text.
        
        Args:
            audio_path: Path to the audio file
            text: Optional text corresponding to the audio
            
        Returns:
            Dictionary containing extracted features and controls
        """
        pass