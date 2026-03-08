# -*- coding: utf-8 -*-
"""
style control extractor — reads style from config, outputs at first word
date: 20250308
author: liuyoude
"""
from typing import Dict, List

from core.feature_extractor.base_extractor import BaseExtractor, TimeSegment


class StyleExtractor(BaseExtractor):
    def __init__(self, config: Dict):
        super().__init__(config)
        self.type = "style"

    def load_model(self) -> None:
        pass

    def extract(
        self,
        audio_path: str,
        time_segments: List[TimeSegment],
        lang: str = None,
    ) -> List[Dict]:
        for idx, seg in enumerate(time_segments):
            if seg.type == "word":
                return [{
                    "type": self.type,
                    "value": self._style,
                    "pos": idx,
                    "info": f"style={self._style}",
                }]
        return []
