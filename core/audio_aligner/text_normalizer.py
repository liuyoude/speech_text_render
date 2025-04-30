# -*- coding: utf-8 -*-
"""
text normalizer for audio alignment
date: 20250430
author: liuyoude
"""
import re

# sentence end char sets
SENTENCE_END_CHARS = {
    '。', '！', '？', '…',  # chinese punctuation
    '.', '!', '?', ':', ';'  # english punctuation
}

class TextNormalizer:
    def __init__(self):
        # 匹配英文单词、汉字（含中文标点）、其他字符
        self.normalize_pattern = re.compile(r"([a-zA-Z']+|[\u4e00-\u9fff\u3000-\u303f]|.)", re.UNICODE)
        # 中文判断包含汉字和中文标点
        # self.chinese_regex = re.compile(r'[\u4e00-\u9fff\u3000-\u303f]')
        self.chinese_regex = re.compile(r'[\u4e00-\u9fff]')

    def normalize(self, text):
        tokens = [match.group() for match in self.normalize_pattern.finditer(text)]
        processed = []
        for token in tokens:
            if len(token) == 1 and not (self.chinese_regex.fullmatch(token) or token.isalnum()):
                if processed:
                    processed[-1] += token
                else:
                    processed.append(token)
            else:
                processed.append(token)
        return ' '.join(processed)

    def denormalize(self, text):
        tokens = text.split()
        result = []
        current_chinese = []
        current_english = []
        
        for token in tokens:
            if self.chinese_regex.fullmatch(token):
                if current_english:
                    eng_str = ' '.join(current_english)
                    eng_str = re.sub(r'\s+([,.!?])', r'\1', eng_str)
                    result.append(eng_str)
                    current_english = []
                current_chinese.append(token)
            else:
                if current_chinese:
                    result.append(''.join(current_chinese))
                    current_chinese = []
                current_english.append(token)
        
        # 处理剩余部分
        if current_english:
            eng_str = ' '.join(current_english)
            eng_str = re.sub(r'\s+([,.!?])', r'\1', eng_str)
            result.append(eng_str)
        if current_chinese:
            result.append(''.join(current_chinese))
        
        return ''.join(result)
    
if __name__ == "__main__":
    text_normalizer = TextNormalizer()
    text = "Hello, 你好！How are you？ 我很 好，谢 谢。"
    # text = "现在是测试VCIPER的音频。 它 能分 辨出 这个 分 段吗？"
    # text = "The examination and testimony of the experts enabled the Commission to conclude that five shots may have been fired."
    normalized_text = text_normalizer.normalize(text)
    print("Normalized Text:", normalized_text.split())
    denormalized_text = text_normalizer.denormalize(normalized_text)
    print("Denormalized Text:", denormalized_text)