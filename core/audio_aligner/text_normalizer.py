# -*- coding: utf-8 -*-
"""
text normalizer for audio alignment
date: 20250430
author: liuyoude
"""
import re
import unicodedata

# sentence end char sets
SENTENCE_END_CHARS = {
    '。', '！', '？', '…', '，', '、',  # chinese punctuation
    '.', '!', '?', ':', ';', ',',  # english punctuation
}

class TextNormalizer:
    def __init__(self):
        # match english words, chinese words and other characters
        self.normalize_pattern = re.compile(r"([a-zA-Z0-9']+|[\u4e00-\u9fff\u3000-\u303f]|.)", re.UNICODE)
        # match chinese words with chinese punctuation
        self.chinese_punctuation_regex = re.compile(r'[\u4e00-\u9fff\u3000-\u303f\uff00-\uffef]+$')
        # match chinese words
        self.chinese_regex = re.compile(r'[\u4e00-\u9fff]')

    def normalize(self, text):
        tokens = [match.group() for match in self.normalize_pattern.finditer(text)]
        processed = []
        for token in tokens:
            if len(token) == 1 and not (self.chinese_regex.fullmatch(token) or token.isalnum()):
                if token.isspace():
                    continue
                elif processed:
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
            if self.chinese_punctuation_regex.fullmatch(token):
                if current_english:
                    eng_str = ' '.join(current_english)
                    # eng_str = re.sub(r'\s+([,.!?])', r'\1', eng_str)
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
            # eng_str = re.sub(r'\s+([,.!?])', r'\1', eng_str)
            result.append(eng_str)
        if current_chinese:
            result.append(''.join(current_chinese))
        
        return ''.join(result)
    
def is_punctuation(char: str) -> bool:
    """judge if a char is punctuation (support chinese and english punctuation)"""
    return len(char) == 1 and unicodedata.category(char).startswith('P')
    # if len(char) != 1:
    #     return False
    # # unicode juadge for punctuation  
    # category = unicodedata.category(char)
    # if category.startswith('P'):
    #     return True
    # judge for chinese punctuation
    # return char in {'，', '。', '？', '！', '；', '：', '“', '”', '（', '）', '《', '》', '【', '】', '、'}
    
if __name__ == "__main__":
    text_normalizer = TextNormalizer()
    text = "Hello, 你好！How are you？ 我很 好，谢 谢。"
    # text = "The invention of movable metal letters in the middle of the 15th century may justly be considered as the invention of the art of printing."
    # text = 'the 15th century'
    # text = "现在是测试VCIPER的音频。 它 能分 辨出 这个 分 段吗？"
    # text = "The examination and testimony of the experts enabled the Commission to conclude that five shots may have been fired."
    normalized_text = text_normalizer.normalize(text)
    print("Normalized Text:", normalized_text)
    print("Normalized Text Split:", normalized_text.split())
    denormalized_text = text_normalizer.denormalize(normalized_text)
    print("Denormalized Text:", denormalized_text)
    print("Is punctuation:", is_punctuation("、"))