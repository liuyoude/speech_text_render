# -*- coding: utf-8 -*-
"""
test audio aligner
date: 20250427
author: liuyoude
"""
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.audio_aligner.audio_aligner import WhisperAligner, MFAAligner, plot_alignment

def test_whisper_aligner():
    # model = whisper.load_model("turbo") # turbo tiny
    file_path_list = [
        "examples/audios/LJ037_0171_en_test.wav",
        "examples/audios/liuyoude_zh_test.wav",
    ]
    aligner = WhisperAligner(model_size="turbo", device="cpu")
    for file_path in file_path_list:
        print(f"file: {file_path}")
        segments = aligner.align(file_path)
        for seg in segments:
            print(f"[{seg.start:.2f}-{seg.end:.2f}] {seg.text} (type: {seg.type})") 
        plot_alignment(file_path, segments)

def test_mfa_aligner():
    file_path_list = [
        "examples/audios/LJ037_0171_en_test.wav",
        "examples/audios/liuyoude_zh_test.wav",
    ]
    aligner_zh = MFAAligner(lang="zh")
    aligner_en = MFAAligner(lang="en")
    for file_path in file_path_list:
        print(f"file: {file_path}")
        file_name = os.path.basename(file_path).split(".")[0]
        text_path = os.path.join(os.path.dirname(file_path), file_name + ".txt")
        with open(text_path, "r", encoding="utf-8") as f:
            text = f.read()
        if "zh" in file_name:
            segments = aligner_zh.align(file_path, text)
        elif "en" in file_name:
            segments = aligner_en.align(file_path, text)
        else:
            print(f"file: {file_path} not support")
            continue
        for seg in segments:
            print(f"[{seg.start:.2f}-{seg.end:.2f}] {seg.text} (type: {seg.type})")
        plot_alignment(file_path, segments)


if __name__ == "__main__":
    # test_whisper_aligner()
    test_mfa_aligner()