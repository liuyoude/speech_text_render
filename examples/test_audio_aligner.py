# -*- coding: utf-8 -*-
"""
test audio aligner
date: 20250427
author: liuyoude
"""
import os
import tqdm
import time
from core.audio_aligner import (
    SpeechTextAligner, 
    plot_alignment
)

def get_file_path_list(root_dir, ext=".wav"):
    file_path_list = []
    for root, dirs, files in os.walk(root_dir):
        for file in files:
            if file.endswith(ext):
                file_path_list.append(os.path.join(root, file))
    return file_path_list

def test_speech_text_aligner():
    save_dir = f"examples/results/speech_text_aligner"
    os.makedirs(save_dir, exist_ok=True)
    file_path_list = get_file_path_list("examples/audios")
    aligner = SpeechTextAligner()
    txt_path = os.path.join(save_dir, "results.txt")
    if os.path.exists(txt_path):
        os.remove(txt_path)
    sum_time = 0
    for file_path in tqdm.tqdm(file_path_list, desc="speech text aligning", total=len(file_path_list)):
        file_name = os.path.basename(file_path).split(".")[0]
        text_path = os.path.join(os.path.dirname(file_path), file_name + ".txt")
        with open(text_path, "r", encoding="utf-8") as f:
            text = f.read()
        dir_name = os.path.basename(os.path.normpath(os.path.dirname(file_path)))
        file_name = f"{dir_name}_{file_name}"
        start_time = time.time()
        segments = aligner.align(file_path, text)
        end_time = time.time()
        sum_time += end_time - start_time
        img_path = os.path.join(save_dir, file_name + ".png")
        with open(txt_path, "a", encoding="utf-8") as f:
            f.write(f"file: {file_path}\n")
            f.write(f"processing time: {end_time - start_time:.2f}s\n")
            f.write(f"real text: {text}\n")
            for seg in segments:
                f.write(f"[{seg.start:.2f}-{seg.end:.2f}] {seg.text} (type: {seg.type})\n"
            )
            f.write("\n")
        plot_alignment(file_path, segments, img_path)
    with open(txt_path, "a", encoding="utf-8") as f:
        f.write(f"average processing time: {sum_time / len(file_path_list):.2f}s\n")



if __name__ == "__main__":
    test_speech_text_aligner()
