# -*- coding: utf-8 -*-
"""
test audio aligner
date: 20250427
author: liuyoude
"""
import os
import sys
import matplotlib.pyplot as plt
import tqdm
import time

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core.audio_aligner import (
    WhisperAligner,
    # WhisperXAligner,
    MFAAligner,
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

def test_whisper_aligner(model_size="turbo", device="cpu", with_text=False):
    save_dir = f"examples/results/whisper_aligner/model_size={model_size}_device={device}"
    if with_text:
        save_dir = f"examples/results/whisper_aligner_with_text/model_size={model_size}_device={device}"
    os.makedirs(save_dir, exist_ok=True)
    file_path_list = get_file_path_list("examples/audios")
    aligner = WhisperAligner(model_size=model_size, device=device)
    txt_path = os.path.join(save_dir, "results.txt")
    if os.path.exists(txt_path):
        os.remove(txt_path)
    sum_time = 0
    for file_path in tqdm.tqdm(file_path_list, desc="whisper aligning", total=len(file_path_list)):
        file_name = os.path.basename(file_path).split(".")[0]
        text_path = os.path.join(os.path.dirname(file_path), file_name + ".txt")
        with open(text_path, "r", encoding="utf-8") as f:
            text = f.read()
        dir_name = os.path.basename(os.path.normpath(os.path.dirname(file_path)))
        file_name = f"{dir_name}_{file_name}"
        start_time = time.time()
        segments = aligner.align(file_path, text=text if with_text else None)
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

def test_mfa_aligner():
    # save_dir = f"examples/results/mfa_aligner/aishell_model"
    save_dir = f"examples/results/mfa_aligner/office_model"
    os.makedirs(save_dir, exist_ok=True)
    file_path_list = get_file_path_list("examples/audios")
    aligner_zh = MFAAligner(lang="zh")
    aligner_en = MFAAligner(lang="en")
    txt_path = os.path.join(save_dir, "results.txt")
    if os.path.exists(txt_path):
        os.remove(txt_path)
    sum_time = 0
    sum_file_num = len(file_path_list)
    for file_path in tqdm.tqdm(file_path_list, desc="mfa aligning", total=len(file_path_list)):
        file_name = os.path.basename(file_path).split(".")[0]
        text_path = os.path.join(os.path.dirname(file_path), file_name + ".txt")
        with open(text_path, "r", encoding="utf-8") as f:
            text = f.read()
        dir_name = os.path.basename(os.path.normpath(os.path.dirname(file_path)))
        file_name = f"{dir_name}_{file_name}"
        start_time = time.time()
        if "zh" in dir_name:
            segments = aligner_zh.align(file_path, text)
        elif "en" in dir_name:
            segments = aligner_en.align(file_path, text)
        else:
            sum_file_num -= 1
            print(f"file: {file_path} not support")
            continue
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
        if sum_file_num == 0:
            f.write(f"average processing time: nan s\n")
        else:
            f.write(f"average processing time: {sum_time / sum_file_num:.2f}s\n")

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
    # test_whisper_aligner(model_size="tiny", device="cpu")
    # test_whisper_aligner(model_size="base", device="cpu")
    # test_whisper_aligner(model_size="small", device="cpu")
    # test_whisper_aligner(model_size="medium", device="cpu")
    # test_whisper_aligner(model_size="turbo", device="cpu")
    # test_whisper_aligner(model_size="large", device="cpu")

    # test_whisper_aligner(model_size="small", device="cpu", with_text=True)
    # test_whisper_aligner(model_size="medium", device="cpu", with_text=True)

    # test_whisper_aligner(model_size="tiny", device="cuda")
    # test_whisper_aligner(model_size="base", device="cuda")
    # test_whisper_aligner(model_size="small", device="cuda")   
  
    # test_mfa_aligner()

    test_speech_text_aligner()