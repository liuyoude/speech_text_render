# speech_text_render

语音控制符文本渲染工具 — 从音频中提取停顿、语速、音量、情感等特征，生成带控制标签的 TTS 文本。

## 功能

- **音频-文本对齐**: Qwen3-ASR（转录）+ Qwen3-ForcedAligner（强制对齐），支持中英文及多语言
- **音量控制符**: 检测耳语 / 低音量片段，生成 `[volume=whisper]` 标签
- **情感控制符**: 基于 emotion2vec 模型，生成 `[emotion=happy]` 等多情感标签
- **停顿控制符**: 检测句间 / 句内停顿
- **语速控制符**: 分析词级语速变化

## 快速开始

### 一键安装（推荐）

```bash
# CUDA 模式（需要 NVIDIA GPU）
bash scripts/setup.sh

# CPU 模式（无 GPU）
bash scripts/setup.sh --cpu
```

### 手动安装

```bash
# 1. 创建 conda 环境
conda create -n speech_text_render python=3.10 -y
conda activate speech_text_render

# 2. 安装 conda 包
conda install -c conda-forge ffmpeg -y

# 3. 安装 PyTorch
#    CUDA:
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
#    CPU:
# pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# 4. 安装项目依赖
pip install -e .
# 或:  pip install -r requirements.txt

# 5. 移除 conda 旧版 CUDA 库 (仅 WSL2 + CUDA 需要)
conda remove -n speech_text_render --force \
    libcublas libcufft libcufile libcurand libcusolver libcusparse \
    cuda-cudart cuda-cupti cuda-libraries cuda-nvrtc cuda-nvtx \
    cuda-opencl cuda-runtime cuda-version -y 2>/dev/null || true
```

## 使用方法

```bash
conda activate speech_text_render
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH

# 运行控制文本生成
python core/control_text_builder.py

# 运行音频对齐测试
python examples/test_audio_aligner.py
```

## 下载 LibriSpeech-100

在 WSL2 下，大体积语料建议优先放在 Linux 文件系统，例如 `~/datasets`，而不是 `/mnt/c` 或 `/mnt/d`。原因是模型推理和大量小文件遍历在 Linux 文件系统下通常更快，也能避免跨文件系统访问带来的额外开销；同时又不会把数 GB 的原始语料直接塞进 Git 仓库。

项目内已提供支持多线程分片、断点续传、MD5 校验与自动解压的下载脚本：

```bash
# 默认下载到 ~/datasets
bash scripts/download_librispeech_100.sh

# 指定目录与线程数
bash scripts/download_librispeech_100.sh --root ~/datasets --threads 4
```

默认目录结构如下：

```text
~/datasets/
├── _archives/librispeech/train-clean-100.tar.gz
└── LibriSpeech/train-clean-100/
```

注意：当前项目里的示例脚本主要按 `.wav + 同名 .txt` 配对来读取数据；而 LibriSpeech 原始格式是 `.flac + *.trans.txt`。因此，这个下载脚本解决的是“原始语料获取”，如果后续要直接接入当前 `examples/` 或 `dataset_config.yaml` 这套流程，通常还需要再做一次格式整理。

### Python API

```python
from core.control_text_builder import ControlBuilder
from core.feature_extractor import VolumeExtractor, EmotionExtractor

config = {
    "volume_extractor": {"number_control": False},
    "emotion_extractor": {
        "sentence_level": True,
        "number_control": False,
        "device": "cuda",
        "multi_emotion": True,
    },
}

builder = ControlBuilder(config)
builder.add_extractor([
    VolumeExtractor(config["volume_extractor"]),
    EmotionExtractor(config["emotion_extractor"]),
])

control_text = builder.build("path/to/audio.wav", "对应文本内容")
print(control_text)
# 输出示例: [emotion=happy]你 好 世 界
```

## 项目结构

```
speech_text_render/
├── core/
│   ├── audio_aligner/          # 音频-文本对齐模块
│   │   ├── audio_aligner.py    #   Qwen3 ASR / ForcedAligner 对齐器
│   │   └── text_normalizer.py  #   文本归一化
│   ├── feature_extractor/      # 特征提取模块
│   │   ├── pause_extractor.py  #   停顿提取
│   │   ├── speed_extractor.py  #   语速提取
│   │   ├── volume_extractor.py #   音量提取
│   │   └── emotion_extractor.py#   情感提取 (emotion2vec)
│   └── control_text_builder.py # 控制文本生成入口
├── examples/
│   ├── audios/                 # 测试音频 (中/英/混合)
│   └── test_audio_aligner.py   # 对齐测试脚本
├── scripts/
│   └── setup.sh                # 一键安装脚本
├── pyproject.toml              # 项目打包配置
├── requirements.txt            # pip 依赖列表
└── FIX_REPORT.md               # 环境搭建修复记录
```

## 环境要求

| 项目 | 要求 |
|------|------|
| Python | 3.10.x |
| PyTorch | >= 2.0 (推荐 2.10+ 以支持最新 GPU) |
| CUDA | 12.x (GPU 模式) |
| conda | Miniforge / Miniconda / Anaconda |
| 系统 | Linux / WSL2 |

## 已知限制

- WSL2 下需设置 `LD_LIBRARY_PATH` 以确保 `numba`/`llvmlite` 正确加载
- 中文字体缺失时绘图中文显示为方块，可通过 `sudo apt install fonts-wqy-zenhei` 解决
