#!/usr/bin/env bash
#
# speech_text_render 一键环境安装脚本
#
# 用法:
#   bash scripts/setup.sh          # CUDA 模式（默认）
#   bash scripts/setup.sh --cpu    # CPU 模式（无 GPU）
#
set -euo pipefail

# ---------------------------------------------------------------------------
# 配置
# ---------------------------------------------------------------------------
ENV_NAME="speech_text_render"
PYTHON_VERSION="3.10"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

USE_CPU=false
for arg in "$@"; do
    case "$arg" in
        --cpu) USE_CPU=true ;;
        --help|-h)
            echo "用法: bash scripts/setup.sh [--cpu]"
            echo "  --cpu   仅安装 CPU 版 PyTorch（无 GPU 环境使用）"
            exit 0
            ;;
    esac
done

# ---------------------------------------------------------------------------
# 辅助函数
# ---------------------------------------------------------------------------
info()  { echo -e "\033[1;32m[INFO]\033[0m  $*"; }
warn()  { echo -e "\033[1;33m[WARN]\033[0m  $*"; }
error() { echo -e "\033[1;31m[ERROR]\033[0m $*"; exit 1; }

check_command() {
    command -v "$1" &>/dev/null || error "未找到命令: $1。请先安装 $1。"
}

# ---------------------------------------------------------------------------
# 前置检查
# ---------------------------------------------------------------------------
check_command conda

CONDA_BASE="$(conda info --base)"
source "$CONDA_BASE/etc/profile.d/conda.sh"

# ---------------------------------------------------------------------------
# 步骤 1: 创建 conda 环境
# ---------------------------------------------------------------------------
if conda env list | grep -qw "$ENV_NAME"; then
    info "conda 环境 '$ENV_NAME' 已存在，跳过创建"
else
    info "创建 conda 环境: $ENV_NAME (Python $PYTHON_VERSION)"
    conda create -n "$ENV_NAME" python="$PYTHON_VERSION" -y
fi

conda activate "$ENV_NAME"

# ---------------------------------------------------------------------------
# 步骤 2: 安装 conda 包 (ffmpeg)
# ---------------------------------------------------------------------------
if conda list -n "$ENV_NAME" | grep -q "^ffmpeg "; then
    info "conda 包 (ffmpeg) 已安装，跳过"
else
    info "安装 conda 包: ffmpeg"
    conda install -n "$ENV_NAME" -c conda-forge ffmpeg -y
fi

# ---------------------------------------------------------------------------
# 步骤 3: 安装 PyTorch
# ---------------------------------------------------------------------------
if python -c "import torch; print(torch.__version__)" &>/dev/null; then
    TORCH_VER=$(python -c "import torch; print(torch.__version__)")
    info "PyTorch 已安装 ($TORCH_VER)，跳过"
else
    if [ "$USE_CPU" = true ]; then
        info "安装 PyTorch (CPU 模式)"
        pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
    else
        info "安装 PyTorch (CUDA 模式)"
        pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
    fi
fi

# ---------------------------------------------------------------------------
# 步骤 4: 安装 pip 依赖
# ---------------------------------------------------------------------------
info "安装 pip 依赖"
pip install -e "$PROJECT_DIR" 2>/dev/null || pip install -r "$PROJECT_DIR/requirements.txt"

# ---------------------------------------------------------------------------
# 步骤 5: 移除 conda 旧版 CUDA 库 (解决 WSL2 冲突)
# ---------------------------------------------------------------------------
if [ "$USE_CPU" = false ]; then
    CUDA_CONFLICT_PKGS=(
        libcublas libcufft libcufile libcurand libcusolver libcusparse
        cuda-cudart cuda-cupti cuda-libraries cuda-nvrtc cuda-nvtx
        cuda-opencl cuda-runtime cuda-version
    )
    FOUND_PKGS=()
    for pkg in "${CUDA_CONFLICT_PKGS[@]}"; do
        if conda list -n "$ENV_NAME" 2>/dev/null | grep -q "^${pkg} "; then
            FOUND_PKGS+=("$pkg")
        fi
    done

    if [ ${#FOUND_PKGS[@]} -gt 0 ]; then
        info "移除 conda 旧版 CUDA 库以避免与 PyTorch 捆绑库冲突..."
        conda remove -n "$ENV_NAME" --force "${FOUND_PKGS[@]}" -y
    else
        info "无 conda CUDA 库冲突，跳过"
    fi
fi

# ---------------------------------------------------------------------------
# 步骤 6: 验证安装
# ---------------------------------------------------------------------------
info "验证安装..."

export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"

python -c "
import torch
import librosa
import funasr
import qwen_asr
import matplotlib
print('所有核心包导入成功')
print(f'  PyTorch:  {torch.__version__}')
print(f'  CUDA:     {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'  GPU:      {torch.cuda.get_device_name(0)}')
print(f'  FunASR:   {funasr.__version__}')
print(f'  qwen-asr: {qwen_asr.__version__}')
"

info "============================================"
info "  环境安装完成！"
info "============================================"
info ""
info "使用方法:"
info "  conda activate $ENV_NAME"
info "  export LD_LIBRARY_PATH=\$CONDA_PREFIX/lib:\$LD_LIBRARY_PATH"
info "  python core/control_text_builder.py"
info ""
if [ "$USE_CPU" = true ]; then
    warn "当前为 CPU 模式，推理速度较慢"
fi
