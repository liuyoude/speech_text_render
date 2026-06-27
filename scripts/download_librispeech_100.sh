#!/usr/bin/env bash
#
# Download LibriSpeech train-clean-100 with multi-part resume support.
#
# Default layout:
#   $HOME/datasets/
#   ├── _archives/librispeech/train-clean-100.tar.gz
#   └── LibriSpeech/train-clean-100/
#
set -euo pipefail

DATA_ROOT="${LIBRISPEECH_ROOT:-$HOME/datasets}"
THREADS="${LIBRISPEECH_THREADS:-4}"
DOWNLOAD_ONLY=false
KEEP_PARTS=false
INTERRUPT_HANDLED=0

DATASET_SLUG="train-clean-100"
ARCHIVE_NAME="${DATASET_SLUG}.tar.gz"
URL="https://www.openslr.org/resources/12/${ARCHIVE_NAME}"
MD5_URL="https://www.openslr.org/resources/12/md5sum.txt"

usage() {
    cat <<'EOF'
用法:
  bash scripts/download_librispeech_100.sh [选项]

选项:
  --root PATH        数据根目录，默认: $HOME/datasets
  --threads N        下载线程数，默认: 4
  --download-only    只下载压缩包，不解压
  --keep-parts       下载成功后保留分片文件
  --help, -h         显示帮助

说明:
  1. 默认把压缩包放到 PATH/_archives/librispeech/
  2. 默认解压到 PATH/LibriSpeech/train-clean-100/
  3. 脚本支持断点续传；重新执行同一命令即可继续未完成的分片
EOF
}

info() {
    printf '\033[1;32m[INFO]\033[0m  %s\n' "$*"
}

warn() {
    printf '\033[1;33m[WARN]\033[0m  %s\n' "$*"
}

error() {
    printf '\033[1;31m[ERROR]\033[0m %s\n' "$*" >&2
    exit 1
}

require_cmd() {
    command -v "$1" >/dev/null 2>&1 || error "未找到命令: $1"
}

expand_path() {
    case "$1" in
        "~")
            printf '%s\n' "$HOME"
            ;;
        "~/"*)
            printf '%s\n' "$HOME/${1#~/}"
            ;;
        *)
            printf '%s\n' "$1"
            ;;
    esac
}

file_size() {
    wc -c < "$1" | tr -d ' '
}

get_remote_header() {
    local key="$1"
    curl -fsSLI "$URL" | awk -v header="$key" '
        tolower($1) == tolower(header) ":" {
            gsub("\r", "", $2)
            print $2
        }
    ' | tail -n 1
}

fetch_expected_md5() {
    curl -fsSL "$MD5_URL" | awk -v target="$ARCHIVE_NAME" '$2 == target { print $1 }'
}

download_segment() {
    trap - INT TERM

    local idx="$1"
    local start="$2"
    local end="$3"
    local part
    local expected_size
    local existing_size
    local resume_start
    local final_size

    part=$(printf '%s/part-%03d' "$PARTS_DIR" "$idx")
    expected_size=$((end - start + 1))
    existing_size=0

    if [ -f "$part" ]; then
        existing_size=$(file_size "$part")
    fi

    if [ "$existing_size" -gt "$expected_size" ]; then
        warn "分片 $idx 大于预期，重置该分片"
        rm -f "$part"
        existing_size=0
    fi

    if [ "$existing_size" -eq "$expected_size" ]; then
        info "分片 $idx 已完成，跳过"
        return 0
    fi

    resume_start=$((start + existing_size))
    info "分片 $idx 下载字节范围 ${resume_start}-${end}"

    curl \
        --fail \
        --location \
        --retry 8 \
        --retry-delay 2 \
        --connect-timeout 20 \
        --silent \
        --show-error \
        --range "${resume_start}-${end}" \
        --output - \
        "$URL" >> "$part"

    final_size=$(file_size "$part")
    if [ "$final_size" -ne "$expected_size" ]; then
        warn "分片 $idx 未下载完整，期望 ${expected_size} 字节，实际 ${final_size} 字节"
        return 1
    fi
}

download_archive() {
    local base_chunk
    local remainder
    local offset
    local start
    local extra
    local length
    local end
    local i
    local status=0
    local pids=()

    mkdir -p "$PARTS_DIR"

    if [ -f "$PARTS_LAYOUT" ]; then
        local current_layout
        current_layout=$(cat "$PARTS_LAYOUT")
        if [ "$current_layout" != "${REMOTE_SIZE}:${THREADS}" ]; then
            warn "检测到分片布局与本次线程数不同，清空旧分片后重新下载"
            rm -rf "$PARTS_DIR"
            mkdir -p "$PARTS_DIR"
        fi
    fi
    printf '%s:%s\n' "$REMOTE_SIZE" "$THREADS" > "$PARTS_LAYOUT"

    base_chunk=$((REMOTE_SIZE / THREADS))
    remainder=$((REMOTE_SIZE % THREADS))
    offset=0

    for ((i = 0; i < THREADS; i++)); do
        extra=0
        if [ "$i" -lt "$remainder" ]; then
            extra=1
        fi
        length=$((base_chunk + extra))
        start=$offset
        end=$((start + length - 1))
        offset=$((end + 1))

        download_segment "$i" "$start" "$end" &
        pids+=("$!")
    done

    for pid in "${pids[@]}"; do
        if ! wait "$pid"; then
            status=1
        fi
    done

    if [ "$status" -ne 0 ]; then
        error "至少有一个分片下载失败；重新运行脚本会继续续传"
    fi
}

merge_parts() {
    local merged_tmp="${ARCHIVE_PATH}.partial"
    local merged_size
    local part
    local i

    rm -f "$merged_tmp"
    : > "$merged_tmp"

    for ((i = 0; i < THREADS; i++)); do
        part=$(printf '%s/part-%03d' "$PARTS_DIR" "$i")
        [ -f "$part" ] || error "缺少分片文件: $part"
        cat "$part" >> "$merged_tmp"
    done

    merged_size=$(file_size "$merged_tmp")
    if [ "$merged_size" -ne "$REMOTE_SIZE" ]; then
        rm -f "$merged_tmp"
        error "合并后的文件大小不正确，期望 ${REMOTE_SIZE} 字节，实际 ${merged_size} 字节"
    fi

    mv "$merged_tmp" "$ARCHIVE_PATH"
}

verify_archive() {
    local actual_md5
    actual_md5=$(md5sum "$ARCHIVE_PATH" | awk '{print $1}')
    if [ "$actual_md5" != "$EXPECTED_MD5" ]; then
        rm -f "$ARCHIVE_PATH"
        rm -rf "$PARTS_DIR"
        error "MD5 校验失败，已删除损坏文件；重新运行会重新下载"
    fi
}

extract_archive() {
    if [ -f "$EXTRACT_MARKER" ] && [ -d "$EXTRACT_DIR" ]; then
        info "解压目录已存在，跳过解压: $EXTRACT_DIR"
        return 0
    fi

    mkdir -p "$DATA_ROOT"
    info "开始解压到: $DATA_ROOT"
    tar -xzf "$ARCHIVE_PATH" -C "$DATA_ROOT"
    mkdir -p "$EXTRACT_DIR"
    touch "$EXTRACT_MARKER"
}

cleanup_children() {
    jobs -pr | xargs -r kill 2>/dev/null || true
}

handle_interrupt() {
    if [ "$INTERRUPT_HANDLED" -eq 1 ]; then
        exit 130
    fi
    INTERRUPT_HANDLED=1
    warn "下载被中断，已保留当前分片；重新运行同一命令即可续传"
    cleanup_children
    exit 130
}

while [ $# -gt 0 ]; do
    case "$1" in
        --root)
            [ $# -ge 2 ] || error "--root 需要一个路径参数"
            DATA_ROOT="$2"
            shift 2
            ;;
        --threads)
            [ $# -ge 2 ] || error "--threads 需要一个整数参数"
            THREADS="$2"
            shift 2
            ;;
        --download-only)
            DOWNLOAD_ONLY=true
            shift
            ;;
        --keep-parts)
            KEEP_PARTS=true
            shift
            ;;
        --help|-h)
            usage
            exit 0
            ;;
        *)
            error "未知参数: $1"
            ;;
    esac
done

[[ "$THREADS" =~ ^[1-9][0-9]*$ ]] || error "--threads 必须是正整数"

DATA_ROOT="$(expand_path "$DATA_ROOT")"
ARCHIVE_DIR="$DATA_ROOT/_archives/librispeech"
ARCHIVE_PATH="$ARCHIVE_DIR/$ARCHIVE_NAME"
PARTS_DIR="$ARCHIVE_DIR/${ARCHIVE_NAME}.parts"
PARTS_LAYOUT="$PARTS_DIR/.layout"
EXTRACT_DIR="$DATA_ROOT/LibriSpeech/$DATASET_SLUG"
EXTRACT_MARKER="$EXTRACT_DIR/.extract_complete"

require_cmd curl
require_cmd tar
require_cmd md5sum

trap handle_interrupt INT TERM

mkdir -p "$ARCHIVE_DIR"

info "下载源: $URL"
info "数据根目录: $DATA_ROOT"
info "压缩包路径: $ARCHIVE_PATH"
info "解压目录: $EXTRACT_DIR"
info "线程数: $THREADS"

REMOTE_SIZE="$(get_remote_header "Content-Length")"
ACCEPT_RANGES="$(get_remote_header "Accept-Ranges")"
EXPECTED_MD5="$(fetch_expected_md5)"

[ -n "$REMOTE_SIZE" ] || error "无法获取远端文件大小"
[ -n "$EXPECTED_MD5" ] || error "无法获取官方 MD5"

if [ "$ACCEPT_RANGES" != "bytes" ] && [ "$THREADS" -gt 1 ]; then
    warn "远端未声明支持 Range，多线程已降级为单线程"
    THREADS=1
fi

if [ -f "$ARCHIVE_PATH" ]; then
    current_size=$(file_size "$ARCHIVE_PATH")
    if [ "$current_size" -eq "$REMOTE_SIZE" ]; then
        info "检测到已存在的压缩包，开始校验 MD5"
        verify_archive
    else
        warn "现有压缩包大小不匹配，删除后重新下载"
        rm -f "$ARCHIVE_PATH"
    fi
fi

if [ ! -f "$ARCHIVE_PATH" ]; then
    info "开始分片下载 ${ARCHIVE_NAME}（约 $((REMOTE_SIZE / 1024 / 1024)) MiB）"
    download_archive
    info "分片下载完成，开始合并"
    merge_parts
    info "开始校验 MD5"
    verify_archive
    info "压缩包校验通过"
    if [ "$KEEP_PARTS" = false ]; then
        rm -rf "$PARTS_DIR"
    fi
fi

if [ "$DOWNLOAD_ONLY" = true ]; then
    info "仅下载模式已完成"
else
    extract_archive
fi

cat <<EOF

完成路径:
  压缩包: $ARCHIVE_PATH
  原始语料: $EXTRACT_DIR

注意:
  当前项目现有脚本主要按 .wav + 同名 .txt 配对读取。
  LibriSpeech 原始格式是 .flac + *.trans.txt，下载后若要直接接入当前流程，通常还需要再做一次数据整理。
EOF
