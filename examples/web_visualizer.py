# -*- coding: utf-8 -*-
"""
Gradio-based audio alignment & control-tag visualizer.

Usage:
    conda activate speech_text_render
    python examples/web_visualizer.py [--device cuda] [--port 8080]
"""
import json
import os
import sys
import copy
import shutil
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple

_conda_bin = os.path.join(sys.prefix, "bin")
if _conda_bin not in os.environ.get("PATH", ""):
    os.environ["PATH"] = _conda_bin + os.pathsep + os.environ.get("PATH", "")

import numpy as np
import librosa
import librosa.display
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import matplotlib.patches as mpatches
import gradio as gr

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT_DIR)

import inspect as _inspect
from core.audio_aligner.audio_aligner import SpeechTextAligner, TimeSegment
from core.control_text_builder import ControlGenerator
import core.feature_extractor as _fe_module
from core.feature_extractor.base_extractor import BaseExtractor

AUDIOS_DIR = os.path.join(ROOT_DIR, "examples", "audios")
WEBSRCS_DIR = os.path.join(ROOT_DIR, "examples", "websrcs")

# ---------------------------------------------------------------------------
# matplotlib CJK font setup
# ---------------------------------------------------------------------------
_FONT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fonts")
_bundled_font = os.path.join(_FONT_DIR, "NotoSansCJKsc-Regular.otf")
if os.path.isfile(_bundled_font):
    fm.fontManager.addfont(_bundled_font)

_zh_fonts = ["Noto Sans CJK SC", "SimHei", "WenQuanYi Zen Hei",
             "AR PL UMing CN", "DejaVu Sans"]
_available = {f.name for f in fm.fontManager.ttflist}
plt.rcParams["font.sans-serif"] = [f for f in _zh_fonts if f in _available] or ["DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

# ---------------------------------------------------------------------------
# Global state
# ---------------------------------------------------------------------------
aligner: Optional[SpeechTextAligner] = None
result_cache: Dict[str, dict] = {}
DEVICE = "cpu"

_COLOR_PALETTE = ["#ff9800", "#4caf50", "#ab47bc", "#ec407a",
                  "#29b6f6", "#ef5350", "#66bb6a", "#ffa726"]

_KNOWN_DEFAULTS: Dict[str, dict] = {
    "pause": {"number_control": False},
    "speed": {"number_control": True},
    "volume": {"number_control": False},
    "emotion": {
        "sentence_level": True, "number_control": False,
        "device": "cuda", "multi_emotion": True,
    },
}


def _discover_extractors():
    """Auto-discover all BaseExtractor subclasses from core.feature_extractor."""
    found = {}
    for name, cls in _inspect.getmembers(_fe_module, _inspect.isclass):
        if not issubclass(cls, BaseExtractor) or cls is BaseExtractor:
            continue
        ext_name = name.replace("Extractor", "").lower()
        try:
            tmp = cls({})
            ctrl_type = tmp.type
        except Exception:
            ctrl_type = ext_name
        found[ext_name] = {"class": cls, "ctrl_type": ctrl_type}
    return found


_DISCOVERED = _discover_extractors()

EXTRACTOR_CLASSES: Dict[str, type] = {k: v["class"] for k, v in _DISCOVERED.items()}
EXTRACTOR_CTRL_TYPES: Dict[str, str] = {k: v["ctrl_type"] for k, v in _DISCOVERED.items()}
EXTRACTOR_DEFAULTS: Dict[str, dict] = {
    k: dict(_KNOWN_DEFAULTS.get(k, {})) for k in EXTRACTOR_CLASSES
}
CTRL_COLORS: Dict[str, str] = {
    v["ctrl_type"]: _COLOR_PALETTE[i % len(_COLOR_PALETTE)]
    for i, v in enumerate(_DISCOVERED.values())
}


def _get_aligner() -> SpeechTextAligner:
    global aligner
    if aligner is None:
        print(f"[visualizer] loading SpeechTextAligner (device={DEVICE}) …")
        aligner = SpeechTextAligner(device=DEVICE)
    return aligner


# ---------------------------------------------------------------------------
# Scan audio files
# ---------------------------------------------------------------------------

def _get_cache_paths(audio_path: str):
    """Map an audio file path to its analysis cache paths under websrcs/."""
    rel = os.path.relpath(audio_path, AUDIOS_DIR)
    base = os.path.splitext(rel)[0]
    png_path = os.path.join(WEBSRCS_DIR, base + "_analysis.png")
    json_path = os.path.join(WEBSRCS_DIR, base + "_analysis.json")
    os.makedirs(os.path.dirname(png_path), exist_ok=True)
    return png_path, json_path


def _scan_audio_files() -> Dict[str, dict]:
    files = {}
    for lang_dir in sorted(Path(AUDIOS_DIR).iterdir()):
        if not lang_dir.is_dir():
            continue
        lang = lang_dir.name
        for wav in sorted(lang_dir.glob("*.wav")):
            txt_path = wav.with_suffix(".txt")
            text = txt_path.read_text(encoding="utf-8").strip() if txt_path.exists() else ""
            label = f"[{lang}] {wav.stem}"
            files[label] = {"path": str(wav), "text": text, "lang": lang}
    return files


FILE_MAP = _scan_audio_files()


# ---------------------------------------------------------------------------
# Analysis pipeline
# ---------------------------------------------------------------------------

def _is_clause_start(pos, segments):
    """Check if pos is the first word of a clause."""
    if pos == 0:
        return True
    for i in range(pos - 1, -1, -1):
        if segments[i].type in ("clause", "sentence"):
            return True
        if segments[i].type == "word":
            return False
    return True


def _analyze(audio_path: str, text: str, extractor_names: List[str]) -> dict:
    if not text or not text.strip():
        text = None
    cache_key = f"{audio_path}||{text}||{'|'.join(sorted(extractor_names))}"
    if cache_key in result_cache:
        return result_cache[cache_key]

    a = _get_aligner()
    segments = a.align(audio_path, text)
    segments_snapshot = copy.deepcopy(segments)

    cfg: dict = {}
    for name in extractor_names:
        cfg[f"{name}_extractor"] = dict(EXTRACTOR_DEFAULTS.get(name, {}))
        if name == "emotion":
            cfg["emotion_extractor"]["device"] = DEVICE

    gen = ControlGenerator(cfg)
    exts = []
    for name in extractor_names:
        cls = EXTRACTOR_CLASSES.get(name)
        if cls:
            exts.append(cls(cfg[f"{name}_extractor"]))
    gen.add_extractor(exts)

    controls_raw: List[dict] = []
    for ext in gen.extractors.values():
        controls_raw.extend(ext.extract(audio_path, segments_snapshot, lang=a.lang))

    controls_merged: Dict[int, str] = {}
    for c in controls_raw:
        tag = f'[{c["type"]}={c["value"]}]'
        pos = c["pos"]
        controls_merged[pos] = controls_merged.get(pos, "") + tag
    for pos in controls_merged:
        parts = controls_merged[pos][1:-1].replace("][", ",")
        controls_merged[pos] = f"[{parts}]"

    segs_list = [
        {"start": float(s.start), "end": float(s.end), "text": s.text, "type": s.type}
        for s in segments_snapshot
    ]

    details = []
    for c in controls_raw:
        pos = c["pos"]
        ctype = c["type"]
        if pos >= len(segments_snapshot):
            t_start = t_end = 0.0
        elif ctype == "break":
            prev_end = 0.0
            for i in range(pos - 1, -1, -1):
                if segments_snapshot[i].type == "word":
                    prev_end = float(segments_snapshot[i].end)
                    break
            t_start = prev_end
            t_end = float(segments_snapshot[pos].start)
        elif ctype == "emotion":
            t_start = float(segments_snapshot[pos].start)
            t_end = t_start
            for i in range(pos, len(segments_snapshot)):
                if segments_snapshot[i].type == "sentence":
                    t_start = float(segments_snapshot[i].start)
                    t_end = float(segments_snapshot[i].end)
                    break
        elif ctype == "speed" or (
            ctype == "volume" and _is_clause_start(pos, segments_snapshot)
        ):
            t_start = float(segments_snapshot[pos].start)
            t_end = t_start
            for i in range(pos, len(segments_snapshot)):
                if segments_snapshot[i].type == "clause":
                    t_start = float(segments_snapshot[i].start)
                    t_end = float(segments_snapshot[i].end)
                    break
        else:
            t_start = float(segments_snapshot[pos].start)
            t_end = float(segments_snapshot[pos].end)
        details.append({
            "type": ctype,
            "value": str(c["value"]),
            "pos": pos,
            "info": c.get("info", ""),
            "time": round(t_start, 4),
            "time_end": round(t_end, 4),
        })

    text_parts = []
    for idx, seg in enumerate(segments_snapshot):
        if seg.type == "word":
            prefix = controls_merged.get(idx, "")
            text_parts.append(prefix + seg.text)
    control_text = " ".join(text_parts)

    result = {
        "segments": segs_list,
        "controls": {str(k): v for k, v in controls_merged.items()},
        "control_text": control_text,
        "control_details": details,
        "audio_path": audio_path,
    }
    result_cache[cache_key] = result
    return result


# ---------------------------------------------------------------------------
# Combined matplotlib plot (unified time-axis)
# ---------------------------------------------------------------------------

def _plot_combined(audio_path, segments, control_details, controls,
                   show_word, show_clause, show_sentence,
                   active_ctrl_types, save_path=None):
    audio, sr = librosa.load(audio_path, sr=None, mono=True)
    duration = len(audio) / sr
    time_axis = np.linspace(0, duration, len(audio))

    n_ctrl_rows = len(active_ctrl_types)
    has_ctrl = n_ctrl_rows > 0

    h_text, h_wave, h_spec = 1.2, 3.0, 3.0
    h_ctrl = max(0.9 * n_ctrl_rows, 0.9) if has_ctrl else 0

    height_ratios = [h_text, h_wave, h_spec]
    if has_ctrl:
        height_ratios.append(h_ctrl)
    n_axes = len(height_ratios)

    fig, axes = plt.subplots(
        n_axes, 1, figsize=(16, sum(height_ratios)), dpi=100,
        sharex=True, gridspec_kw={"height_ratios": height_ratios},
    )
    if n_axes == 1:
        axes = [axes]

    ax_text = axes[0]
    ax_wave = axes[1]
    ax_spec = axes[2]
    ax_ctrl = axes[3] if has_ctrl else None

    # ---- ax_text: Word Timeline Bar ----
    word_segs = [(i, seg) for i, seg in enumerate(segments) if seg["type"] == "word"]

    for wi, (idx, seg) in enumerate(word_segs):
        w = seg["end"] - seg["start"]
        bg = "#f0f0f0" if wi % 2 == 0 else "#dcdcdc"

        ctrl_str = controls.get(str(idx))
        if ctrl_str:
            first_type = ctrl_str.strip("[]").split(",")[0].split("=")[0]
            ec = CTRL_COLORS.get(first_type, "#888888")
            lw = 2.0
        else:
            ec = "#bbbbbb"
            lw = 0.5

        rect = mpatches.FancyBboxPatch(
            (seg["start"], 0.15), w, 0.5,
            boxstyle="round,pad=0.008", facecolor=bg, edgecolor=ec, linewidth=lw,
        )
        ax_text.add_patch(rect)

        mid_x = seg["start"] + w / 2
        ax_text.text(mid_x, 0.4, seg["text"], ha="center", va="center",
                     fontsize=6.5, clip_on=True)

        if ctrl_str:
            inner = ctrl_str.strip("[]")
            parts = inner.split(",")
            n_p = len(parts)
            for pi, part in enumerate(parts):
                ctype = part.split("=")[0]
                cval = part.split("=")[1] if "=" in part else ""
                c_color = CTRL_COLORS.get(ctype, "#888888")
                x_pos = seg["start"] + (pi + 0.5) * w / n_p
                ax_text.plot(x_pos, 0.78, marker="v", markersize=5,
                             color=c_color, clip_on=False, zorder=5)
                ax_text.text(x_pos, 0.88, cval, ha="center", va="bottom",
                             fontsize=5, color=c_color, clip_on=False,
                             fontweight="bold")

    ax_text.set_ylim(0, 1.15)
    ax_text.set_xlim(0, duration)
    ax_text.set_yticks([])
    ax_text.set_title("Word Timeline", fontsize=10, loc="left")

    # ---- ax_wave: Waveform ----
    ax_wave.plot(time_axis, audio, linewidth=0.4)
    ax_wave.set_ylabel("Amplitude")

    for seg in segments:
        if seg["type"] == "word" and show_word:
            ax_wave.axvspan(seg["start"], seg["end"], alpha=0.15, color="red")
        elif seg["type"] == "clause" and show_clause:
            ax_wave.axvline(seg["start"], color="green", linestyle=":", linewidth=0.8)
            ax_wave.axvline(seg["end"], color="green", linestyle=":", linewidth=0.8)
        elif seg["type"] == "sentence" and show_sentence:
            ax_wave.axvline(seg["start"], color="blue", linestyle="--", linewidth=0.8)
            ax_wave.axvline(seg["end"], color="blue", linestyle="--", linewidth=0.8)

    ax_wave.set_title("Waveform", fontsize=10, loc="left")

    # ---- ax_spec: Spectrogram ----
    n_fft, hop_length, n_mels = 1024, 512, 128
    S = librosa.feature.melspectrogram(
        y=audio, sr=sr, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels,
    )
    S_db = librosa.amplitude_to_db(S, ref=np.max)
    librosa.display.specshow(
        S_db, sr=sr, x_axis="time", y_axis="mel",
        ax=ax_spec, hop_length=hop_length, cmap="viridis",
    )
    ax_spec.set_ylabel("Hz")

    for seg in segments:
        if seg["type"] == "word" and show_word:
            ax_spec.axvspan(seg["start"], seg["end"], alpha=0.1, color="white")
        elif seg["type"] == "clause" and show_clause:
            ax_spec.axvline(seg["start"], color="green", linestyle=":", linewidth=0.8)
            ax_spec.axvline(seg["end"], color="green", linestyle=":", linewidth=0.8)
        elif seg["type"] == "sentence" and show_sentence:
            ax_spec.axvline(seg["start"], color="blue", linestyle="--", linewidth=0.8)
            ax_spec.axvline(seg["end"], color="blue", linestyle="--", linewidth=0.8)

    ax_spec.set_title("Mel Spectrogram", fontsize=10, loc="left")
    if not has_ctrl:
        ax_spec.set_xlabel("Time (s)")

    # ---- ax_ctrl: Control Tags Timeline ----
    if ax_ctrl is not None:
        active_reversed = list(reversed(active_ctrl_types))
        type_to_row = {ctype: i for i, ctype in enumerate(active_reversed)}

        ax_ctrl.set_ylim(-0.5, n_ctrl_rows - 0.5)
        ax_ctrl.set_yticks(range(n_ctrl_rows))
        ax_ctrl.set_yticklabels(active_reversed, fontsize=9)
        ax_ctrl.tick_params(axis="y", length=0)
        for y_line in range(n_ctrl_rows + 1):
            ax_ctrl.axhline(y_line - 0.5, color="#dddddd", linewidth=0.5)

        bar_h = 0.7
        for c in control_details:
            ctype = c["type"]
            if ctype not in type_to_row:
                continue
            row = type_to_row[ctype]
            color = CTRL_COLORS.get(ctype, "#888888")
            w = max(c["time_end"] - c["time"], 0.02)

            rect = mpatches.FancyBboxPatch(
                (c["time"], row - bar_h / 2), w, bar_h,
                boxstyle="round,pad=0.02", facecolor=color,
                edgecolor="white", linewidth=0.5, alpha=0.85,
            )
            ax_ctrl.add_patch(rect)
            mid = c["time"] + w / 2
            ax_ctrl.text(mid, row, c["value"], ha="center", va="center",
                         fontsize=7, color="white", fontweight="bold",
                         clip_on=True)

        ax_ctrl.set_title("Control Tags", fontsize=10, loc="left")
        ax_ctrl.set_xlabel("Time (s)")

    fig.tight_layout(pad=0.6)
    fig.subplots_adjust(hspace=0.15)

    if save_path:
        fig.savefig(save_path, format="png", dpi=100, bbox_inches="tight")
        plt.close(fig)
        return save_path

    plt.close(fig)
    return None


# ---------------------------------------------------------------------------
# Build Gradio outputs
# ---------------------------------------------------------------------------

def _build_highlighted_text(segments, controls):
    result = []
    color_map = {}
    for idx, seg in enumerate(segments):
        if seg["type"] != "word":
            continue
        ctrl_str = controls.get(str(idx))
        if ctrl_str:
            inner = ctrl_str.strip("[]")
            parts = inner.split(",")
            for part in parts:
                ctype = part.split("=")[0]
                result.append((f"[{part}]", ctype))
                if ctype not in color_map:
                    color_map[ctype] = CTRL_COLORS.get(ctype, "#888888")
        result.append((seg["text"] + " ", None))
    return result, color_map


def _build_segments_df(segments, controls):
    rows = []
    for idx, seg in enumerate(segments):
        ctrl = controls.get(str(idx), "")
        rows.append([
            f'{seg["start"]:.3f}',
            f'{seg["end"]:.3f}',
            f'{seg["end"] - seg["start"]:.3f}',
            seg["type"],
            seg["text"],
            ctrl,
        ])
    return rows


# ---------------------------------------------------------------------------
# Gradio callbacks
# ---------------------------------------------------------------------------

def on_file_select(file_choice):
    info = FILE_MAP.get(file_choice)
    if info is None:
        return "", None, None, None, gr.HighlightedText(value=None), []

    audio_path = info["path"]
    png_path, json_path = _get_cache_paths(audio_path)

    if os.path.isfile(png_path) and os.path.isfile(json_path):
        with open(json_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
        return (
            info["text"], audio_path, audio_path,
            png_path,
            gr.HighlightedText(
                value=meta["highlighted_text"],
                color_map=meta["color_map"],
                label="控制文本 / Control Text",
            ),
            meta["df_rows"],
        )

    return (info["text"], audio_path, audio_path,
            None, gr.HighlightedText(value=None), [])


def on_recording_done(audio_path):
    import time as _time
    if audio_path is None:
        return "", "en", None, None, f"rec_{_time.strftime('%Y%m%d_%H%M%S')}"
    try:
        a = _get_aligner()
        text, lang = a.qwen3_asr.transcribe(audio_path)
        text = text.strip()
        lang_short = lang if lang else "en"
        default_name = f"rec_{_time.strftime('%Y%m%d_%H%M%S')}"
        return text, lang_short, audio_path, audio_path, default_name
    except Exception as e:
        raise gr.Error(f"语音识别失败: {e}")


def on_save_recording(rec_audio, text, lang, filename):
    if not rec_audio:
        raise gr.Error("没有录音文件")
    if not filename or not filename.strip():
        raise gr.Error("请填写文件名")
    filename = filename.strip()
    lang_dir = os.path.join(AUDIOS_DIR, lang)
    os.makedirs(lang_dir, exist_ok=True)
    dst_wav = os.path.join(lang_dir, f"{filename}.wav")
    dst_txt = os.path.join(lang_dir, f"{filename}.txt")
    if os.path.exists(dst_wav):
        raise gr.Error(f"文件已存在: {dst_wav}")
    shutil.copy2(rec_audio, dst_wav)
    with open(dst_txt, "w", encoding="utf-8") as f:
        f.write(text.strip())
    label = f"[{lang}] {filename}"
    FILE_MAP[label] = {"path": dst_wav, "text": text.strip(), "lang": lang}
    gr.Info(f"已保存: {dst_wav}")
    return gr.Dropdown(choices=list(FILE_MAP.keys()), value=label)


def analyze_and_plot(audio_path, text, *args):
    if not audio_path:
        raise gr.Error("请先选择或录制音频文件")

    ext_names = list(EXTRACTOR_CLASSES.keys())
    ext_flags = args[:len(ext_names)]
    show_word, show_clause, show_sentence = args[len(ext_names):]

    extractors = [n for n, flag in zip(ext_names, ext_flags) if flag]

    keys_to_remove = [k for k in result_cache if k.startswith(f"{audio_path}||")]
    for k in keys_to_remove:
        del result_cache[k]

    try:
        result = _analyze(audio_path, text, extractors)
    except Exception as e:
        raise gr.Error(f"分析失败: {e}")

    segments = result["segments"]
    controls = result["controls"]
    details = result["control_details"]

    active_ctrl_types = []
    for n, flag in zip(ext_names, ext_flags):
        if flag:
            active_ctrl_types.append(EXTRACTOR_CTRL_TYPES[n])

    png_path, json_path = _get_cache_paths(audio_path)

    _plot_combined(
        audio_path, segments, details, controls,
        show_word, show_clause, show_sentence, active_ctrl_types,
        save_path=png_path,
    )

    hl_data, color_map = _build_highlighted_text(segments, controls)
    df_rows = _build_segments_df(segments, controls)

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump({"highlighted_text": hl_data, "color_map": color_map, "df_rows": df_rows}, f, ensure_ascii=False)

    return (
        audio_path,
        png_path,
        gr.HighlightedText(
            value=hl_data, color_map=color_map,
            label="控制文本 / Control Text",
        ),
        df_rows,
    )


# ---------------------------------------------------------------------------
# Gradio UI
# ---------------------------------------------------------------------------

def build_app():
    file_choices = list(FILE_MAP.keys())

    with gr.Blocks(title="Speech-Text-Render Visualizer") as app:
        gr.Markdown("## Speech-Text-Render Visualizer")

        audio_state = gr.State(value=None)

        default_file = file_choices[0] if file_choices else None

        with gr.Tabs():
            with gr.Tab("选择已有文件"):
                file_dd = gr.Dropdown(
                    choices=file_choices, value=default_file,
                    label="选择音频文件",
                )

            with gr.Tab("上传 / 录音"):
                with gr.Tabs():
                    with gr.Tab("麦克风录音"):
                        mic_recorder = gr.Microphone(
                            type="filepath", label="点击录音按钮开始，再次点击停止",
                            format="wav",
                        )
                    with gr.Tab("上传文件"):
                        mic_upload = gr.Audio(
                            sources=["upload"], type="filepath",
                            label="上传音频文件",
                        )
                mic_audio = gr.State(value=None)
                with gr.Row():
                    save_lang = gr.Dropdown(
                        choices=["en", "zh", "zh_en"], value="zh",
                        label="语种", scale=1,
                    )
                    save_name = gr.Textbox(
                        label="文件名 (不含扩展名)",
                        value=f"rec_{__import__('time').strftime('%Y%m%d_%H%M%S')}",
                        scale=2,
                    )
                    save_btn = gr.Button(
                        "保存到 audios", variant="secondary", scale=1,
                    )

        text_box = gr.Textbox(
            label="文本 (自动识别/填入, 可手动编辑)", lines=2,
        )

        with gr.Row():
            with gr.Column(scale=1, min_width=200):
                gr.Markdown("**Extractor 配置**")
                _ext_default_off = {"emotion"}
                ext_checks = {}
                for ext_name, ext_info in _DISCOVERED.items():
                    ext_checks[ext_name] = gr.Checkbox(
                        label=f"{ext_info['ctrl_type'].capitalize()} ({ext_name})",
                        value=(ext_name not in _ext_default_off),
                    )

                gr.Markdown("**显示选项**")
                chk_word = gr.Checkbox(label="Word 段", value=True)
                chk_sentence = gr.Checkbox(label="Sentence 段", value=True)
                chk_clause = gr.Checkbox(label="Clause 分句", value=True)

                btn = gr.Button("分析 Analyze", variant="primary")

            with gr.Column(scale=4):
                audio_out = gr.Audio(label="音频播放", type="filepath",
                                     elem_id="audio_player")
                plot_combined = gr.Image(label="可视化分析", type="filepath")
                hl_text = gr.HighlightedText(label="控制文本 / Control Text")
                df_out = gr.Dataframe(
                    headers=["Start(s)", "End(s)", "Dur(s)", "Type", "Text", "Controls"],
                    label="时间段详情", wrap=True,
                )

        # --- Event bindings ---
        _analyze_inputs = ([audio_state, text_box]
                           + list(ext_checks.values())
                           + [chk_word, chk_clause, chk_sentence])
        _analyze_outputs = [audio_out, plot_combined, hl_text, df_out]

        file_dd.change(
            fn=on_file_select, inputs=file_dd,
            outputs=[text_box, audio_out, audio_state,
                     plot_combined, hl_text, df_out],
        )

        mic_recorder.stop_recording(
            fn=on_recording_done, inputs=[mic_recorder],
            outputs=[text_box, save_lang, audio_state, audio_out, save_name],
        ).then(
            fn=analyze_and_plot,
            inputs=_analyze_inputs,
            outputs=_analyze_outputs,
        )

        mic_upload.upload(
            fn=on_recording_done, inputs=[mic_upload],
            outputs=[text_box, save_lang, audio_state, audio_out, save_name],
        ).then(
            fn=analyze_and_plot,
            inputs=_analyze_inputs,
            outputs=_analyze_outputs,
        )

        save_btn.click(
            fn=on_save_recording,
            inputs=[audio_state, text_box, save_lang, save_name],
            outputs=[file_dd],
        )

        btn.click(
            fn=analyze_and_plot,
            inputs=_analyze_inputs,
            outputs=_analyze_outputs,
        )

        app.load(
            fn=on_file_select, inputs=[file_dd],
            outputs=[text_box, audio_out, audio_state,
                     plot_combined, hl_text, df_out],
        )

    return app


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _ensure_ssl_cert():
    """Generate a self-signed certificate for HTTPS (enables microphone in browsers)."""
    cert_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".ssl")
    cert_file = os.path.join(cert_dir, "cert.pem")
    key_file = os.path.join(cert_dir, "key.pem")
    if os.path.isfile(cert_file) and os.path.isfile(key_file):
        return key_file, cert_file
    os.makedirs(cert_dir, exist_ok=True)
    import subprocess
    subprocess.run([
        "openssl", "req", "-x509", "-newkey", "rsa:2048",
        "-keyout", key_file, "-out", cert_file,
        "-days", "365", "-nodes",
        "-subj", "/CN=localhost",
    ], check=True, capture_output=True)
    return key_file, cert_file


def main():
    parser = argparse.ArgumentParser(description="Speech-Text-Render Gradio Visualizer")
    parser.add_argument("--device", default="cpu", help="torch device (cpu / cuda)")
    parser.add_argument("--port", type=int, default=8080, help="server port")
    parser.add_argument("--host", default="0.0.0.0", help="server host")
    parser.add_argument("--share", action="store_true", help="create public link")
    parser.add_argument("--no-ssl", action="store_true", help="disable auto HTTPS")
    args = parser.parse_args()

    global DEVICE
    DEVICE = args.device

    app = build_app()

    launch_kwargs = dict(
        server_name=args.host, server_port=args.port, share=args.share,
    )
    if not args.no_ssl:
        try:
            key_file, cert_file = _ensure_ssl_cert()
            launch_kwargs["ssl_keyfile"] = key_file
            launch_kwargs["ssl_certfile"] = cert_file
            launch_kwargs["ssl_verify"] = False
            print(f"\n>>> HTTPS 已启用，请使用 https://localhost:{args.port} 访问")
            print(">>> 浏览器会提示证书不安全，点击'高级'->'继续访问'即可")
            print(">>> 如需禁用 HTTPS，添加 --no-ssl 参数\n")
        except Exception as e:
            print(f"\n>>> 自签名证书生成失败 ({e})，使用 HTTP 模式")
            print(f">>> 麦克风录音需使用 http://localhost:{args.port} 访问\n")

    app.launch(**launch_kwargs)


if __name__ == "__main__":
    main()
