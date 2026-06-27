# Speech ControlNet — 智能语音分析与控制符提取引擎 执行计划

> **项目定位**: 从已有音频中提取结构化控制符 (情感/语速/音量/音高/停顿/重读/风格)，为可控 TTS 提供训练数据管线。类比图像 ControlNet 的"条件提取器"（提取侧）。
>
> **硬件环境**: RTX 5070 Laptop (8GB VRAM), 1TB 存储, WSL2
>
> **下游项目**: [ControllableCosyVoice](docs/contrable_cosyvoice_plan.md) — 基于本项目提取的控制符，对 CosyVoice 3 进行 LoRA 微调（注入侧）

---

## 项目架构: 提取侧 + 注入侧分离

本项目专注于 **"提取侧"** — 从音频中提取结构化控制符，类似图像 ControlNet 的"条件提取器"（Canny/Depth/Pose）。**"注入侧"** — 将控制符注入 TTS 模型进行训练和合成，已拆分为独立项目 [ControllableCosyVoice](docs/contrable_cosyvoice_plan.md)。

| 项目 | 职责 | 阶段 |
|------|------|------|
| **speech_text_render** (本项目) | 音频对齐、特征提取、控制符生成、批量处理、数据集校准 | 阶段一~三 |
| **ControllableCosyVoice** (下游项目) | CosyVoice 3 集成、LoRA 训练、效果评估、产品化 | 独立计划 |

**拆分理由**: 提取侧是通用工具库，可服务于任何 TTS 模型（CosyVoice / F5-TTS / GPT-SoVITS 等）；注入侧的 CosyVoice 依赖（WeNet、peft、accelerate 等）与提取侧完全不同，独立管理更清晰。

---

## 使用说明

1. 按顺序逐项执行子任务，每完成一项将 `[ ]` 改为 `[x]`
2. 完成每个子任务后，检查是否需要根据实际执行情况更新后续任务的说明
3. 如果某项任务执行中发现问题或有更好方案，在该任务下方添加 `> 执行备注: ...` 记录
4. 每个大阶段完成后做一次整体回顾，决定是否调整后续阶段的计划

### 执行环境要求

- **conda 环境**: 所有任务必须在 `speech_text_render` conda 环境下执行
  - 激活方式: `conda activate speech_text_render`
  - 或使用: `conda run -n speech_text_render <command>`
- **LD_LIBRARY_PATH**: WSL2 环境下 `llvmlite` 可能因 `libstdc++` 版本不匹配报错，需设置:
  ```bash
  export LD_LIBRARY_PATH=/home/liuyoude/miniforge3/envs/speech_text_render/lib:$LD_LIBRARY_PATH
  ```
- **GPU 依赖**: 完整 pipeline 运行需要 CUDA GPU (Qwen3 模型加载依赖 CUDA)，无 GPU 环境下可通过修改 config `device='cpu'` 降级运行

---

## 阶段一: 代码修复 + 对齐层升级 (预计 1-2 周)

> 目标: 修复现有 Bug，工程规范化，用 Qwen3 系列替换 Whisper + MFA

### 1.1 Bug 修复

- [x] **1.1.1** 修复 `base_extractor.py` L21: `extract()` 返回类型标注从 `Dict` 改为 `List[Dict]`
- [x] **1.1.2** 修复 `control_text_builder.py` L40: `generate()` 返回类型标注从 `List[Dict]` 改为 `Dict[int, str]`
- [x] **1.1.3** 修复 `emotion_extractor.py` L39: `self.sentence_level = "True"` 改为 `self.sentence_level = True`
- [x] **1.1.4** 修复 `control_text_builder.py` L55: 控制符合并逻辑，改用列表收集再统一格式化，避免字符串裁剪风险

> 完成后检查: 运行 `python core/control_text_builder.py` 确认无报错
>
> 执行备注: 1.1.1-1.1.4 已全部完成。

### 1.2 工程规范化

- [x] **1.2.1** 移除所有文件中的 `sys.path.append`，确认 `pyproject.toml` 配置正确，使用 `pip install -e .` 作为唯一导入方式
- [x] **1.2.2** 引入 `logging` 模块，替换 `control_text_builder.py` L48 等处的 `print()` 调试输出
- [x] **1.2.3** 抽取 `SegmentFixer._calculate_perceptual_energy` 和 `VolumeExtractor._calculate_perceptual_energy` 的重复代码为 `core/utils/audio_utils.py` 公共函数

> 完成后检查: `pip install -e . && python -c "from core.feature_extractor import *"` 正常导入
>
> 执行备注: 1.2.1-1.2.3 已全部完成。新增 `core/__init__.py` 和 `core/utils/__init__.py` 确保包结构正确。共移除 8 个文件中的 `sys.path.append`，3 个 core/ 文件引入 logging 替换 print()（`__main__` 块中的 print 保留），`_calculate_perceptual_energy` 抽取至 `core/utils/audio_utils.py`。

### 1.3 对齐层升级: 集成 Qwen3（先兼容共存）

- [x] **1.3.1** 安装 qwen-asr: `pip install qwen-asr`，更新 `requirements.txt` 和 `pyproject.toml` 添加 `qwen-asr`
- [x] **1.3.2** 在 examples 音频上测试 Qwen3-ASR 转录效果（中文 + 英文），记录 WER
- [x] **1.3.3** 在 examples 音频上测试 Qwen3-ForcedAligner 对齐效果，与当前 Whisper/MFA 逐条对比时间戳精度
- [x] **1.3.4** 实现 `Qwen3Aligner` 类，封装 Qwen3-ForcedAligner，输出 `List[TimeSegment]`（暂用现有 word/sentence/paragraph 格式，粒度重构在 1.5 统一做）
- [x] **1.3.5** 实现 `Qwen3ASR` 类，封装 Qwen3-ASR，支持转录 + 语言检测
- [x] **1.3.6** 在 `SpeechTextAligner` 中新增 `aligner_backend` 参数（`"qwen3"` / `"whisper_mfa"`），默认 `"qwen3"`，保留原有 Whisper/MFA 作为 fallback
- [x] **1.3.7** 保留 `SegmentFixer` 作为可选后处理步骤
- [x] **1.3.8** 在全部 examples 音频上对比 Qwen3 与 Whisper/MFA 的对齐结果，记录详细对比报告

> 完成后检查: 两种 backend 都能正常运行，Qwen3 效果不劣于 Whisper/MFA
>
> **决策点**: 如果 Qwen3 在所有测试音频上效果达标，后续阶段默认使用 Qwen3。如果部分场景不如 MFA，记录具体 case 并保留双后端。
>
> 执行备注: 1.3.1-1.3.8 已全部完成。安装 `qwen-asr==0.0.6`，选用 0.6B 模型 (ASR: `Qwen/Qwen3-ASR-0.6B`, Aligner: `Qwen/Qwen3-ForcedAligner-0.6B`)。ASR 在英文和中文情感音频上 WER 接近 0，中文新闻文本因同音字存在少量差异但不影响使用。ForcedAligner 对齐精度高（中文逐字、英文逐词），平均每条 0.479s，支持中英混合文本。新增 `Qwen3Aligner`、`Qwen3ASR` 类，`SpeechTextAligner` 支持 `aligner_backend="qwen3"/"whisper_mfa"` 切换，延迟加载模型。**决策: Qwen3 在全部测试音频上效果达标，且支持更多语言、不依赖 MFA 外部工具，后续阶段默认使用 Qwen3。保留 whisper_mfa 后端作为 fallback。**

### 1.4 清理旧依赖（确认 Qwen3 效果后执行）

- [x] **1.4.1** 移除 Whisper 相关代码: `WhisperAligner`, `_convert_format`, `_correct_text_Errors`, `_dynamic_align_words`
- [x] **1.4.2** 移除 MFA 相关代码: `MFAAligner`, `mfa_model/` 目录
- [x] **1.4.3** 更新 `requirements.txt` 和 `pyproject.toml`: 移除 `openai-whisper`、`textgrid`、`pypinyin`
- [x] **1.4.4** 更新 `setup.sh`: 移除 MFA 和 Whisper 相关安装步骤
- [x] **1.4.5** 简化 `SpeechTextAligner`，移除 `aligner_backend` 切换逻辑和语言分支

> 执行备注: 1.4.1-1.4.5 已全部完成。移除 `WhisperAligner`（~240 行）、`MFAAligner`（~145 行）及其专用导入（`whisper`, `textgrid`, `pypinyin`, `subprocess`, `tempfile`）。删除 `mfa_model/` 目录（3 文件）。`SpeechTextAligner` 简化为仅使用 Qwen3 后端，移除 `aligner_backend` 参数。同步更新了 `__init__.py`、`requirements.txt`、`pyproject.toml`、`setup.sh`、示例脚本（`test_audio_aligner.py`、`collect_emotional_audio.py`、`web_visualizer.py`）和 `README.md`。删除了已无意义的 `test_qwen3_comparison.py`。验证 `pip install -e .` 和导入测试均通过。

### 1.5 分段粒度重构: paragraph/sentence → word/clause/sentence

> 在 Qwen3 替换并清理旧代码后执行，只需改 Qwen3Aligner + SegmentFixer + 提取器 + ControlBuilder，无需碰已删除的 Whisper/MFA 代码。

- [x] **1.5.1** `text_normalizer.py`: 拆分 `SENTENCE_END_CHARS` 为两个集合
  - `CLAUSE_END_CHARS`: `。！？.!?` + `，,；;`（含句终符，clause 在任何标点处切分）
  - `SENTENCE_END_CHARS`: 仅 `。！？.!?`（sentence 只在句终符切分）
  - 移除 `、`（顿号，过碎）和 `…`（省略号，非边界）
- [x] **1.5.2** `audio_aligner.py` — `TimeSegment`: type 改为三级 `word` / `clause` / `sentence`，移除 `paragraph`
- [x] **1.5.3** `Qwen3Aligner` 的 segment 合并逻辑：
  - 遇到 `CLAUSE_END_CHARS` 中的标点 → 生成一个 clause 级 segment
  - 遇到 `SENTENCE_END_CHARS` 中的标点 → 同时生成一个 sentence 级 segment
  - 不再生成 paragraph 级 segment
- [x] **1.5.4** `SegmentFixer.fix_segments`: 将 `seg.type == "sentence"` 改为 `seg.type == "clause"`，移除 paragraph 相关逻辑
- [x] **1.5.5** `plot_alignment`: 更新可视化中的 type 引用和颜色映射
- [x] **1.5.6** 各提取器适配: 检查 `speed_extractor.py` / `volume_extractor.py` / `pause_extractor.py` / `emotion_extractor.py` 中遍历 segment 的 type 过滤条件，改为使用正确的 clause 或 sentence 级别
- [x] **1.5.7** `control_text_builder.py` — `ControlBuilder.build()`: 更新 `seg.type == 'word'` 的拼接逻辑，确保 clause/sentence 级 segment 不参与文本拼接
- [x] **1.5.8** 在 examples 音频上运行完整 pipeline，验证三级 segment 正确生成，控制符输出格式正常

> 完成后检查: 输出中 clause 在逗号处切分，sentence 在句号处切分，无 paragraph 级 segment
>
> **如果 1.4 被跳过（保留双后端）**: 需额外适配 `WhisperAligner._convert_format` 和 `MFAAligner._merge_to_sentences` 的粒度输出。
>
> 执行备注: 1.5.1-1.5.8 已全部完成。`SENTENCE_END_CHARS` 拆分为 `SENTENCE_END_CHARS`（仅句终符 `。！？.!?`）和 `CLAUSE_END_CHARS`（句终符 + `，,；;`），移除了 `、` 和 `…`。`TimeSegment.type` 改为三级 `word/clause/sentence`，移除 `paragraph`。`Qwen3Aligner._items_to_segments` 重写为在 `CLAUSE_END_CHARS` 处生成 clause、在 `SENTENCE_END_CHARS` 处同时生成 sentence，合并方法重命名为 `_create_group_segment`。`SegmentFixer` 适配 clause 级，增加 `first_clause_word_seg`/`last_clause_word_seg` 跟踪。`plot_alignment` 新增 clause 绿色虚线标注。`SpeedExtractor` 改为 clause 级触发，`VolumeExtractor` 改为按 clause 分组（重构为基于实际索引的 `_group_by_clause`），`EmotionExtractor` 保持 sentence 级但跳过 clause 段，`PauseExtractor` 无需修改。`web_visualizer.py` 中 paragraph 全部替换为 clause，更新 `_is_clause_start`、显示选项和绘图逻辑。单元测试验证：中文在逗号处正确切分 clause（2 clauses, 1 sentence），英文在 `,` `.` `?` 处正确切分（3 clauses, 2 sentences），无 paragraph 输出。

> **阶段一完成后回顾**: 检查分段粒度是否合理（clause 在逗号处、sentence 在句号处）。检查 Qwen3 对齐精度是否满足后续提取器需求。

---

## 阶段二: 控制符提取重构 + 新增 (预计 1-2 周)

> 目标: 实现 population z-score 归一化框架，重构现有提取器，新增 Pitch/Emphasis/Style 提取器

### 2.1 PopulationCalibrator 实现

- [x] **2.1.1** 创建 `core/calibrator.py`，实现 `PopulationCalibrator` 类
  - 接受 `dataset_config.yaml` 配置 (path, lang, style)
  - 扫描数据集中所有音频，提取原始特征值 (speed, energy)
  - 按 `lang_style` 分组计算 mean/std/n
  - 支持 Welford 增量更新算法
  - 输出 `population_stats.json`
- [x] **2.1.2** 创建 `dataset_config.yaml` 模板和说明
- [x] **2.1.3** 内置 `default_population_stats.json` (初始可用手动经验值占位，阶段三校准后替换)
- [x] **2.1.4** 编写校准测试脚本，在 `examples/audios/` 上验证校准流程

> 完成后检查: 校准脚本能正常运行，输出合理的统计量
>
> 执行备注: 2.1.1-2.1.4 已全部完成。实现了 `WelfordAccumulator`（Welford 增量统计器）+ `PopulationCalibrator`（数据集扫描与特征提取）。提取 6 类特征：clause 级 4 类（speed、energy、loudness_lufs、duration）+ word 级 2 类（word_duration、word_energy）。F0 相关特征（f0_mean、f0_std、f0_range、word_f0）已在 2.4 中移除——pitch 采用 speaker-relative z-score，population 级 F0 统计对混合说话人数据集无意义。speed 使用 effective_duration（去除 >0.2s 停顿）；LUFS 使用 pyloudnorm（<0.4s 片段跳过）。新增依赖 `torchfcpe`、`pyyaml`、`pyloudnorm`。在 examples 的 20 条音频上校准通过，4 组（zh_read/en_read/zh_emotion/en_emotion）均输出合理统计量。

### 2.2 BaseExtractor z-score 框架

- [x] **2.2.1** 在 `BaseExtractor` 中增加 `population_stats` 加载逻辑 (从 json 文件)
- [x] **2.2.2** 实现 `_zscore(value, feature_name, lang_style)` 公共方法
- [x] **2.2.3** 实现 `_should_annotate(z)` 方法: |z| <= 0.5 返回 False (normal 不标注)
- [x] **2.2.4** 实现 `_z_to_label(z)` 方法: z-score 到离散标签的映射 (xlow/low/normal/high/xhigh)

> 完成后检查: 单元测试验证 z-score 计算和标签映射正确
>
> 执行备注: 2.2.1-2.2.4 已全部完成。`BaseExtractor.__init__` 新增 `population_stats` 加载（默认 `core/default_population_stats.json`，通过 `__file__` 定位）和 `_style` 配置（默认 `"read"`）。新增 `_zscore(value, feature_name, lang_style)` 计算 `(value-mean)/std`（含 None 安全保护）、`_get_lang_style(lang)` 拼接 `lang_style` 键、`_should_annotate(z)` 判断 `|z|>0.5`、`_z_to_label(z, labels)` 5 级映射（支持自定义标签元组，如 SpeedExtractor 传 `("xslow","slow","normal","fast","xfast")`）。`__main__` 内联测试全部通过。向后兼容：现有 4 个子类无需修改。

### 2.3 重构现有提取器

- [x] **2.3.1** 重构 `VolumeExtractor`:
  - clause 级: A-weighting 感知能量 + population z-score 标注绝对音量
  - 移除固定阈值 `word_volume_levels` / `sentence_volume_levels` / `sentence_duration_threshold`
  - ~~词级 volume_local 已移除~~：纯能量单维度信号噪声大，将由 2.5 EmphasisExtractor（复合 z-score: energy+duration+F0）替代
- [x] **2.3.2** 重构 `SpeedExtractor`:
  - 作用级别: clause 级 (逗号分句)
  - 计算 `effective_duration`（去除 >0.2s 停顿）
  - 语速用 population z-score 归一化
  - 移除固定参考速度 3.8/4.2 和 `speed_levels`
- [x] **2.3.3** 优化 `PauseExtractor`:
  - 区分 clause 间停顿 vs clause 内停顿（`_build_word_to_clause_map`）
  - 保持绝对时间阈值方案 (不做 z-score)
- [x] **2.3.4** 优化 `EmotionExtractor`:
  - 作用级别: sentence 级 — 遍历 `type=="sentence"` 的 segment 提取情感
  - 多情感输出按置信度降序排列，限制 Top-2，neutral/other/unknow 不输出
- [x] **2.3.5** 在 examples 音频上验证重构后各提取器效果，确认各提取器使用了正确的 segment 级别 (clause/sentence/word)

> 完成后检查: 运行 `python core/control_text_builder.py`，对比重构前后的控制符输出，验证密度合理 (~30-40%)
>
> 执行备注: 2.3.1-2.3.5 已全部完成。共性改动：`_group_by_clause` 提升至 `BaseExtractor` 公共静态方法；新增 `_Z_CLAMP=3.0` z-score 截断 + `_clamp_z()` / `_format_z()` 方法支持数值/标签双模式输出（由 `number_control` 配置切换）。VolumeExtractor 只保留 clause 级 population z-score（energy 维度），**移除了 volume_local**（词级 clause 内 z-score），原因：纯能量单维度信号噪声大、与计划中的 EmphasisExtractor 高度冗余，词级重读判断应由 2.5 的复合 z-score（energy+duration+F0）统一处理。SpeedExtractor 使用 `effective_duration`（去除 >0.2s 停顿）+ population z-score。PauseExtractor 新增 `_build_word_to_clause_map` 区分 inter/intra clause 停顿。EmotionExtractor 简化为 sentence 级、Top-2、降序排列、过滤 neutral/other/unknow。`default_population_stats.json` 已用 examples 音频校准值替换占位值（n=5~155，Stage 3 大规模校准后再更新）。web_visualizer 同步移除 volume_local 相关代码。

### 2.4 新增 PitchExtractor

- [x] **2.4.1** 确认 torchfcpe 已安装（2.1 中已安装 `torchfcpe`）
- [x] **2.4.2** 创建 `core/feature_extractor/pitch_extractor.py`
  - 逐帧提取 F0 序列 (Hz)，使用 `core/utils/audio_utils.extract_f0()` (FCPE)
  - clause 级: **speaker-relative z-score**（非 population z-score）
  - 词级: 不实现，由 2.5 EmphasisExtractor 处理
  - 滤除无声帧 (F0=0)，有声帧 < 5 帧则跳过
- [x] **2.4.3** 在 examples 音频上验证 Pitch 提取效果
- [x] **2.4.4** 将 PitchExtractor 注册到 `__init__.py` 和 `ControlBuilder`

> 执行备注: 2.4.1-2.4.4 已全部完成。**关键决策: pitch 采用 speaker-relative z-score 而非 population z-score**，原因：F0 高度依赖说话人身份（男 85-180Hz，女 165-255Hz），混合说话人的 population z-score 会退化为性别/说话人分类器，不反映语调意图。实现方式：每个音频文件提取全局 F0，计算 speaker baseline（mean, std），每个 clause 的 z = (clause_f0_mean - speaker_f0_mean) / speaker_f0_std，speaker_f0_std 设下限 10.0Hz 防止极短音频 z-score 爆炸。F0 序列和 speaker baseline 缓存到实例变量供后续复用。在 examples 20 条音频上测试：34 clauses 中 4 个被标注（11.8%），标注语义合理。同步清理了 PopulationCalibrator 和 `default_population_stats.json` 中的 F0 相关字段（`f0_mean`、`f0_std`、`f0_range`、`word_f0`），calibrator 不再依赖 `torchfcpe`/`extract_f0`。

### 2.5 新增 EmphasisExtractor

- [x] **2.5.1** 创建 `core/feature_extractor/emphasis_extractor.py`
  - 词级: 复合 z-score = w1 * z_energy + w2 * z_duration + w3 * z_pitch
  - 默认权重 w1=0.4, w2=0.3, w3=0.3 (可配置)
  - 超过阈值 (如 composite_z > 1.0) 标注为 emphasis
  - 需要依赖 VolumeExtractor 和 PitchExtractor 的中间数据
- [x] **2.5.2** 设计提取器间数据共享机制 (避免重复计算音频特征)
- [x] **2.5.3** 在 examples 音频上验证 Emphasis 检测效果
- [x] **2.5.4** 将 EmphasisExtractor 注册到 `__init__.py` 和 `ControlBuilder`

> 执行备注: 2.5.1-2.5.4 已全部完成。**EmphasisExtractor** 实现词级复合 z-score：`composite_z = (w1*z_energy + w2*z_duration + w3*z_pitch) / w_sum`，默认权重 0.4/0.3/0.3，阈值 1.0（只标注正向强调，`composite_z > 1.0`），缺维时按剩余维度归一化权重。energy/duration 使用 population z-score（`word_energy`/`word_duration` 字段），pitch 使用 speaker-relative z-score（每个音频文件视为一个说话人，不依赖外部说话人 ID）。**提取器间数据共享机制**：在 `BaseExtractor` 中新增 `_shared_context: Dict` 属性和三个共享缓存方法（`_load_audio_shared`、`_get_f0_shared`、`_get_speaker_baseline_shared`），`_get_voiced_f0` 从 PitchExtractor 提升至 BaseExtractor。`ControlGenerator.generate()` 在调用前创建共享 context dict 并赋值给每个提取器，保证 PitchExtractor 先于 EmphasisExtractor 执行时 F0/speaker baseline 可复用。PitchExtractor 和 VolumeExtractor 已迁移使用共享缓存，移除了各自的实例级缓存（`_f0_cache`/`_speaker_cache`/`_audio_cache`/`audio_cache`）。在 20 条 examples 音频上验证：约 7 条出现 emphasis（总计约 15 处），密度 ~5-10%，语义合理（强调词、关键动词、专有名词、中文语境中的英文词被正确检测），sad/whisper 类低能量音频无 emphasis 标注。

### 2.6 新增 StyleExtractor

- [x] **2.6.1** 创建 `core/feature_extractor/style_extractor.py`
  - 从 config 中读取 `style` 字段
  - 不做声学分析，仅在首句首词 (pos=0) 输出 `[style=xxx]`（sentence 级，无需 paragraph 级）
- [x] **2.6.2** 将 StyleExtractor 注册到 `__init__.py` 和 `ControlBuilder`

> 执行备注: 2.6.1-2.6.2 已全部完成。`StyleExtractor` 继承 `BaseExtractor`，复用父类 `self._style`（默认 `"read"`），`load_model()` 为空，`extract()` 遍历 segments 找到第一个 `type=="word"` 的 segment 后输出 `[style=xxx]` 并返回。已注册到 `__init__.py` 和 `control_text_builder.py`（import + `__main__` 测试配置 + extractors 列表首位）。导入测试通过。

### 2.7 更新 ControlBuilder

- [x] **2.7.1** 更新 `ControlGenerator.generate()`:
  - 支持 z-score 连续值输出 (训练模式)
  - 支持离散标签输出 (推理/展示模式)
  - 实现 `max_controls_per_clause` 密度限制
- [x] **2.7.2** 更新 `ControlBuilder.build()` 控制符渲染逻辑:
  - 控制符直接插入目标位置，拼到目标词前面
  - sentence 级 (style/emotion) 拼到该句首词前
  - clause 级 (speed/volume/pitch) 拼到该分句首词前
  - word 级 (emphasis/break) 拼到目标词前
- [x] **2.7.3** 在 examples 音频上运行完整 pipeline，检查最终输出文本格式

> 执行备注: 2.7.1-2.7.3 已全部完成。**ControlGenerator 重构**：新增 `_CONTROL_PRIORITY` 控制符优先级常量（style=0 > emotion=1 > speed=2 > volume=3 > pitch=4 > break=5 > emphasis=6），`generate()` 拆分为三步内部方法：`_extract_all()`（含 shared_context 设置）、`_apply_density_limit()`（按 clause 分组 + 按优先级截断）、`_format_controls()`（按 pos 分组 + 优先级排序 + 独立方括号渲染）。新增 `extract_raw()` 公共方法供 web_visualizer 获取原始控制符列表。新增模块级 `_format_single_control()` 函数，emphasis 标签模式渲染为 `[emphasis]`（省略 `=value`），其余为 `[type=value]`。`max_controls_per_clause` 从 config 读取（默认 None 不限制）。**ControlBuilder.build()** 改用 `.items()` 迭代。**web_visualizer.py** 改用 `gen.extract_raw()` + `ControlGenerator._format_controls()` 替代手动收集和格式化（删除了将 `][` 转为 `,` 的旧逻辑），补充 pitch 类型的 clause 级时间范围查找。**__main__ 测试脚本**新增 `--number` 和 `--max-per-clause` 命令行参数，支持切换训练/推理模式。在 20 条 examples 音频上验证：标签模式输出 `[style=read][emotion=happy]...`（独立方括号，优先级排序），数值模式输出 `[speed=-3.0][volume=-0.89]...`（z-score 截断在 [-3, 3]），emphasis 标签模式 `[emphasis]`、数值模式 `[emphasis=2.22]`。

> **阶段二完成后回顾** (详见 [docs/stage2_review.md](docs/stage2_review.md)):
> 1. 密度 36.2% 在目标范围 (30-40%) 内，无需调整
> 2. Emphasis 权重 w=(0.4,0.3,0.3) threshold=1.0 保持不变，4.1% 密度合理
> 3. Z-score 范围基本集中在 [-3,3]；speed 31% 撞截断是 population stats 样本量不足所致，Stage 3 大数据集校准后自然解决
> 4. 当前 23 条 examples 对开发验证已足够，不增加测试音频
>
> **结论: 阶段二全部任务已完成，可以进入阶段三。**

---

## 阶段三: 批量处理 + 数据集校准 (预计 1-2 周)

> 目标: 实现批量提取脚本，下载并校准训练数据集，为下游项目 ControllableCosyVoice 提供高质量控制符标注数据

### 3.1 批量处理脚本

- [ ] **3.1.1** 创建 `scripts/batch_extract.py`:
  - 读取 `dataset_config.yaml` 中的数据集列表
  - 多进程 + GPU 推理并行
  - 逐条: 对齐 -> 提取控制符 -> 输出带控制符的文本
  - 支持断点续跑 (记录已处理文件)
  - 进度条 + 日志
- [ ] **3.1.2** 创建 `scripts/quality_check.py`:
  - 统计各控制符的分布 (各 level 占比)
  - 统计控制符密度 (每句平均控制符数)
  - 可视化抽检 (随机抽取 N 条，绘制波形 + 控制符标注)

### 3.2 数据集下载与校准

- [ ] **3.2.1** 下载 AISHELL-3 (~30GB)
- [ ] **3.2.2** 下载 ESD (~10GB)
- [ ] **3.2.3** 下载 LJSpeech (~25GB)
- [ ] **3.2.4** 编写 `dataset_config.yaml` 配置 (path, lang, style)
- [ ] **3.2.5** 运行 PopulationCalibrator 校准，生成正式 `population_stats.json`
- [ ] **3.2.6** 检查统计结果是否合理 (speed/energy/f0 的 mean/std)，替换 default_population_stats.json

### 3.3 批量提取控制符

- [ ] **3.3.1** 在 AISHELL-3 上批量提取 (预计 RTX 5070 上 ~数小时)
- [ ] **3.3.2** 在 ESD 上批量提取
- [ ] **3.3.3** 在 LJSpeech 上批量提取
- [ ] **3.3.4** 运行 `quality_check.py` 验证标注质量:
  - 各 level 占比是否符合正态分布预期
  - 控制符密度是否在 30-40%
  - 抽检可视化是否合理
- [ ] **3.3.5** 如有问题，调整 z-score 阈值或提取器参数后重跑

> 完成后检查: 输出带控制符标注的完整训练数据集，格式正确，质量合格
>
> **阶段三完成后回顾**:
> 1. 各 level 占比是否符合正态分布预期
> 2. 控制符密度是否在 30-40%
> 3. speed z-score 截断率是否从 31% 下降到合理水平 (< 10%)
> 4. 发布 v1.0 版本标签 (git tag v1.0.0)，供 ControllableCosyVoice 项目作为上游依赖引用
>
> **本项目至此完成。** 后续的 CosyVoice 3 集成、LoRA 训练、效果评估、产品化等工作，已拆分至独立项目 [ControllableCosyVoice](docs/contrable_cosyvoice_plan.md)。

---

## 技术参考速查

### 核心依赖

| 组件 | 包名 | 用途 |
|------|------|------|
| Qwen3-ASR | `qwen-asr` | ASR 转录 + 语言检测 |
| Qwen3-ForcedAligner | `qwen-asr` | 文本-音频时间戳对齐 |
| emotion2vec | `funasr` | 句级情感识别 |
| torchfcpe | `torchfcpe` | F0 基频提取 |
| pyloudnorm | `pyloudnorm` | LUFS 响度归一化 |

### 分段层级

| 类型 | 含义 | 切分依据 |
|------|------|---------|
| word | 单词 | 对齐器输出 |
| clause | 分句 | `，` `,` `；` `;` `。` `！` `？` `.` `!` `?` |
| sentence | 语句 | 仅 `。` `！` `？` `.` `!` `?` |

移除了原来的 paragraph 级（无实际用途）。移除了 `、`（顿号，过碎）和 `…`（省略号，非边界）作为切分符。

### 控制符放置与作用域

控制符直接插入目标位置，拼到目标词前面，词间空格由自然拼接产生。**作用域由控制符类型名决定，不依赖格式约定**——模型从训练数据中的类型+位置+声学关联学习每种控制符的影响范围。

| 控制符 | 类型 | 粒度 | 值格式 (训练) | 提取方式 |
|--------|------|------|-------------|---------|
| style | `style` | sentence (首句) | 离散标签 | 数据集配置手动标注 |
| emotion | `emotion` | sentence | 离散标签 | emotion2vec 模型 |
| speed | `speed` | clause | z-score 连续值 | 词数/有效时长 + population z-score |
| volume | `volume` | clause | z-score 连续值 | A-weighting 能量 + LUFS + population z-score |
| pitch | `pitch` | clause | z-score 连续值 | torchfcpe F0 + speaker-relative z-score |
| emphasis | `emphasis` | word | z-score 连续值 | 复合 z-score (energy+duration+F0) |
| break | `break` | word 间 | 秒数 | 词间时间间隔 |

词级速度/音高变化由 emphasis 的 duration 和 F0 分量间接覆盖，不单独提取词级 speed/pitch。如后续评估不足可扩展 `speed_local` / `pitch_local`。

输出示例（中文）：

```
[style=read][emotion=happy]今天 天气 真 好，[speed=1.3]出去 [emphasis]走走 吧。
```

输出示例（英文）：

```
[style=read][emotion=neutral]Hello，[speed=0.9]good [emphasis]morning how are you.
```

### z-score 标签映射

| z-score 区间 | 离散标签 | 是否标注 |
|-------------|---------|---------|
| z < -1.5 | xlow / xslow / whisper | 标注 |
| -1.5 <= z < -0.5 | low / slow / soft | 标注 |
| -0.5 <= z <= 0.5 | normal | **不标注** |
| 0.5 < z <= 1.5 | high / fast / loud | 标注 |
| z > 1.5 | xhigh / xfast / shout | 标注 |

### 下游项目

CosyVoice 3 集成、LoRA 训练、效果评估、产品化等工作，详见 [ControllableCosyVoice 执行计划](docs/contrable_cosyvoice_plan.md)。

### 参考文献

- TTS-CtrlNet: Time varying emotion aligned TTS with ControlNet (arXiv 2507.04349)
- TTS-Hub: Leveraging Modular LoRAs and Arithmetic Composition for Controllable TTS (ICLR 2026)
- ControlSpeech: Decoupled Codec for Controllable Speech Synthesis (controlspeech.github.io)
