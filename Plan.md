# Speech ControlNet — 智能语音分析与控制引擎 执行计划

> **项目定位**: 从已有音频中提取结构化控制符 (情感/语速/音量/音高/停顿/重读/风格)，为可控 TTS 提供训练数据管线和语音风格迁移能力。类比图像领域的 ControlNet。
>
> **硬件环境**: RTX 5070 Laptop (8GB VRAM), 1TB 存储, WSL2
>
> **目标 TTS 模型**: CosyVoice 3.0 (Fun-CosyVoice3-0.5B)

---

## 训练策略: ControlNet 启发的渐进式方案

本项目的"提取侧"（从音频中提取控制符）类似图像 ControlNet 的"条件提取器"（Canny/Depth/Pose），当前设计已经合理。但在"注入侧"（如何将控制符注入 TTS 模型），ControlNet 的设计原则提供了重要指导。

TTS 领域已有两项直接相关工作:
- **TTS-CtrlNet** (arXiv 2507.04349): 冻结预训练 TTS，训练可训练副本处理情感条件信号，支持帧级时变情感控制
- **TTS-Hub** (ICLR 2026): 每个控制属性训练一个独立 LoRA 模块，推理时通过算术组合融合多属性控制

**三个核心设计原则**:
1. **不动基座模型** — 全量微调有退化风险，应用 LoRA/Adapter 只训练控制层
2. **控制维度应模块化** — 每个维度 (speed/volume/pitch/emotion) 训练独立模块，避免互相干扰
3. **渐进式验证** — 先用简单方案跑通闭环，再升级训练架构

**本项目采用渐进式策略**:

| 阶段 | 方案 | 目的 |
|------|------|------|
| v1 (阶段三) | 文本 Token 注入 + CosyVoice 3 LoRA 微调 | 最快验证"提取->注入->合成"闭环 |
| v2 (阶段三升级) | 模块化 LoRA (每个控制维度独立 LoRA) + 算术组合 | 提升控制精度，降低维度间干扰 |
| v3 (远期) | TTS-CtrlNet 式特征级注入 | 支持时变/帧级控制 (如渐快、语调曲线) |

v1 -> v2 的升级判定: 阶段三 3.6 效果评估后，若控制精度不足或维度间干扰明显，则启动 v2。

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
- **GPU 依赖**: 完整 pipeline 运行需要 CUDA GPU (Whisper 模型加载依赖 CUDA)，无 GPU 环境下可通过修改 config `device='cpu'` 降级运行

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

- [ ] **2.1.1** 创建 `core/calibrator.py`，实现 `PopulationCalibrator` 类
  - 接受 `dataset_config.yaml` 配置 (path, lang, style)
  - 扫描数据集中所有音频，提取原始特征值 (speed, energy, f0)
  - 按 `lang_style` 分组计算 mean/std/n
  - 支持 Welford 增量更新算法
  - 输出 `population_stats.json`
- [ ] **2.1.2** 创建 `dataset_config.yaml` 模板和说明
- [ ] **2.1.3** 内置 `default_population_stats.json` (初始可用手动经验值占位，阶段三校准后替换)
- [ ] **2.1.4** 编写校准测试脚本，在 `examples/audios/` 上验证校准流程

> 完成后检查: 校准脚本能正常运行，输出合理的统计量

### 2.2 BaseExtractor z-score 框架

- [ ] **2.2.1** 在 `BaseExtractor` 中增加 `population_stats` 加载逻辑 (从 json 文件)
- [ ] **2.2.2** 实现 `_zscore(value, feature_name, lang_style)` 公共方法
- [ ] **2.2.3** 实现 `_should_annotate(z)` 方法: |z| <= 0.5 返回 False (normal 不标注)
- [ ] **2.2.4** 实现 `_z_to_label(z)` 方法: z-score 到离散标签的映射 (xlow/low/normal/high/xhigh)

> 完成后检查: 单元测试验证 z-score 计算和标签映射正确

### 2.3 重构现有提取器

- [ ] **2.3.1** 重构 `VolumeExtractor`:
  - 增加 LUFS 响度归一化预处理 (pyloudnorm 或 librosa)
  - clause 级: population z-score 标注绝对音量
  - 词级: clause 内 z-score 标注相对音量 (volume_local)
  - 移除 `sentence_duration_threshold`
  - 移除固定阈值 `word_volume_levels` / `sentence_volume_levels`
- [ ] **2.3.2** 重构 `SpeedExtractor`:
  - 作用级别: clause 级 (逗号分句)
  - 从 `PauseExtractor` 获取停顿时间，计算 `effective_duration`
  - 语速用 population z-score 归一化
  - 移除固定参考速度 3.8/4.2 和 `speed_levels`
- [ ] **2.3.3** 优化 `PauseExtractor`:
  - 区分 clause 间停顿 vs clause 内停顿
  - 保持绝对时间阈值方案 (不做 z-score)
- [ ] **2.3.4** 优化 `EmotionExtractor`:
  - 作用级别: sentence 级 — 遍历 `type=="sentence"` 的 segment 提取情感
  - 多情感输出按置信度降序排列
  - 限制最多输出 Top-2 情感
  - neutral 情感不输出
- [ ] **2.3.5** 在 examples 音频上验证重构后各提取器效果，确认各提取器使用了正确的 segment 级别 (clause/sentence/word)

> 完成后检查: 运行 `python core/control_text_builder.py`，对比重构前后的控制符输出，验证密度合理 (~30-40%)

### 2.4 新增 PitchExtractor

- [ ] **2.4.1** 安装 torchfcpe: `pip install torchfcpe`
- [ ] **2.4.2** 创建 `core/feature_extractor/pitch_extractor.py`
  - 逐帧提取 F0 序列 (Hz)
  - clause 级: clause 内有声帧平均 F0 的 population z-score
  - 词级 (可选): 词内 F0 相对 clause 均值的偏移
  - 滤除无声帧 (F0=0)
- [ ] **2.4.3** 在 examples 音频上验证 Pitch 提取效果
- [ ] **2.4.4** 将 PitchExtractor 注册到 `__init__.py` 和 `ControlBuilder`

### 2.5 新增 EmphasisExtractor

- [ ] **2.5.1** 创建 `core/feature_extractor/emphasis_extractor.py`
  - 词级: 复合 z-score = w1 * z_energy + w2 * z_duration + w3 * z_pitch
  - 默认权重 w1=0.4, w2=0.3, w3=0.3 (可配置)
  - 超过阈值 (如 composite_z > 1.0) 标注为 emphasis
  - 需要依赖 VolumeExtractor 和 PitchExtractor 的中间数据
- [ ] **2.5.2** 设计提取器间数据共享机制 (避免重复计算音频特征)
- [ ] **2.5.3** 在 examples 音频上验证 Emphasis 检测效果
- [ ] **2.5.4** 将 EmphasisExtractor 注册到 `__init__.py` 和 `ControlBuilder`

### 2.6 新增 StyleExtractor

- [ ] **2.6.1** 创建 `core/feature_extractor/style_extractor.py`
  - 从 config 中读取 `style` 字段
  - 不做声学分析，仅在首句首词 (pos=0) 输出 `[style=xxx]`（sentence 级，无需 paragraph 级）
- [ ] **2.6.2** 将 StyleExtractor 注册到 `__init__.py` 和 `ControlBuilder`

### 2.7 更新 ControlBuilder

- [ ] **2.7.1** 更新 `ControlGenerator.generate()`:
  - 支持 z-score 连续值输出 (训练模式)
  - 支持离散标签输出 (推理/展示模式)
  - 实现 `max_controls_per_clause` 密度限制
- [ ] **2.7.2** 更新 `ControlBuilder.build()` 控制符渲染逻辑:
  - 控制符直接插入目标位置，拼到目标词前面
  - sentence 级 (style/emotion) 拼到该句首词前
  - clause 级 (speed/volume/pitch) 拼到该分句首词前
  - word 级 (emphasis/volume_local/break) 拼到目标词前
- [ ] **2.7.3** 在 examples 音频上运行完整 pipeline，检查最终输出文本格式

> **阶段二完成后回顾**:
> 1. 检查各控制符的提取质量和密度是否合理
> 2. 记录 emphasis 的权重是否需要调整
> 3. 确认 z-score 连续值的数值范围是否集中在 [-3, 3] (对 tokenizer 友好)
> 4. 决定是否需要增加更多 examples 测试音频覆盖更多场景

---

## 阶段三: 数据处理 + TTS 微调 (预计 2-3 周)

> 目标: 批量提取训练数据，本地验证，云端训练 CosyVoice 3，评估控制效果

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

### 3.4 本地 CosyVoice 3 验证 (上云前必做)

- [ ] **3.4.1** 本地安装 CosyVoice 3 环境，加载预训练模型，跑通推理 (仅需 ~3GB 显存)
- [ ] **3.4.2** 测试 CosyVoice 3 的原生控制指令格式，理解其输入协议
- [ ] **3.4.3** 调研 CosyVoice 3 的 LoRA 支持情况:
  - 检查训练脚本是否原生支持 LoRA (如 `peft` 集成)
  - 如不支持，评估手动集成 LoRA 的改动量 (通常只需在 LLM/Flow 模块添加 `peft.get_peft_model()`)
  - 确认 LoRA 可注入的模块列表 (attention layers, FFN 等)
  - 记录结论: 使用 LoRA 还是全量微调
- [ ] **3.4.4** 设计控制符到 CosyVoice 3 输入格式的映射方案
  - 确认哪些控制符可以直接映射到 CosyVoice 3 的原生指令
  - 哪些需要扩展 (如 emphasis, pitch)
  - 记录具体映射方案
- [ ] **3.4.5** 将提取的控制符数据转换为 CosyVoice 3 训练格式 (wav.scp, text, utt2spk, spk2utt)
- [ ] **3.4.6** 本地验证数据加载: 用转换后的数据跑 CosyVoice 3 的数据加载脚本，确认无错误
- [ ] **3.4.7** 本地 dry-run: 用 ~10 条音频在 RTX 5070 上跑 1-2 步训练 (batch_size=1, fp16)，确认 loss 在下降
- [ ] **3.4.8** 打包训练数据 + 配置 + 脚本为 tar.gz

> **决策点 A (3.4.3)**: 如果 CosyVoice 3 支持 LoRA，优先采用 LoRA 微调 (训练成本更低，保留基座质量)。如果不支持且改动量大，退回全量微调方案。
>
> **决策点 B (3.4.4)**: 如果映射困难，评估是否改用 F5-TTS 作为备选 (text encoder 更灵活)。在此处做决定并更新后续任务。

### 3.5 AutoDL 云端训练 (v1: 统一 LoRA / 全量微调)

- [ ] **3.5.1** 注册 AutoDL 账号，充值 ~¥300 (含余量)
- [ ] **3.5.2** 创建 RTX 4090 实例 (PyTorch 2.x + CUDA 12.x 镜像)，上传训练数据
- [ ] **3.5.3** RTX 4090 调参实验 (~10h):
  - 如使用 LoRA: rank=16/32, alpha=32/64, target_modules 配置
  - 如使用全量微调: lr/batch_size/warmup 配置
  - 确认 loss 正常下降
  - 确认控制符维度在生效 (对比有/无控制符的合成结果)
- [ ] **3.5.4** 切换到 A100 80GB 实例，正式训练 (~24-30h):
  - 全量数据: AISHELL-3 + ESD + LJSpeech 混合
  - 温度采样 T=2 平衡情感数据比例
  - 每 5000 步保存 checkpoint
- [ ] **3.5.5** 下载最优 checkpoint 到本地，释放云端实例

> 预估云端费用: LoRA 方案 ~¥100-150 (训练更快), 全量微调 ~¥200-230

### 3.6 本地效果评估

- [ ] **3.6.1** 加载微调后 CosyVoice 3 checkpoint 到本地 RTX 5070
- [ ] **3.6.2** 基础对比: 原始模型 vs 微调模型的合成质量 (确认未退化)
- [ ] **3.6.3** 逐维度控制测试:
  - speed: [speed=0.8] vs [speed=1.5] 是否有明显速度差异
  - volume: [volume=-1.5] vs [volume=1.5] 是否有明显音量差异
  - pitch: [pitch=-1.0] vs [pitch=1.0]
  - emotion: [emotion=happy] vs [emotion=sad]
  - style: [style=read] vs [style=emotion]
  - break: [break=0.5] vs [break=1.5]
  - emphasis: 有/无 emphasis 标注的对比
- [ ] **3.6.4** 多维度组合测试: 同时施加 2-3 个控制符，检查是否有维度间干扰
- [ ] **3.6.5** 记录评估结果，标注哪些维度控制效果好/差
- [ ] **3.6.6** 根据评估结果决策:
  - 效果良好 -> 进入阶段四产品化
  - 控制精度不足或维度干扰明显 -> 进入 3.7 模块化 LoRA 升级 (v2)
  - 模型质量退化 -> 检查训练超参/切换 LoRA 方案

> **阶段三 v1 完成后回顾**:
> 1. 总结哪些控制维度生效了，哪些需要改进
> 2. 评估是否需要扩大训练数据 (加入 DiDiSpeech-2 / LibriTTS-R)
> 3. 评估是否需要升级到 v2 模块化 LoRA 方案

### 3.7 模块化 LoRA 升级 (v2, 按需启动)

> 如果 3.6 评估发现控制精度不足或维度间干扰明显，启动此步骤。参考 TTS-Hub (ICLR 2026) 的方案。

- [ ] **3.7.1** 按控制维度拆分训练数据:
  - `data_speed`: 仅保留 speed 控制符标注
  - `data_emotion`: 仅保留 emotion 控制符标注
  - 以此类推每个维度
- [ ] **3.7.2** 为每个维度训练独立 LoRA 模块:
  - `lora_speed.pt` (~几 MB): 只用 speed 标注数据训练
  - `lora_emotion.pt`: 只用 emotion 标注数据训练
  - `lora_volume.pt`, `lora_pitch.pt`, `lora_emphasis.pt`
  - 每个 LoRA 训练成本极低 (RTX 4090 几小时即可)
- [ ] **3.7.3** 实现 LoRA 算术组合推理:
  - `merged_lora = α * lora_speed + β * lora_emotion + γ * lora_pitch`
  - α/β/γ 为各维度权重，默认均为 1.0
  - 支持动态调节权重控制各维度的影响强度
- [ ] **3.7.4** 对比 v1 (统一微调) vs v2 (模块化 LoRA) 的控制效果
- [ ] **3.7.5** 如果 v2 明显优于 v1，替换为 v2 作为最终方案

> **模块化 LoRA 的优势**: 每个控制维度独立训练、独立迭代；新增控制维度无需重训所有模块；推理时可自由组合；保留基座模型 100% 质量。
>
> 预估额外云端费用: ~¥50-80 (每个 LoRA 训练仅需 2-3h)

---

## 阶段四: 产品化扩展 (后续规划)

> 此阶段在阶段三完成并验证效果后启动，具体计划根据阶段三的评估结果调整。

### 4.1 语音风格迁移模块

- [ ] **4.1.1** 实现 `StyleTransfer` 类:
  - 输入: 参考音频 + 新文本
  - 流程: 分析参考音频 -> 提取控制符 -> 应用到新文本 -> CosyVoice 3 合成
- [ ] **4.1.2** 支持跨说话人风格迁移 (保留 A 的风格控制符，用 B 的声音合成)
- [ ] **4.1.3** 编写风格迁移的 API 和 CLI 接口

### 4.2 交互式语音编辑器 UI (远期)

- [ ] **4.2.1** 设计 UI 原型: 波形 + 控制符时间轴
- [ ] **4.2.2** 选择 UI 框架 (如 Gradio / Streamlit / Electron)
- [ ] **4.2.3** 实现控制符可视化层
- [ ] **4.2.4** 实现拖拽编辑控制符值
- [ ] **4.2.5** 集成 CosyVoice 3 实时合成预览

### 4.3 TTS-CtrlNet 式特征级注入 (v3, 远期)

> 参考 TTS-CtrlNet (arXiv 2507.04349) 的架构，实现帧级/时变控制。仅在 v2 模块化 LoRA 仍无法满足精度需求时考虑。

- [ ] **4.3.1** 复制 CosyVoice 3 的 Flow 编码器作为控制分支
- [ ] **4.3.2** 使用零初始化连接控制分支到主模型中间层
- [ ] **4.3.3** 训练控制分支处理连续控制信号 (F0 曲线、能量包络等)
- [ ] **4.3.4** 验证帧级时变控制效果 (如渐快、渐强、语调曲线)

### 4.4 大规模数据扩展

- [ ] **4.4.1** 加入 DiDiSpeech-2 (800h) + LibriTTS-R (585h)
- [ ] **4.4.2** 更新 PopulationCalibrator 统计量
- [ ] **4.4.3** 在扩大数据上重新训练
- [ ] **4.4.4** 加入 EMILIA 扩展到万小时级别

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
| CosyVoice 3 | github | TTS 合成 + 训练 |

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
| pitch | `pitch` | clause | z-score 连续值 | torchfcpe F0 + population z-score |
| volume_local | `volume_local` | word | z-score 连续值 | clause 内相对能量 z-score |
| emphasis | `emphasis` | word | 布尔 | 复合 z-score (energy+duration+F0) |
| break | `break` | word 间 | 秒数 | 词间时间间隔 |

词级速度/音高变化由 emphasis 的 duration 和 F0 分量间接覆盖，v1 不单独提取词级 speed/pitch。如后续评估不足可扩展 `speed_local` / `pitch_local`。

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

### 训练策略演进路线

| 版本 | 方案 | 触发条件 | 预估成本 |
|------|------|---------|---------|
| v1 | 文本 Token 注入 + LoRA/全量微调 | 阶段三默认执行 | ¥100-230 |
| v2 | 模块化 LoRA + 算术组合 | v1 控制精度不足或维度干扰 | 额外 ¥50-80 |
| v3 | TTS-CtrlNet 特征级注入 | v2 仍无法满足帧级时变控制 | 待评估 |

### 参考文献

- TTS-CtrlNet: Time varying emotion aligned TTS with ControlNet (arXiv 2507.04349)
- TTS-Hub: Leveraging Modular LoRAs and Arithmetic Composition for Controllable TTS (ICLR 2026)
- ControlSpeech: Decoupled Codec for Controllable Speech Synthesis (controlspeech.github.io)
