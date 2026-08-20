# 工作流

本工具把“发现媒体、字幕/语义拆分、本地 ASR、质量门禁、模型标注、人工复核、导出、训练、模型注册、情绪规划、合成、可选 VC”分为独立阶段。每一步都读取前一步的持久化清单，因此可以检查中间结果、重复运行安全步骤，并在训练前停下来确认。

```text
本地文件或目录
    -> init / ingest
    -> preprocess (sidecar -> embedded -> vision -> silence)
    -> transcribe / quality (SenseVoice + 声学门禁)
    -> label (Gemini；台词、情绪、聚类)
    -> review (人工确认，可断点续作)
    -> export (物化训练集)
    -> train gpt-sovits
    -> train rvc / RVC 后处理
    -> model register / activate
    -> emotion -> reference selection -> synthesize
    -> optional RVC postprocess
```

快速导航：

- [预检与初始化](#0-预检)
- [导入与拆分](#2-导入媒体)
- [Gemini 标注与人工复核](#4-gemini-标注与聚类)
- [导出与训练](#6-导出训练集)
- [成本与副作用](#成本与副作用边界)
- [完整命令参考](#完整命令参考)

命令示例表达的是稳定的工作流概念。不同版本的参数细节以 `voice-dataset <子命令> --help` 为准。

## 0. 预检

创建并激活 Python 环境后，先检查 CLI 和 FFmpeg：

```powershell
voice-dataset --help
voice-dataset init --help
voice-dataset ingest --help
voice-dataset split --help
voice-dataset preprocess --help
voice-dataset transcribe --help
voice-dataset quality --help
voice-dataset label --help
voice-dataset review --help
voice-dataset export --help
voice-dataset train --help
voice-dataset model --help
voice-dataset emotion --help
voice-dataset synthesize --help
voice-dataset status --help
python .\scripts\generate_configs.py --help
ffmpeg -version
```

如果使用 Gemini，优先检查独立敏感配置是否存在；也可以确认当前进程中存在环境变量，
但不要输出密钥值：

```powershell
$workspace = 'D:\voice-workspaces\character-a'
$secrets = Join-Path $workspace 'secrets\credentials.toml'
if (Test-Path -LiteralPath $secrets) {
    Write-Output "[OK] secrets file exists: $secrets"
}
elseif ([string]::IsNullOrWhiteSpace($env:GEMINI_API_KEY)) {
    Write-Output '[WARN] no secrets file and GEMINI_API_KEY is not set'
}
else {
    Write-Output '[OK] GEMINI_API_KEY is set'
}
```

如果使用训练适配器，还应分别在 GPT-SoVITS 和 RVC 工程中验证它们自己的 Python 环境与基础模型。工具只负责验证和编排，不取代上游项目的安装步骤。

## 1. 初始化工作区

每个角色使用一个独立工作区：

```powershell
$workspace = 'D:\voice-workspaces\character-a'
voice-dataset init $workspace
```

初始化是本地、低成本操作，不会扫描媒体、调用 Gemini 或启动训练。它会生成：

- `config/pipeline.toml`：非敏感项目配置，可以提交和评审。
- `secrets/credentials.toml`：API Key、Token 等敏感值。
- `secrets/.gitignore`：忽略敏感目录中的所有内容，仅保留规则本身。

也可以使用独立生成脚本：

```powershell
python .\scripts\generate_configs.py $workspace
```

脚本默认不覆盖任何已有配置。项目模板位于 `examples/project/`，敏感空白模板位于
`examples/secrets/`；两类文件不共用目录。不要在 `config/pipeline.toml` 中写入
API Key。

## 2. 导入媒体

输入可以是单个文件、多个文件或目录。目录会递归扫描：

```powershell
voice-dataset ingest $workspace 'D:\media\character-a'
```

也可以显式传递若干来源；实际多输入语法以帮助为准：

```powershell
voice-dataset ingest --help
```

导入阶段应完成：

1. 按扩展名识别音频和视频。
2. 递归发现目录中的受支持文件。
3. 计算内容指纹，避免重复导入同一内容。
4. 通过 FFmpeg 提取或规范化音轨。
5. 把来源、时长和规范化文件写入清单。

该阶段只读取原始输入，不在原目录移动、重命名或删除文件。
采样率等规范化设置属于来源指纹；同一内容改用新设置重新导入时，工具会使旧边界、切片清单、标签和复核状态失效，但保留旧 WAV 文件以便恢复。

## 3. 拆分语音

拆分只生成时间边界并物化不可变 WAV 切片，不负责情绪分类。

### 本地能量拆分

```powershell
voice-dataset split $workspace --backend energy --modality audio
voice-dataset split $workspace --backend energy --modality video
```

`energy` 使用本地 Python 音频处理：

- 不上传媒体。
- 不消耗模型 API 配额。
- 适合背景干净、句间停顿清楚的素材。
- 只分析音轨；显式选择 `video` 时也只使用 FFmpeg 取得的规范化音频，不读取画面。

应通过配置调整最短语音、最短静音、补边、合并间隔和最大切片时长。算法只在相邻片段中至少有一个过短时才自动合并，避免把两个完整语境粗暴拼接。

### Gemini 多模态拆分

只有明确执行以下命令才会产生上传和 API 消耗：

```powershell
voice-dataset split $workspace --backend gemini --modality audio
voice-dataset split $workspace --backend gemini --modality video
voice-dataset split $workspace --backend gemini --modality auto
```

模式含义：

| 模式 | 输入给拆分器的内容 | 适合场景 |
| --- | --- | --- |
| `audio` | 规范化音频 | 只需听觉语义，上传量较小 |
| `video` | 原始视频或受支持的视频表示 | 画面变化有助于区分说话情境 |
| `auto` | 根据媒体类型和后端选择 | 混合音视频目录 |

`energy` 的 `video` 模式只处理已提取音轨；上表中的原始视频/画面语义仅适用于
Gemini。后端与模式组合会在执行前校验，当前版本支持的确切组合以
`voice-dataset split --help` 为准。

Gemini 只允许返回原始时间轴上的边界。它不能把生成或改写的台词直接写进训练集；边界必须经过范围、顺序、持续时间和重叠校验。

长视频不会整段上传。超过 `[gemini.chunking].threshold_seconds` 后，管线使用本地
FFmpeg 静音检测，在 `target_seconds` 附近选择切点且严格不超过 `max_seconds`，转码为
低分辨率/低码率 MP4 后逐块调用 Gemini。块窗口必须从 0 连续覆盖到源媒体结尾；每块
返回值先通过局部校验，再回映到全局时间轴并再次检查顺序和范围。缓存位于
`<workspace>\state\gemini_chunks`：`manifest.json` 记录完整窗口，`*.segments.json`
记录已经成功的远端边界，因而后续块失败后重跑不会重复前面块的 API 费用。
`reuse_chunks` 控制复用，`keep_chunks=false` 仅清除临时 MP4，不清除可审计的边界缓存。

### 高层 `preprocess`：回退拆分、质量门禁和本地 ASR

```powershell
voice-dataset preprocess `
  <workspace> `
  [<input> ...] `
  [--mode auto|audio|video] `
  [--config <project-config.toml>] `
  [--secrets <credentials.toml>] `
  [--replace] `
  [--asr | --skip-asr] [--force-asr] [--force-quality]
```

当命令收到输入路径时会先递归导入；省略输入时处理工作区中已有来源。每个来源按以下顺序选第一个产生有效边界的策略：

1. 同名或语言后缀的 `.srt/.vtt/.ass/.ssa` sidecar；
2. FFprobe/FFmpeg 发现并提取的首个文本字幕流；
3. 配置为 `splitting.backend=gemini` 时的视频语义拆分；
4. 本地音轨静音/能量拆分。

实际策略、尝试链和失败原因写入 segment provenance。随后声学门禁流式计算时长、RMS、峰值、削顶比和静音比，写入 `manifests/quality.jsonl`。存在质量记录时，未通过条目不会进入导出训练集。

`--asr` 或 `asr.enabled=true` 会延迟加载 SenseVoice，写入 `manifests/asr.jsonl`，并为尚无标签的片段生成可复核台词/情绪草稿：

```powershell
voice-dataset transcribe $workspace
voice-dataset transcribe $workspace --force
voice-dataset quality $workspace
voice-dataset quality $workspace --force
```

`asr` 可选依赖不包含 PyTorch，因为 CUDA 轮子必须按本机驱动单独选择。
如果当前编排器 venv 没有可用的 GPU PyTorch，使用 `--skip-asr` 忽略配置中的
`asr.enabled=true`，再在已安装 PyTorch/FunASR 且能够导入本项目的独立环境执行
`python -m voice_dataset_pipeline transcribe <workspace>`。

`--force` 会重新计算；否则音频哈希、模型或阈值配置未变化的条目直接复用。ASR 相似度是风险提示，最终台词仍由 TUI 人审决定。

`[asr]` 中的 `model_revision`、`vad_revision`、`funasr_version` 和
`modelscope_version` 都参与缓存与严格导出门禁。执行时会先核对已安装库版本；这避免
环境升级后静默复用旧转写。默认 `master` 仍是浮动的 provider revision，不是模型文件
内容哈希；需要位级复现时必须把它替换为服务端支持的不可变 revision，并在同一 ASR
环境中重新运行 `transcribe --force`。

## 4. Gemini 标注与聚类

在 `secrets/credentials.toml` 或当前进程环境变量中设置 Key 后，显式运行：

```powershell
voice-dataset label $workspace --provider gemini
```

该阶段对每个切片生成候选信息：

- 台词转写。
- 情绪类别。
- 从 `review.clusters` 共享词表中选择的聚类名称。
- 置信度或简短判断依据（如果当前适配器提供）。

情绪集合来自角色配置，TUI 的数字顺序与配置完全一致。模型输出只是草稿；人工复核前不要直接训练。

项目配置中的 `gemini.api_key_env` 只保存查找名称，默认是 `GEMINI_API_KEY`。程序先从
独立的 `secrets/credentials.toml` 查找该名称，再回退到当前进程环境变量。项目配置、
状态文件和日志不得持久化 Key；也可通过 `--secrets` 显式选择另一个敏感配置文件。

如果调用失败：

1. 检查敏感配置中的键名是否与 `gemini.api_key_env` 一致，或环境变量是否有效。
2. 检查模型是否对当前账号和地区开放。
3. 检查媒体大小、网络和配额。
4. 重新执行 `label`；已经成功并持久化的条目不应重复计费。

## 5. 人工复核

```powershell
voice-dataset review $workspace
```

每屏显示一个条目的音频路径、当前台词、候选情绪和聚类。操作直接按键，不需要回车：

| 按键 | 操作 |
| --- | --- |
| `0` | 用默认音频播放器播放 |
| `M` | 将当前与下一片段重建为一个新片段，并停留在该条复核 |
| `1`–`N` | 选择配置中对应的情绪并确认条目 |
| `X` | 排除当前条目 |
| `E` | 编辑台词 |
| `K` | 编辑聚类名 |
| `S` | 暂时跳过并放到队尾 |
| `R` | 刷新当前条目 |
| `B` | 撤销上一次操作 |
| `Q` | 保存并退出 |

界面会显示前后相邻台词。`M` 使用规范化源音频从当前起点到下一条终点重新物化 WAV，旧 WAV 不删除；合并后应重新执行 `quality` 和 `transcribe`，再继续导出。

实现应在每次决定后原子写入本地 JSON 状态，至少保留处理顺序、当前位置、最新决定和撤销历史。`Ctrl+C`、终端关闭或播放器异常不应丢失此前决定；再次运行 `review` 会从断点继续。

复核阶段不直接移动切片。这样可以避免 Windows 默认播放器或资源管理器预览占用文件时出现 `WinError 5`。真正的情绪目录和文件名映射在导出阶段创建。

## 6. 导出训练集

```powershell
voice-dataset export $workspace --output 'D:\voice-datasets\character-a'
```

导出只包含没有被排除且已通过人工确认的条目。推荐产物同时提供：

- 按情绪或聚类组织的目录。
- 每个 WAV 旁边的同名 `.txt`，便于在 Windows 资源管理器中看到声音与台词的联系。
- GPT-SoVITS 可消费的标注清单。
- RVC 可消费的纯音频数据集。
- 包含来源哈希、切片哈希、配置哈希和复核状态的清单。

如果输入、配置、边界或复核决定发生变化，应重新导出，并避免复用指纹不一致的旧训练目录。

## 7. GPT-SoVITS 训练

先执行依赖预检并生成计划：

```powershell
voice-dataset train $workspace gpt-sovits
```

计划应列出外部工程、Python 解释器、数据集、基础模型、实验目录、预计执行的上游命令和待生成的权重。此阶段会调用目标 Python 检查依赖并写入计划、生成配置和暂存目录，但不会启动数据预处理或模型训练。

确认以下条件后再授权执行：

- GPT-SoVITS 工程可正常运行。
- 配置中的 Python 属于该工程环境。
- 基础模型版本与训练配置匹配。
- 导出清单和数据指纹是本次复核结果。
- GPU 显存、磁盘空间和输出目录满足要求。

```powershell
voice-dataset train $workspace gpt-sovits --execute
```

具体 epoch、batch size 和版本参数在角色 `config/pipeline.toml` 的
`training.gpt_sovits` 中配置；完整字段见仓库示例，实际底模路径、命令和指纹以生成的
`training-plan.json` 为准。

## 8. RVC 后处理模型训练

RVC 同样先检查计划：

```powershell
voice-dataset train $workspace rvc
```

执行前确认：

- RVC 工程及其独立 Python 环境可运行。
- 预处理、F0 与 HuBERT 特征工具都能从 RVC 工程根目录导入。
- 所需基础生成器、判别器和 HuBERT 模型存在。
- 实验目录与当前数据指纹一致。
- `experiment_name` 未被 RVC 上游的旧日志、权重或索引占用；为防旧特征混入，失败重跑时使用新的实验名。

显式执行：

```powershell
voice-dataset train $workspace rvc --execute
```

成功标准不是只生成 `.index`。必须存在对应的可推理 `.pth` 权重，才能把 GPT-SoVITS 结果送入 RVC 后处理。RVC 与 GPT-SoVITS 必须使用各自独立 Python 环境；合成和后处理分别由其注册解释器中的隔离 worker 完成。

## 9. 模型注册、情绪规划和直接合成

训练成功时可自动登记 GPT/SoVITS 权重：

```powershell
voice-dataset train $workspace gpt-sovits `
  --execute `
  --register-as character_a `
  --activate
```

也可登记已有权重：

```powershell
voice-dataset model register $workspace `
  --name character_a `
  --persona character_a `
  --repository 'D:\GPT-SoVITS' `
  --python 'D:\GPT-SoVITS\.venv\Scripts\python.exe' `
  --gpt 'D:\weights\character-a.ckpt' `
  --sovits 'D:\weights\character-a.pth' `
  --manifest 'D:\dataset\manifest.jsonl' `
  --version v2ProPlus `
  --activate

voice-dataset model list $workspace
voice-dataset model activate $workspace character_a
```

目标文本先转换为统一的 `emotion/intensity/pace/pitch/energy/pause_style` 情绪计划。默认规则分析完全离线。启用兼容网关时必须同时设置 `emotion.provider="openai-compatible"`、实际服务的 `base_url`、该服务可用的 `model`，Token 仍只从独立 secrets 查找；模板中的 `example.invalid` 和 `replace-with-model-id` 是防误调用占位符：

```powershell
voice-dataset emotion $workspace --text '太好了，终于等到你了！'
```

不传参考音频时，参考选择器从注册模型的 reviewed manifest 中优先选择同情绪、3–10 秒、已人工确认的片段；传入显式参考时必须同时给出逐字一致的参考台词：

```powershell
voice-dataset synthesize $workspace `
  --text '你好，很高兴见到你。' `
  --output 'D:\output\character-a.wav'

voice-dataset synthesize $workspace `
  --model character_a `
  --text '你好，很高兴见到你。' `
  --emotion happy `
  --intensity 0.7 `
  --reference 'D:\reference\happy.wav' `
  --reference-text '参考音频中实际说出的台词。' `
  --output 'D:\output\character-a-explicit.wav'
```

模型记录中的 `--python` 是推理环境接口的一部分，不是可选提示。工具通过该解释器启动 worker，再由 worker 在 GPT-SoVITS 仓库中调用 `TTS_Config/TTS.run`；因此不要求 WebUI 或 API 服务常驻，也不会把上游 PyTorch/音频依赖导入主管线进程。旧注册记录如果没有 `python` 字段，需要使用 `model register ... --python <解释器>` 重新登记。

provider worker 由绝对脚本路径启动，入口只依赖 Python 3.10 标准库；PyTorch、NumPy
与上游包只在 provider 进程内加载。每次登记都会封存 GPT/SoVITS/reference manifest、
BERT/CNHubERT、G2PW、语言检测、SV/版本相关声码器，以及可选 RVC
模型/index/HuBERT/RMVPE 的 SHA-256；同时记录 provider Git HEAD、tracked dirty
fingerprint 和仓库内 Python 源码树摘要。自动挑选参考 WAV 时还会核对 manifest 行内的音频
SHA-256。上述输入发生漂移时推理会 fail-closed；旧注册记录必须重新执行 `model register`，
不要手工补写摘要。

### 可选 RVC 后处理

登记模型时追加 `--rvc-repository/--rvc-python/--rvc-model/--rvc-index` 后，可执行：

```powershell
voice-dataset synthesize $workspace `
  --text '固定的 A/B 测试台词。' `
  --postprocess rvc `
  --output 'D:\output\character-a-rvc.wav'
```

原始结果保存为 `character-a-rvc.sovits.wav`。只有 RVC 版本的 ASR 一致性、声学指标、音色相似度和人工 A/B 均胜出时才采用；VC 不是必然增强步骤。

## 成本与副作用边界

| 操作 | 网络/API | 是否启动训练 | 主要写入 |
| --- | --- | --- | --- |
| `init` | 否 | 否 | 工作区骨架、项目配置、Git 忽略的敏感配置 |
| `ingest` | 否 | 否 | 来源清单、规范化媒体 |
| `split --backend energy` | 否 | 否 | 边界、切片 |
| `split --backend gemini` | 是 | 否 | 边界、切片 |
| `preprocess` | Gemini 视觉回退时联网；ASR 模型可能首次下载 | 否 | 来源、边界、切片、质量与可选 ASR |
| `transcribe` | 本地模型可能首次下载 | 否 | ASR 清单 |
| `quality` | 否 | 否 | 质量清单 |
| `label` | 是 | 否 | 候选标签 |
| `review` | 否 | 否 | 人工复核状态 |
| `export` | 否 | 否 | 训练集 |
| `train ...` | 否 | 否 | 目标 Python 依赖预检、暂存数据、生成配置、训练计划；RVC 外部实验目录 |
| `train ... --execute` | 取决于上游 | 是 | 外部实验与模型产物 |
| `emotion` | 仅 openai-compatible | 否 | 只输出情绪计划 |
| `synthesize` | rules/显式 emotion 时否；openai-compatible 自动情绪时是 | 否 | 角色 WAV 与可选 VC WAV |

任何高成本行为都必须来自清楚、独立的用户命令。检查状态和打开 TUI 不会隐式触发 Gemini 或训练；重新生成训练计划会重复预检，但不会启动模型训练。

## 完整命令参考

以下命令均以 Windows PowerShell 为例。路径含空格时必须使用引号。建议先定义变量，
减少复制时改错角色工作区的风险：

```powershell
$repo = 'D:\voice-dataset-pipeline'
$workspace = 'D:\voice-workspaces\character-a'
$inputPath = 'D:\media\character-a'
$exportPath = 'D:\voice-datasets\character-a'

Set-Location $repo
```

### 查看版本与帮助

```powershell
voice-dataset --version
voice-dataset --help
voice-dataset <子命令> --help
```

可用子命令：

```text
init  ingest  split  preprocess  transcribe  quality  label  review  export
train  model  emotion  synthesize  status
```

`--help`、`--version` 不写入工作区，不调用网络，也不会启动训练。

### 生成项目配置与敏感配置

完整语法：

```powershell
python .\scripts\generate_configs.py `
  [目标工作区] `
  [--overwrite-project] `
  [--overwrite-secrets]
```

参数：

| 参数 | 含义 |
| --- | --- |
| `目标工作区` | 可省略；省略时使用当前目录 |
| `--overwrite-project` | 用默认模板覆盖 `config/pipeline.toml` |
| `--overwrite-secrets` | 用空白模板覆盖敏感配置，会清除已填写的 Token |

安全生成：

```powershell
python .\scripts\generate_configs.py $workspace
```

只重置非敏感项目配置：

```powershell
python .\scripts\generate_configs.py $workspace --overwrite-project
```

同时重置两类配置：

```powershell
# 警告：会把 credentials.toml 重置为空白模板。
python .\scripts\generate_configs.py `
  $workspace `
  --overwrite-project `
  --overwrite-secrets
```

生成产物：

```text
<workspace>\config\pipeline.toml
<workspace>\secrets\.gitignore
<workspace>\secrets\credentials.toml
```

脚本默认保留已有文件。敏感目录的 `.gitignore` 必须且只能保留 `*` 和
`!.gitignore` 两条有效规则，防止后续规则意外重新纳入凭据。

### `init`：初始化工作区

完整语法：

```powershell
voice-dataset init `
  <workspace> `
  [--config <project-config.toml>] `
  [--overwrite-config] `
  [--overwrite-secrets]
```

参数：

| 参数 | 含义 |
| --- | --- |
| `workspace` | 必填；角色工作区根目录 |
| `--config` | 从指定的非敏感 TOML 初始化项目配置 |
| `--overwrite-config` | 覆盖现有项目配置 |
| `--overwrite-secrets` | 将敏感配置重置为空白模板，会清除已填写的 Token |

使用仓库项目模板：

```powershell
voice-dataset init `
  $workspace `
  --config '.\examples\project\pipeline.toml'
```

使用内置默认配置：

```powershell
voice-dataset init $workspace
```

明确重置项目配置：

```powershell
voice-dataset init $workspace --overwrite-config
```

`init` 创建工作区骨架以及相互独立的 `config/`、`secrets/`。如果只存在旧版
`<workspace>\pipeline.toml`，再次运行 `init` 会校验并复制到
`<workspace>\config\pipeline.toml`，不会删除旧文件。

### `ingest`：递归导入音视频

完整语法：

```powershell
voice-dataset ingest `
  <workspace> `
  <input> [<input> ...] `
  [--config <project-config.toml>] `
  [--mode auto|audio|video]
```

参数：

| 参数 | 含义 |
| --- | --- |
| `workspace` | 目标工作区 |
| `inputs` | 一个或多个文件、目录；目录会递归扫描 |
| `--config` | 覆盖默认项目配置路径 |
| `--mode auto` | 接收配置支持的全部音频和视频，默认值 |
| `--mode audio` | 只接收音频文件 |
| `--mode video` | 只接收视频文件 |

导入一个目录：

```powershell
voice-dataset ingest $workspace $inputPath
```

导入多个文件和目录：

```powershell
voice-dataset ingest `
  $workspace `
  'D:\media\chapter-1' `
  'D:\media\chapter-2\dialogue.wav' `
  'D:\media\chapter-3\scene.mp4'
```

只导入视频：

```powershell
voice-dataset ingest $workspace $inputPath --mode video
```

主要产物：

```text
<workspace>\normalized\...
<workspace>\manifests\sources.jsonl
```

同内容文件按 SHA-256 去重；视频会提取规范化音轨。该命令不移动或删除原文件。

### `split`：只拆分语音边界

完整语法：

```powershell
voice-dataset split `
  <workspace> `
  [--config <project-config.toml>] `
  [--backend energy|gemini] `
  [--modality auto|audio|video] `
  [--secrets <credentials.toml>] `
  [--limit <数量>] `
  [--replace]
```

参数：

| 参数 | 含义 |
| --- | --- |
| `--backend energy` | 本地 Python 能量检测，不调用网络 |
| `--backend gemini` | 上传媒体并调用 Gemini，只接收时间边界 |
| `--modality auto` | 按来源媒体自动选择，默认取配置值 |
| `--modality audio` | 分析规范化音轨 |
| `--modality video` | Gemini 读取原始视频；energy 仍只分析提取音轨 |
| `--secrets` | 指定另一个敏感配置，仅 Gemini 后端使用 |
| `--limit N` | 本次最多处理 N 个尚未处理的来源，适合小批量试跑 |
| `--replace` | 重新拆分已有来源，并重置已失效片段的标签与复核决定 |

纯本地音频拆分：

```powershell
voice-dataset split `
  $workspace `
  --backend energy `
  --modality audio
```

纯本地处理视频的音轨：

```powershell
voice-dataset split `
  $workspace `
  --backend energy `
  --modality video
```

Gemini 音频语义拆分：

```powershell
voice-dataset split `
  $workspace `
  --backend gemini `
  --modality audio
```

Gemini 视频多模态拆分：

```powershell
voice-dataset split `
  $workspace `
  --backend gemini `
  --modality video
```

先试跑 5 个来源：

```powershell
voice-dataset split `
  $workspace `
  --backend gemini `
  --modality video `
  --limit 5
```

确认新参数后重新拆分：

```powershell
voice-dataset split `
  $workspace `
  --backend energy `
  --modality audio `
  --replace
```

主要产物：

```text
<workspace>\manifests\segments.jsonl
<workspace>\manifests\clips.jsonl
<workspace>\clips\*.wav
```

没有 `--replace` 时，已有边界的来源会跳过。替换拆分如果没有产生任何有效片段，命令会
报错并保留旧清单。

### `label`：Gemini 转写、情绪和聚类

完整语法：

```powershell
voice-dataset label `
  <workspace> `
  [--config <project-config.toml>] `
  [--provider gemini] `
  [--language <语言提示>] `
  [--secrets <credentials.toml>] `
  [--limit <数量>] `
  [--force]
```

参数：

| 参数 | 含义 |
| --- | --- |
| `--provider gemini` | 当前唯一标注提供方 |
| `--language auto` | 自动判断语言，默认值 |
| `--language zh` | 提示模型按中文转写；也可传入 `en`、`ja` 等提示 |
| `--secrets` | 指定另一个敏感配置文件 |
| `--limit N` | 本次最多标注 N 个待处理片段 |
| `--force` | 重新调用模型并覆盖已有候选标注，会再次消耗 API 配额 |

标注全部未处理片段：

```powershell
voice-dataset label $workspace --provider gemini --language zh
```

SenseVoice 自动写入的草稿会被视为待处理候选：普通 `label` 会用 Gemini
结果升级这些 provisional seed，但不会覆盖已有的 Gemini/人工标签。
`--force` 才会重新调用并覆盖所有候选。

先标注 20 条检查效果：

```powershell
voice-dataset label `
  $workspace `
  --provider gemini `
  --language zh `
  --limit 20
```

使用指定敏感配置重新标注：

```powershell
voice-dataset label `
  $workspace `
  --provider gemini `
  --secrets 'D:\private-config\character-a.credentials.toml' `
  --force
```

产物：

```text
<workspace>\manifests\labels.jsonl
```

模型只能从 `review.emotions` 和 `review.clusters` 共享词表中选择类别。模型结果仍是候选，
必须经过人工复核。

### `review`：启动断点续作 TUI

完整语法：

```powershell
voice-dataset review `
  <workspace> `
  [--config <project-config.toml>]
```

启动或继续复核：

```powershell
voice-dataset review $workspace
```

使用另一套非敏感项目配置：

```powershell
voice-dataset review `
  $workspace `
  --config 'D:\configs\character-a.pipeline.toml'
```

按键：

| 按键 | 操作 |
| --- | --- |
| `0` | 播放当前音频 |
| `M` | 与下一片段合并；随后重跑质量与 ASR |
| `1`–`N` | 选择对应情绪并确认 |
| `X` | 排除 |
| `E` | 编辑台词 |
| `K` | 编辑聚类 |
| `S` | 跳过并放到队尾 |
| `R` | 刷新 |
| `B` | 撤销上一次操作 |
| `Q` | 保存并退出 |

状态实时写入：

```text
<workspace>\state\review.json
```

`Ctrl+C` 或终端异常退出后，执行同一命令即可继续。

### `status`：只读检查阶段计数

完整语法：

```powershell
voice-dataset status <workspace>
```

示例：

```powershell
voice-dataset status $workspace
```

输出字段包括：

```text
sources
segments
clips
labels
reviewed
draft_decisions
review_cursor
```

该命令不接受 `--config`，也不修改工作区。

### `export`：物化训练集

完整语法：

```powershell
voice-dataset export `
  <workspace> `
  [--config <project-config.toml>] `
  [--output <directory>] `
  [--speaker <speaker-name>] `
  [--language <zh|ja|en|ko|yue>] `
  [--allow-unreviewed]
```

参数：

| 参数 | 含义 |
| --- | --- |
| `--output` | 导出父目录；省略时写入 `<workspace>\training\exports` |
| `--speaker` | 覆盖导出清单中的说话人名称 |
| `--language` | 覆盖 GPT-SoVITS 清单语言 |
| `--allow-unreviewed` | 允许导出未确认条目；默认禁止 |

标准导出：

```powershell
voice-dataset export `
  $workspace `
  --output $exportPath `
  --speaker 'character_a' `
  --language zh
```

使用默认输出目录：

```powershell
voice-dataset export $workspace
```

调试性导出未复核条目：

```powershell
# 不建议用于正式训练。训练器默认仍会拒绝 reviewed=false 的记录。
voice-dataset export $workspace --allow-unreviewed
```

每次导出使用内容指纹生成不可变目录：

```text
dataset-<fingerprint>\
  metadata.json
  manifest.jsonl
  reviewed\<emotion>\<cluster>\<clip-id>.wav
  reviewed\<emotion>\<cluster>\<clip-id>.txt
  gpt-sovits\dataset.list
  rvc\dataset\*.wav
```

review、切片或影响数据集的配置变化后，旧导出会被 `train` 判定为过期，必须重新执行
`export`。

### `train`：GPT-SoVITS 与 RVC 训练编排

完整语法：

```powershell
voice-dataset train `
  <workspace> `
  gpt-sovits|rvc `
  [--config <project-config.toml>] `
  [--dataset <dataset-directory>] `
  [--execute] `
  [--register-as <model-name>] `
  [--activate]
```

参数：

| 参数 | 含义 |
| --- | --- |
| `gpt-sovits` | 生成或执行 GPT-SoVITS 两阶段训练计划 |
| `rvc` | 生成或执行 RVC 预处理、F0、HuBERT、训练和索引计划 |
| `--dataset` | 指定某个 `dataset-<fingerprint>`；省略时选择最新有效导出 |
| `--execute` | 真正执行外部预处理和训练 |
| `--register-as` | GPT-SoVITS 成功后把产物登记为角色模型 |
| `--activate` | 与 `--register-as` 同用，将该模型设为默认 |

只生成 GPT-SoVITS 计划：

```powershell
voice-dataset train $workspace gpt-sovits
```

只生成 RVC 计划：

```powershell
voice-dataset train $workspace rvc
```

指定导出数据集：

```powershell
$dataset = 'D:\voice-datasets\character-a\dataset-0123456789ab'
voice-dataset train $workspace gpt-sovits --dataset $dataset
```

显式执行：

```powershell
# 还必须在 config/pipeline.toml 中同时启用：
# [training] enabled = true
# [training.gpt_sovits] enabled = true
voice-dataset train $workspace gpt-sovits --execute

# RVC 对应需要 [training.rvc] enabled = true
voice-dataset train $workspace rvc --execute
```

生成计划后即可检查：

```text
<workspace>\training\gpt_sovits\<experiment_name>\training-plan.json
<workspace>\training\rvc\<experiment_name>\training-plan.json
```

执行 `--execute` 后，每个外部命令分别写入日志：

```text
<workspace>\training\gpt_sovits\<experiment_name>\logs\<command>.log
<workspace>\training\rvc\<experiment_name>\logs\<command>.log
```

训练完成并通过产物校验后还会生成 `artifacts.json` 和 `training-result.json`。同名
experiment 的数据、配置、底模或外部仓库代码指纹发生变化时，工具会拒绝覆盖；请使用
新的 `experiment_name`，不要删除门禁后强行复用旧特征。

查看某次训练的日志：

```powershell
$run = Join-Path $workspace 'training\gpt_sovits\character_a_v2proplus'
Get-ChildItem -LiteralPath (Join-Path $run 'logs') -Filter '*.log'
Get-Content -LiteralPath (Join-Path $run 'logs\train-gpt.log') -Tail 100
```

### 一套可复制的纯本地拆分流程

以下流程不会调用 Gemini，只生成可供后续人工处理的 WAV 切片：

```powershell
$repo = 'D:\voice-dataset-pipeline'
$workspace = 'D:\voice-workspaces\character-a'
$inputPath = 'D:\media\character-a'

Set-Location $repo
voice-dataset init $workspace --config '.\examples\project\pipeline.toml'
voice-dataset ingest $workspace $inputPath --mode auto
voice-dataset split $workspace --backend energy --modality audio
voice-dataset status $workspace
```

如果继续执行 `review`，没有模型候选转写的条目需要使用 `E` 手工填写台词，再用数字键
确认情绪。

### 一套可复制的 Gemini 完整流程

```powershell
$repo = 'D:\voice-dataset-pipeline'
$workspace = 'D:\voice-workspaces\character-a'
$inputPath = 'D:\media\character-a'
$exportPath = 'D:\voice-datasets\character-a'

Set-Location $repo
python .\scripts\generate_configs.py $workspace

# 编辑以下两个互相独立的文件：
# $workspace\config\pipeline.toml
# $workspace\secrets\credentials.toml

voice-dataset init $workspace
voice-dataset ingest $workspace $inputPath --mode auto
voice-dataset split $workspace --backend gemini --modality video
voice-dataset label $workspace --provider gemini --language zh
voice-dataset review $workspace
voice-dataset status $workspace
voice-dataset export $workspace --output $exportPath --speaker 'character_a' --language zh
voice-dataset train $workspace gpt-sovits
voice-dataset train $workspace rvc
```

最后两条命令只生成计划。确认计划、日志路径、外部工程和基础模型正确后，再分别附加
`--execute`。
