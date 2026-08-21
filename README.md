# voice-dataset-pipeline

一个独立、无前端的 Python 工具，用于把本地音频或视频整理为可人工复核的语音训练集，编排 GPT-SoVITS/RVC 训练，并以“文本 + 可选参考音频”直接生成角色语音。

项目由 DogTwoMey 维护：[DogTwoMey/voice-dataset-pipeline](https://github.com/DogTwoMey/voice-dataset-pipeline)。仓库只保存代码、示例配置和文档；原始媒体、切片、人工复核状态、导出数据集、API Key 与模型权重都不应提交到 Git。

## 能做什么

- 接收单个文件、本地目录或多个输入路径；目录会递归扫描受支持的音视频。
- 使用本地能量检测，或显式调用 Gemini 多模态模型，仅生成语音片段边界。
- 使用 Gemini 为切片生成台词、情绪和聚类标签；聚类从角色配置的共享词表中选择，避免每段生成一个同义孤立类别。
- 在终端 TUI 中逐条播放、修正情绪、编辑文本、排除素材，并在异常退出后继续。
- 导出适合资源管理器检查的训练集：音频与同名文本文件直观对应。
- 为 GPT-SoVITS 和 RVC 生成训练计划；只有传入 `--execute` 才会启动外部训练。
- 以 sidecar 字幕、内嵌字幕、视觉模型、静音切分的顺序回退，记录实际边界来源。
- 使用本地 SenseVoice 生成转写与语音情绪草稿，并以时长、RMS、削顶和静音比做质量门禁。
- 把已训练 GPT/SoVITS 权重登记为角色模型；按情绪自动挑选经过复核的参考音频。
- 对目标文本生成统一情绪计划，通过模型注册时记录的 GPT-SoVITS 专属 Python 启动隔离 worker，无需启动 WebUI/API 服务。
- 可选调用独立 RVC 环境做后处理，并始终保留未转换的 SoVITS 原始 WAV。

拆分与标注是两个独立阶段。拆分阶段只决定音频边界，不会把模型改写的文本当作训练台词；最终标签仍以人工复核结果为准。

## 环境要求

- 编排器使用 Python 3.11 或更高版本；推荐 Python 3.12。
- [FFmpeg](https://ffmpeg.org/) 可从 `PATH` 直接调用，或在 TOML 配置中指定可执行文件。
- 使用 Gemini 时安装 `gemini` 可选依赖；密钥只能来自独立的
  `secrets/credentials.toml` 或配置指定的环境变量（默认 `GEMINI_API_KEY`）。
- 训练 GPT-SoVITS 时，需要用户自行准备可运行的 GPT-SoVITS 工程、Python 环境和基础模型。
- 训练或使用 RVC 后处理时，需要用户自行准备可运行的 RVC 工程、Python 环境和基础模型。

GPT-SoVITS、RVC、IndexTTS/Qwen3-TTS 等上游必须使用各自钉版仓库和独立 Python 环境。不要把所有模型塞入一个 Python 3.13 venv；不同上游的 PyTorch、音频与推理依赖并不兼容。本项目不会安装 CUDA、基础模型或 FFmpeg；但首次运行 SenseVoice 时，FunASR/ModelScope 可能自动下载尚未缓存的 ASR/VAD 模型。

模型注册表同时保存 GPT-SoVITS 仓库、专属 Python、GPT/SoVITS 权重和参考清单，并封存已知推理底模与 provider Python 源码摘要。`synthesize` 只在该解释器的 worker 子进程中导入上游推理依赖，主管线环境不会直接加载另一套 PyTorch。训练时使用 `--register-as` 会自动登记 `[training.gpt_sovits].python`；手工登记已有权重时必须传入 `--python`。

## 安装

Windows PowerShell：

```powershell
Set-Location 'D:\voice-dataset-pipeline'
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e '.[gemini,train,asr]'
voice-dataset --help
```

`asr` 只安装 FunASR/ModelScope，故意不替你选择 PyTorch 的 CPU/CUDA
轮子。要在这个编排器环境中运行 SenseVoice，先按
[PyTorch 官方安装器](https://pytorch.org/get-started/locally/) 安装与显卡驱动匹配的
`torch`/`torchaudio`，并用以下命令确认：

```powershell
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

如果 ASR 使用另一个已验证的模型环境，先用 `preprocess --skip-asr` 完成切分和
质量门禁，再在该环境中运行 `python -m voice_dataset_pipeline transcribe`。

ASR 配置同时固定模型/VAD revision 以及 FunASR、ModelScope 版本；运行环境不匹配时
会在加载模型前失败。`master` 只是 provider 的符号 revision，并非模型内容哈希；追求
严格复现时应把它替换为 ModelScope 支持的不可变 revision。

只使用本地能量拆分、不调用 Gemini 或训练适配器时，可以安装基础依赖：

```powershell
python -m pip install -e .
```

## 快速开始

以下 PowerShell 命令展示从素材到可合成角色模型的完整流程：

```powershell
$repo = 'D:\voice-dataset-pipeline'
$workspace = 'D:\voice-workspaces\character-a'
$inputPath = 'D:\media\character-a'
$exportPath = 'D:\voice-datasets\character-a'

Set-Location $repo
voice-dataset init $workspace --config '.\examples\project\pipeline.toml'
# 非敏感项目配置：$workspace\config\pipeline.toml
# 敏感本地配置：$workspace\secrets\credentials.toml（整个目录由 .gitignore 保护）

voice-dataset preprocess $workspace $inputPath --asr
voice-dataset label $workspace --provider gemini
voice-dataset review $workspace
voice-dataset status $workspace
voice-dataset export $workspace --output $exportPath

# 默认只生成计划，不启动训练
voice-dataset train $workspace gpt-sovits
voice-dataset train $workspace rvc

# 检查计划无误后才执行；成功后自动登记并激活模型：
# voice-dataset train $workspace gpt-sovits --execute --register-as character_a --activate
# voice-dataset train $workspace rvc --execute

# 文本 + 自动情绪参考 -> WAV
# voice-dataset synthesize $workspace --text '你好，很高兴见到你。' --output '.\out.wav'
```

`ingest` 会递归处理目录，并跳过不受支持的文件。工作区保存来源清单、中间状态、不可变切片和复核记录；请为不同角色使用不同工作区。

### 使用 Gemini

Gemini 调用不会在 `init`、`ingest` 或本地 `energy` 拆分时发生。执行 Gemini 拆分、`label`，或执行一个已在项目配置中选择 `splitting.backend=gemini` 的 `preprocess`，都会上传相应媒体并产生网络请求和配额消耗；配置选择也视为明确授权。

`init` 会生成独立的敏感配置，编辑其中的空值即可：

```toml
# <workspace>/secrets/credentials.toml
[environment]
GEMINI_API_KEY = ""
```

只在本地把空值替换为实际密钥。该目录包含自己的 `.gitignore`，默认忽略除规则文件外
的全部内容。也可以不写敏感
文件，改为在当前 PowerShell 进程中输入密钥：

```powershell
$secureKey = Read-Host 'Gemini API key' -AsSecureString
$credential = New-Object System.Net.NetworkCredential('', $secureKey)
$env:GEMINI_API_KEY = $credential.Password
```

不要把密钥写入 `config/pipeline.toml`、命令行参数或 Git。显式
`--secrets` 可选择另一个敏感配置文件；没有敏感文件或其中的值为空时，程序回退到
环境变量。

Gemini 可以按音频或视频内容建议边界；模型拆分仍然只产出时间区间：

```powershell
voice-dataset split $workspace --backend gemini --modality audio
voice-dataset split $workspace --backend gemini --modality video
voice-dataset split $workspace --backend gemini --modality auto
voice-dataset label $workspace --provider gemini
```

视频超过 `[gemini.chunking].threshold_seconds` 时，工具先用本地 FFmpeg
`silencedetect` 在静音附近规划连续窗口，再转成低码率预览块逐块调用 Gemini；短视频
和音频保持单次调用。窗口、转码块和每块已验证的边界分别缓存在
`<workspace>\state\gemini_chunks`，因此中途失败后不会重复已经成功的远端调用。
缓存指纹绑定源内容、块参数、Gemini 模型和片段时长约束。设置
`reuse_chunks=false` 可强制重新调用；设置 `keep_chunks=false` 会在调用后删除预览 MP4，
但保留时间窗口清单和边界 JSON 供审计、断点续跑。

`energy` 是纯本地音频算法，不理解画面语义；对视频执行
`--backend energy --modality video` 时，它只分析导入阶段提取的规范化音轨。
`gemini` 会上传所需媒体并调用远端模型，适合需要多模态上下文的素材。
可用组合和覆盖参数请以 `voice-dataset split --help` 为准。

### 人工复核

```powershell
voice-dataset review $workspace
```

TUI 每次显示一个切片及其台词。主要按键：

- `0`：用系统播放器播放当前音频。
- `M`：把当前片段与下一片段无损重建为一个新片段；旧 WAV 保留。
- `1`–`N`：直接指定配置中对应的情绪类别。
- `X`：排除当前条目。
- `E`：编辑台词。
- `K`：编辑聚类名。
- `S`：暂时跳过并放到队尾。
- `R`：刷新当前条目。
- `B`：撤销上一次操作。
- `Q`：保存并退出。

界面同时显示前一条和后一条台词，便于识别跨句粗切与过细切分。执行 `M` 后，新片段需要再次运行 `quality` 和 `transcribe`；结构性合并会清空旧撤销栈，避免撤销记录指向已失效 clip。

每次操作都会持久化。正常退出、`Ctrl+C` 或意外中断后，重新执行同一命令即可从断点继续。

### 导出与训练

导出只使用人工复核后的有效条目：

```powershell
voice-dataset export $workspace --output 'D:\voice-datasets\character-a'
```

训练命令默认执行预检并生成计划，不启动预处理或训练。预检会调用目标 Python
检查依赖，并创建计划、配置和暂存目录；RVC 预检还会准备其工程所需的实验目录：

```powershell
voice-dataset train $workspace gpt-sovits
voice-dataset train $workspace rvc
```

确认工作区、外部仓库、Python 解释器、基础模型和输出目录都正确后，才显式执行：

```powershell
voice-dataset train $workspace gpt-sovits --execute
voice-dataset train $workspace rvc --execute
```

RVC 训练完成必须同时存在可推理的 `.pth` 权重；只有 `.index` 文件不代表模型已经可用于后处理。将 RVC 权重与索引登记到角色模型后，可通过 `synthesize --postprocess rvc` 转换；工具同时保留 `*.sovits.wav`，方便 A/B 检查吐字是否劣化。

SoX/SoX_ng 是独立的最终母带阶段，顺序固定为
`GPT-SoVITS -> 可选 RVC -> 可选 SoX`。内置 `speech`（默认）、`singing`、
`audiobook`、`asmr`、`stage` 五种场景；场景不仅选择母带方案，也会调整真正传给
GPT-SoVITS 的语速/采样参数并影响参考片段偏好：

```powershell
# 单一常规语音场景；auto 是否调用 SoX 由配置决定
voice-dataset synthesize $workspace --text '你好。' --scene speech `
  --mastering auto --output 'D:\output\voice.wav'

# 同一文本、seed 和模型一次生成五种场景
voice-dataset synthesize $workspace --text '你好。' --scene all `
  --mastering sox --seed 2333 --output 'D:\output\voice.wav'
```

第二条命令生成 `voice.speech.wav`、`voice.singing.wav`、
`voice.audiobook.wav`、`voice.asmr.wav`、`voice.stage.wav`。SoX 前的文件以
`*.sovits.wav` 保存；若同时使用 RVC，还会保留 `*.rvc.wav`。`singing` 只表示歌唱素材
偏好和保守母带，不会从普通文本自动生成旋律；真正歌唱需要演唱参考或独立 SVS provider。

短台词合成默认使用 `[inference].text_split_method="cut0"`，让 GPT-SoVITS 在一次推理中
自行学习标点停顿。不要对短句使用 `cut5`：它会在每个标点处分段，并在每段尾插入
`fragment_interval` 静音，听起来像突然截断。长文本可显式改为 `cut2` 后再 A/B；人名中的
`·` 会被上游中文规范器当成逗号，要求连读时应移除。

固定 checkpoint 后，可用相同台词、参考音频和 seed 做 A/B，并将胜出的 `top_k`、
`top_p`、`temperature`、`pace` 写入角色配置的
`[inference.emotion_overrides.<emotion>]`。覆盖只作用于该角色的对应情绪，非法值会在启动
上游推理前被拒绝；不要把某个角色的结果当成全局默认值。

## 配置

配置按职责分离：

| 类型 | 默认位置 | Git 策略 |
| --- | --- | --- |
| 非敏感项目配置 | `<workspace>/config/pipeline.toml` | 可以提交和评审 |
| API Key、Token 等敏感配置 | `<workspace>/secrets/credentials.toml` | 整个 `secrets` 目录默认忽略 |

可以通过 `init` 或独立 Python 脚本生成两套默认文件；脚本默认保留已有文件，只有显式
传入覆盖参数才会重写：

```powershell
python .\scripts\generate_configs.py 'D:\voice-workspaces\character-a'
python .\scripts\generate_configs.py 'D:\voice-workspaces\character-a' --help
```

可提交模板见 [examples/project/pipeline.toml](examples/project/pipeline.toml)，空白敏感
模板见
[examples/secrets/credentials.toml.example](examples/secrets/credentials.toml.example)。
旧版 `<workspace>/pipeline.toml` 仍可读取；再次执行 `init` 会将其复制到新的
`config/` 目录。

常用训练字段：

| 配置段 | 关键字段 |
| --- | --- |
| `training` | `enabled`：只有启用后，`--execute` 才获准执行 |
| `training.gpt_sovits` | `repository`、`python`、`experiment_name`、`model_version`、`gpu`、两阶段的 batch/epoch/save 参数 |
| `training.rvc` | `repository`、`python`、`experiment_name`、`version`、`sample_rate`、`gpu`、`workers`、`batch_size`、`epochs`、`save_every` |
| `scenes` | `default`：缺省合成场景，默认 `speech` |
| `postprocess.sox` | `enabled`、SoX/SoX_ng `binary`、`output_bits` |

完整字段和默认值见示例 TOML；生成的 `training-plan.json` 会记录实际命令与指纹。

完整阶段说明见 [docs/WORKFLOW.md](docs/WORKFLOW.md)。

### 命令速查

| 目的 | 命令 |
| --- | --- |
| 生成两类配置 | `python scripts/generate_configs.py <workspace>` |
| 初始化工作区 | `voice-dataset init <workspace>` |
| 递归导入 | `voice-dataset ingest <workspace> <file-or-directory> [...]` |
| 高层预处理 | `voice-dataset preprocess <workspace> <input...> [--asr]` |
| 本地拆分 | `voice-dataset split <workspace> --backend energy` |
| Gemini 拆分 | `voice-dataset split <workspace> --backend gemini --modality video` |
| Gemini 标注 | `voice-dataset label <workspace> --provider gemini` |
| 人工复核 | `voice-dataset review <workspace>` |
| 查看进度 | `voice-dataset status <workspace>` |
| 本地转写 | `voice-dataset transcribe <workspace>` |
| 音质门禁 | `voice-dataset quality <workspace>` |
| 导出训练集 | `voice-dataset export <workspace> [--output <directory>]` |
| 生成训练计划 | `voice-dataset train <workspace> {gpt-sovits,rvc}` |
| 执行训练 | `voice-dataset train <workspace> {gpt-sovits,rvc} --execute` |
| 模型注册 | `voice-dataset model register <workspace> ...` |
| 情绪计划 | `voice-dataset emotion <workspace> --text <台词>` |
| 合成语音 | `voice-dataset synthesize <workspace> --text <台词> --output <wav>` |

所有参数、覆盖行为、输出位置和可复制示例见
[详细命令手册](docs/WORKFLOW.md#完整命令参考)。

## 安全与可复现性

- 原始输入只读；人工复核不会直接移动或改写不可变切片。
- 导出阶段才根据复核状态物化训练目录，避免资源管理器或播放器锁定文件时破坏状态。
- API Key 只从独立的敏感配置或指定环境变量读取；项目配置只保存变量名。
- Gemini 调用和实际训练都必须由用户显式启动；`preprocess` 会遵循已保存的 Gemini 后端配置。
- 默认训练命令只做预检和计划准备；`--execute` 是启动预处理及训练的明确授权。
- 请在训练前保存配置、清单和工作区指纹，以免把旧中间产物误用于新数据。

## 开发说明

本项目按用户本地 Python 环境运行，不配置 CI。提交前可在本地执行：

```powershell
python -m pip install -e '.[dev,train]'
python -m pytest
python -m ruff check .
```
