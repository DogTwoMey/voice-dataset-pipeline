# voice-dataset-pipeline

一个独立、无前端的 Python 工具，用于把本地音频或视频整理为可人工复核的语音训练集，并编排 GPT-SoVITS 与 RVC 的训练流程。

项目由 DogTwoMey 维护：[DogTwoMey/voice-dataset-pipeline](https://github.com/DogTwoMey/voice-dataset-pipeline)。仓库只保存代码、示例配置和文档；原始媒体、切片、人工复核状态、导出数据集、API Key 与模型权重都不应提交到 Git。

## 能做什么

- 接收单个文件、本地目录或多个输入路径；目录会递归扫描受支持的音视频。
- 使用本地能量检测，或显式调用 Gemini 多模态模型，仅生成语音片段边界。
- 使用 Gemini 为切片生成台词、情绪和聚类标签；聚类从角色配置的共享词表中选择，避免每段生成一个同义孤立类别。
- 在终端 TUI 中逐条播放、修正情绪、编辑文本、排除素材，并在异常退出后继续。
- 导出适合资源管理器检查的训练集：音频与同名文本文件直观对应。
- 为 GPT-SoVITS 和 RVC 生成训练计划；只有传入 `--execute` 才会启动外部训练。

拆分与标注是两个独立阶段。拆分阶段只决定音频边界，不会把模型改写的文本当作训练台词；最终标签仍以人工复核结果为准。

## 环境要求

- Python 3.11 或更高版本。
- [FFmpeg](https://ffmpeg.org/) 可从 `PATH` 直接调用，或在 TOML 配置中指定可执行文件。
- 使用 Gemini 时安装 `gemini` 可选依赖，并仅通过配置指定的环境变量
  （默认 `GEMINI_API_KEY`）提供密钥。
- 训练 GPT-SoVITS 时，需要用户自行准备可运行的 GPT-SoVITS 工程、Python 环境和基础模型。
- 训练或使用 RVC 后处理时，需要用户自行准备可运行的 RVC 工程、Python 环境和基础模型。

本项目不会安装 GPT-SoVITS、RVC、CUDA、基础模型或 FFmpeg，也不会自动下载大型模型。

## 安装

Windows PowerShell：

```powershell
Set-Location 'D:\voice-dataset-pipeline'
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e '.[all]'
voice-dataset --help
```

只使用本地能量拆分、不调用 Gemini 或训练适配器时，可以安装基础依赖：

```powershell
python -m pip install -e .
```

## 快速开始

以下命令展示完整流程。CLI 的最终参数以当前版本的 `--help` 为准；第一次使用时建议先查看每个子命令的帮助。

```powershell
$workspace = 'D:\voice-workspaces\character-a'
$inputPath = 'D:\media\character-a'

voice-dataset init $workspace --config '.\examples\pipeline.example.toml'
# 在继续训练前编辑 $workspace\pipeline.toml 中的外部工程路径、参数和 enabled 开关。
voice-dataset ingest $workspace $inputPath
voice-dataset split $workspace --backend energy --modality audio
voice-dataset label $workspace --provider gemini
voice-dataset review $workspace
voice-dataset export $workspace --output 'D:\voice-datasets\character-a'
voice-dataset train $workspace gpt-sovits
voice-dataset train $workspace rvc
```

`ingest` 会递归处理目录，并跳过不受支持的文件。工作区保存来源清单、中间状态、不可变切片和复核记录；请为不同角色使用不同工作区。

### 使用 Gemini

Gemini 调用不会在 `init`、`ingest` 或本地 `energy` 拆分时隐式发生。只有用户明确选择 Gemini 拆分或执行 `label` 时才会产生网络请求和配额消耗。

在当前 PowerShell 进程中安全地输入密钥：

```powershell
$secureKey = Read-Host 'Gemini API key' -AsSecureString
$credential = New-Object System.Net.NetworkCredential('', $secureKey)
$env:GEMINI_API_KEY = $credential.Password
```

不要把密钥写入 TOML、`.env`、命令行参数或 Git。

Gemini 可以按音频或视频内容建议边界；模型拆分仍然只产出时间区间：

```powershell
voice-dataset split $workspace --backend gemini --modality audio
voice-dataset split $workspace --backend gemini --modality video
voice-dataset split $workspace --backend gemini --modality auto
voice-dataset label $workspace --provider gemini
```

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
- `1`–`N`：直接指定配置中对应的情绪类别。
- `X`：排除当前条目。
- `E`：编辑台词。
- `K`：编辑聚类名。
- `S`：暂时跳过并放到队尾。
- `R`：刷新当前条目。
- `B`：撤销上一次操作。
- `Q`：保存并退出。

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

RVC 训练完成必须同时存在可推理的 `.pth` 权重；只有 `.index` 文件不代表模型已经可用于后处理。本项目训练 RVC 后处理模型，但不包含 RVC 推理命令。

## 配置

复制 [examples/pipeline.example.toml](examples/pipeline.example.toml)，再为角色调整情绪类别、拆分阈值和外部训练工程路径。工作区由每条命令的位置参数指定；配置中不提供 Gemini Key。

```powershell
Copy-Item '.\examples\pipeline.example.toml' 'D:\voice-workspaces\character-a\pipeline.toml'
voice-dataset --help
```

常用训练字段：

| 配置段 | 关键字段 |
| --- | --- |
| `training` | `enabled`：只有启用后，`--execute` 才获准执行 |
| `training.gpt_sovits` | `repository`、`python`、`experiment_name`、`model_version`、`gpu`、两阶段的 batch/epoch/save 参数 |
| `training.rvc` | `repository`、`python`、`experiment_name`、`version`、`sample_rate`、`gpu`、`workers`、`batch_size`、`epochs`、`save_every` |

完整字段和默认值见示例 TOML；生成的 `training-plan.json` 会记录实际命令与指纹。

完整阶段说明见 [docs/WORKFLOW.md](docs/WORKFLOW.md)。

## 安全与可复现性

- 原始输入只读；人工复核不会直接移动或改写不可变切片。
- 导出阶段才根据复核状态物化训练目录，避免资源管理器或播放器锁定文件时破坏状态。
- API Key 只从配置指定的环境变量读取，默认变量名为 `GEMINI_API_KEY`。
- Gemini 调用和实际训练都必须由用户显式启动。
- 默认训练命令只做预检和计划准备；`--execute` 是启动预处理及训练的明确授权。
- 请在训练前保存配置、清单和工作区指纹，以免把旧中间产物误用于新数据。

## 开发说明

本项目按用户本地 Python 环境运行，不配置 CI。提交前可在本地执行：

```powershell
python -m pip install -e '.[dev,train]'
python -m pytest
python -m ruff check .
```
