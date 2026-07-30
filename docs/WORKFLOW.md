# 工作流

本工具把“发现媒体、拆分、模型标注、人工复核、导出、训练”分为独立阶段。每一步都读取前一步的持久化清单，因此可以检查中间结果、重复运行安全步骤，并在训练前停下来确认。

```text
本地文件或目录
    -> init / ingest
    -> split (energy 或 Gemini；只生成边界)
    -> label (Gemini；台词、情绪、聚类)
    -> review (人工确认，可断点续作)
    -> export (物化训练集)
    -> train gpt-sovits
    -> train rvc / RVC 后处理
```

命令示例表达的是稳定的工作流概念。不同版本的参数细节以 `voice-dataset <子命令> --help` 为准。

## 0. 预检

创建并激活 Python 环境后，先检查 CLI 和 FFmpeg：

```powershell
voice-dataset --help
voice-dataset init --help
voice-dataset ingest --help
voice-dataset split --help
voice-dataset label --help
voice-dataset review --help
voice-dataset export --help
voice-dataset train --help
ffmpeg -version
```

如果使用 Gemini，确认当前进程中存在密钥，但不要输出它：

```powershell
if ([string]::IsNullOrWhiteSpace($env:GEMINI_API_KEY)) {
    Write-Output '[WARN] GEMINI_API_KEY is not set'
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

初始化是本地、低成本操作，不会扫描媒体、调用 Gemini 或启动训练。工作区用于保存配置、来源清单、拆分结果、不可变音频切片、标签、复核状态与导出记录。

配置可从仓库示例复制后修改：

```powershell
New-Item -ItemType Directory -Force -Path $workspace | Out-Null
Copy-Item '.\examples\pipeline.example.toml' (Join-Path $workspace 'pipeline.toml')
```

不要在配置中写入 API Key。

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

## 4. Gemini 标注与聚类

设置当前进程的 Key 后，显式运行：

```powershell
voice-dataset label $workspace --provider gemini
```

该阶段对每个切片生成候选信息：

- 台词转写。
- 情绪类别。
- 从 `review.clusters` 共享词表中选择的聚类名称。
- 置信度或简短判断依据（如果当前适配器提供）。

情绪集合来自角色配置，TUI 的数字顺序与配置完全一致。模型输出只是草稿；人工复核前不要直接训练。

API Key 只允许从配置指定的环境变量读取，默认变量名是 `GEMINI_API_KEY`。项目配置、状态文件和日志不得持久化 Key。

如果调用失败：

1. 检查 `GEMINI_API_KEY` 是否只在当前进程中有效。
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
| `1`–`N` | 选择配置中对应的情绪并确认条目 |
| `X` | 排除当前条目 |
| `E` | 编辑台词 |
| `K` | 编辑聚类名 |
| `S` | 暂时跳过并放到队尾 |
| `R` | 刷新当前条目 |
| `B` | 撤销上一次操作 |
| `Q` | 保存并退出 |

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

具体 epoch、batch size 和版本参数在角色 `pipeline.toml` 的 `training.gpt_sovits` 中配置；完整字段见仓库示例，实际底模路径、命令和指纹以生成的 `training-plan.json` 为准。

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

成功标准不是只生成 `.index`。必须存在对应的可推理 `.pth` 权重，才能把 GPT-SoVITS 结果送入 RVC 后处理。本项目负责训练和校验该后处理模型，不包含 RVC 推理；实际转换参数与 checkpoint 对比请在 RVC 上游工具中完成。

## 成本与副作用边界

| 操作 | 网络/API | 是否启动训练 | 主要写入 |
| --- | --- | --- | --- |
| `init` | 否 | 否 | 工作区骨架 |
| `ingest` | 否 | 否 | 来源清单、规范化媒体 |
| `split --backend energy` | 否 | 否 | 边界、切片 |
| `split --backend gemini` | 是 | 否 | 边界、切片 |
| `label` | 是 | 否 | 候选标签 |
| `review` | 否 | 否 | 人工复核状态 |
| `export` | 否 | 否 | 训练集 |
| `train ...` | 否 | 否 | 目标 Python 依赖预检、暂存数据、生成配置、训练计划；RVC 外部实验目录 |
| `train ... --execute` | 取决于上游 | 是 | 外部实验与模型产物 |

任何高成本行为都必须来自清楚、独立的用户命令。检查状态和打开 TUI 不会隐式触发 Gemini 或训练；重新生成训练计划会重复预检，但不会启动模型训练。
