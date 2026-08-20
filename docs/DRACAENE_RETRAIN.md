# 德蕾琪娜高质量重训与合成

本方案为源素材新建 `dracaene_mainline_v3_hq`，不会覆盖既有 v1/v2、RVC refresh 权重或回滚检查点。已核验源目录只有一个 15:56 的 MP4，且没有侧载/内嵌字幕，因此实际边界链为：Gemini 视频语义边界（可用时）→ 本地静音切分；SenseVoice 负责本地转写/语音情绪草稿，专名词表与 TUI 人审负责最终文本。

> 2026-08-19 的已制备工作区当前为 312 个静音候选片段、0 个已复核。
> 这些边界中仍有“星辉骑士／这个嘛”类语义错配，不得直接 export/train。
> 必须先完成下面的长视频分块 Gemini 重切和全量人审。

## 1. 环境和变量

```powershell
$repo = 'D:\voice-dataset-pipeline'
$workspace = 'D:\MaiBot2\runtime\dracaene_voice_training\dracaene_mainline_v3_hq'
$source = 'D:\DownKyi-1.0.24-1.win-x64\Media\3.0主线小挽昼妈妈语音纯享全收集【自用留档】（已换源更正）'
$profile = Join-Path $repo 'examples\project\dracaene.toml'

Set-Location $repo
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e '.[gemini,train]'
ffmpeg -version
```

GPT-SoVITS 和 RVC 继续使用各自环境，不安装进编排器 venv：

```powershell
& 'D:\GPT-SoVITS\.venv\Scripts\python.exe' -c "import torch; print(torch.__version__, torch.cuda.is_available())"
& 'D:\RVC-WebUI\.venv\Scripts\python.exe' -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

## 2. 配置与密钥

```powershell
voice-dataset init $workspace --config $profile
notepad (Join-Path $workspace 'config\pipeline.toml')
notepad (Join-Path $workspace 'secrets\credentials.toml')
```

`config/pipeline.toml` 是可审查的非敏感配置。API Key 只写入 Git 忽略的 `secrets/credentials.toml`：

```toml
[environment]
GEMINI_API_KEY = "<在本机填写>"
OPENAI_COMPAT_API_KEY = "" # 只有 emotion.provider=openai-compatible 时才需要
```

## 3. 预处理、校验和人工复核

```powershell
voice-dataset preprocess $workspace $source --mode video --skip-asr
$env:PYTHONPATH = Join-Path $repo 'src'
& 'D:\GPT-SoVITS\.venv\Scripts\python.exe' -m voice_dataset_pipeline transcribe $workspace
Remove-Item Env:PYTHONPATH -ErrorAction SilentlyContinue
voice-dataset status $workspace
voice-dataset label $workspace --provider gemini --language zh
voice-dataset review $workspace
```

对本机已存在的 `dracaene_mainline_v3_hq`，使用以下命令替换旧的 energy
边界。这会将 956 秒视频在本地静音附近转成约 10–13 个低码率预览块，
每个成功边界响应都会缓存，因此中断后不会重复已成功的 API 调用：

```powershell
$secureKey = Read-Host 'Gemini API key' -AsSecureString
$credential = [System.Net.NetworkCredential]::new('', $secureKey)
$env:GEMINI_API_KEY = $credential.Password
voice-dataset split $workspace --backend gemini --modality video --replace
voice-dataset quality $workspace --force
$env:PYTHONPATH = Join-Path $repo 'src'
& 'D:\GPT-SoVITS\.venv\Scripts\python.exe' -m voice_dataset_pipeline transcribe `
  $workspace --force --no-seed-labels
Remove-Item Env:PYTHONPATH -ErrorAction SilentlyContinue
voice-dataset label $workspace --provider gemini --language zh --force
Remove-Item Env:GEMINI_API_KEY -ErrorAction SilentlyContinue
Remove-Variable credential, secureKey -ErrorAction SilentlyContinue
voice-dataset status $workspace
voice-dataset review $workspace
```

`split --replace` 只有在全部分块均返回合法边界后才会覆盖清单；失败时保留旧切片。
以上命令会上传低码率预览块及各语音切片到 Gemini，必须由素材持有者在了解数据传输和配额消耗后主动执行。

`preprocess --skip-asr` 会递归导入、规范化音轨、按回退链拆分并执行声学质量门禁；随后的命令在已验证的 GPT-SoVITS GPU 环境内运行 SenseVoice。该角色模板会把常见误识别（德蕾奇娜、晚咒、阿尔达、林德威恩等）规范为角色专名，但人工复核仍必须检查：

- 不把两个不同情境或说话意图拼为一个切片；
- 不在主谓、动宾、转折续接处粗暴断句；
- 台词必须与音频逐字对应；
- 排除 BGM、其他角色、叠音、爆音、过长静音和明显 ASR 错误；
- 情绪数字键只确认实际听感，不盲从 SenseVoice/Gemini 草稿。

TUI 会显示前后条目。遇到“看来。” + “你们是……”这类过细切分时按 `M` 与下一段合并；合并完成后退出 TUI，重新运行质量门禁，并在 GPT-SoVITS 环境内强制转写：

```powershell
voice-dataset quality $workspace --force
$env:PYTHONPATH = Join-Path $repo 'src'
& 'D:\GPT-SoVITS\.venv\Scripts\python.exe' -m voice_dataset_pipeline transcribe `
  $workspace --force --no-seed-labels
Remove-Item Env:PYTHONPATH -ErrorAction SilentlyContinue
voice-dataset review $workspace
```

任意中断后重新执行 `review` 会从断点继续。需要重做边界时，先备份人工状态，再显式执行：

```powershell
voice-dataset preprocess $workspace --replace --skip-asr --force-quality
$env:PYTHONPATH = Join-Path $repo 'src'
& 'D:\GPT-SoVITS\.venv\Scripts\python.exe' -m voice_dataset_pipeline transcribe $workspace --force
Remove-Item Env:PYTHONPATH -ErrorAction SilentlyContinue
```

## 4. 导出和训练计划

```powershell
voice-dataset quality $workspace
voice-dataset status $workspace
voice-dataset export $workspace --speaker dracaene --language zh
voice-dataset train $workspace gpt-sovits
```

先检查计划和路径，不要直接训练：

```powershell
$plan = Join-Path $workspace 'training\gpt_sovits\dracaene_mainline_v3_hq\training-plan.json'
Get-Content -LiteralPath $plan
```

首轮预算为 SoVITS 8 epoch、GPT 12 epoch，并保存中间 checkpoint。该数值不是“最佳轮数”结论；训练后应固定台词、参考音频和随机种子逐 checkpoint A/B，按吐字、音色、韵律共同选型。

## 5. 执行 GPT-SoVITS 训练并登记模型

```powershell
voice-dataset train $workspace gpt-sovits `
  --execute `
  --register-as dracaene_v3_hq `
  --activate
```

日志位置：

```powershell
$run = Join-Path $workspace 'training\gpt_sovits\dracaene_mainline_v3_hq'
Get-ChildItem -LiteralPath (Join-Path $run 'logs') -Filter '*.log'
Get-Content -LiteralPath (Join-Path $run 'logs\train-gpt.log') -Tail 100
Get-Content -LiteralPath (Join-Path $run 'artifacts.json')
voice-dataset model list $workspace
```

## 6. 文本直接合成

不传参考音频时，工具从质量门禁通过的已复核训练集中，按目标情绪、时长、复核状态和 cluster 自动选择参考：

```powershell
$text = '你好，我是德蕾琪娜挽昼，是算枢局的总务官，希望你能安心在这里治疗，我会尽我所能帮助你的。'
$output = 'D:\MaiBot2\runtime\dracaene_voice_training\outputs\dracaene-v3-hq.wav'

voice-dataset emotion $workspace --text $text
voice-dataset synthesize $workspace --text $text --output $output
```

显式参考音频必须同时提供准确参考文本：

```powershell
voice-dataset synthesize $workspace `
  --model dracaene_v3_hq `
  --text $text `
  --emotion happy `
  --intensity 0.65 `
  --reference 'D:\reference\happy.wav' `
  --reference-text '这里填写与参考音频逐字一致的台词。' `
  --output 'D:\output\dracaene-explicit-ref.wav'
```

## 7. 可选 RVC 训练与后处理

RVC/Seed-VC 不是默认增强器；VC 可能破坏辅音、吐字和原始韵律。只有原始 SoVITS 已清楚、固定测试集 A/B 确认 VC 确实提升音色时才启用。

```powershell
# 先在 config/pipeline.toml 中启用 training.rvc.enabled=true，并使用独立实验名。
voice-dataset train $workspace rvc
voice-dataset train $workspace rvc --execute
```

取得对应推理 `.pth` 和 `added_*.index` 后，重新登记同名角色（权重路径以 `artifacts.json` 为准）：

```powershell
voice-dataset model register $workspace `
  --name dracaene_v3_hq `
  --persona dracaene `
  --repository 'D:\GPT-SoVITS' `
  --python 'D:\GPT-SoVITS\.venv\Scripts\python.exe' `
  --gpt '<GPT权重.ckpt>' `
  --sovits '<SoVITS权重.pth>' `
  --manifest '<导出目录\manifest.jsonl>' `
  --rvc-repository 'D:\RVC-WebUI' `
  --rvc-python 'D:\RVC-WebUI\.venv\Scripts\python.exe' `
  --rvc-model '<RVC推理权重.pth>' `
  --rvc-index '<added_索引.index>' `
  --activate
```

登记会自动封存权重、参考清单、推理底模、provider Python 源码树及 Git HEAD/tracked diff；
自动参考音频还会按清单中的 SHA-256 校验。旧注册记录缺少完整摘要时必须重新登记。此操作
不会复制或覆盖模型文件。

```powershell
voice-dataset synthesize $workspace `
  --text $text `
  --postprocess rvc `
  --output 'D:\output\dracaene-v3-hq-rvc.wav'
```

命令会同时保留 `dracaene-v3-hq-rvc.sovits.wav`。只有 RVC 版本的 ASR 一致性、音质指标和人工 A/B 均不劣于原始文件时，才把它当成最终产物。
