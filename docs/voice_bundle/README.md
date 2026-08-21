# GPT-SoVITS 本地语音包

`voice-bundle` 把已经完成训练、评估和显式 checkpoint 选择的 GPT-SoVITS
权重整理为一个可由推理插件读取的本地目录。它不会训练模型，也不会自动挑选“最后一个”
checkpoint。

## 安全与权利门禁

构建命令强制读取 `rights-attestation.json`。其中 `training_allowed` 和
`local_inference_allowed` 必须显式为 `true`；模型和参考音频的分发许可默认均为
`false`。声明中的 `dataset_fingerprint` 和 `training_plan_fingerprint` 还必须
分别与导出数据集 metadata 和训练计划精确一致，避免把一份泛化声明误用于另一批训练输入
或另一轮训练。权利声明只是一项可审计门禁，不会替调用者判断特定素材、声音主体或角色的
法律状态。

语音包只包含以下文件：

- 显式选择的一个 GPT `.ckpt`；
- 显式选择的一个 SoVITS `.pth`；
- `reference-profile.json` 明确列出的推理参考 WAV；
- `voice-bundle.json`。

原始媒体、切片集合、训练 manifest、标注、复核记录和权利证据均不会被复制。构建后，
loader 还会核对文件白名单、相对路径、字节数和 SHA-256；目录内出现未声明文件也会失败。

## 输入文件

示例位于：

- `examples/voice_bundle/selection.example.json`
- `examples/voice_bundle/reference-profile.example.json`
- `examples/voice_bundle/rendering-profile.example.json`（可选）
- `examples/voice_bundle/rights-attestation.example.json`

此外直接复用训练管线已经生成的：

- `training-plan.json`：提供模型版本、训练 fingerprint 和 GPT-SoVITS provider 来源；
- `training-result.json`：可选，必须与 plan fingerprint 一致；
- `artifacts.json`：可选，会核验其中已登记产物的 SHA-256，但不会从中自动选择模型；
- 导出数据集的 `metadata.json`：读取 fingerprint、条目数及 manifest/list 哈希；其中
  `dataset_list_sha256` 和 `included` 必须分别匹配训练计划的
  `metadata.dataset.dataset_list_sha256/items`。

`selection.json` 必须显式给出 `gpt_path/gpt_sha256`、
`sovits_path/sovits_sha256` 和 `training_plan_fingerprint`。评估报告不能直接充当
selection：应从报告派生一份显式 selection 文件，并把报告自己的 `plan_fingerprint`
写成可选的 `evaluation_fingerprint`，不要与训练计划 fingerprint 混用。

## 构建

在仓库根目录执行：

```powershell
$run = 'D:\voice-workspace\training\gpt_sovits\character_v1'

uv run --frozen voice-bundle build `
  --bundle-id 'character.voice.v2' `
  --display-name '角色语音 V2' `
  --selection 'D:\voice-evaluation\selection.json' `
  --reference-profile '.\examples\voice_bundle\reference-profile.example.json' `
  --rendering-profile '.\examples\voice_bundle\rendering-profile.example.json' `
  --rights-attestation 'D:\private\rights-attestation.json' `
  --training-plan (Join-Path $run 'training-plan.json') `
  --training-result (Join-Path $run 'training-result.json') `
  --artifacts (Join-Path $run 'artifacts.json') `
  --dataset-metadata 'D:\voice-workspace\training\exports\dataset-id\metadata.json' `
  --output 'D:\MaiBot2\runtime\gpt_sovits_voice\bundles\character.voice.v2'

if ($LASTEXITCODE -ne 0) {
    throw "voice bundle build failed: $LASTEXITCODE"
}
```

安装项目后也可以使用 `voice-bundle build ...`。输出目录必须不存在，避免意外覆盖既有模型。

## 校验与 schema

```powershell
uv run --frozen voice-bundle verify `
  'D:\MaiBot2\runtime\gpt_sovits_voice\bundles\character.voice.v2\voice-bundle.json'

uv run --frozen voice-bundle schema
```

`schema` 输出由 `VoiceBundleManifest` 生成的权威 JSON Schema；同一份 schema 也固定在
`schemas/voice-bundle.schema.json`，单测会检查两者完全一致。所有对象均
`additionalProperties: false`，未知字段、错误类型和未声明资产都会被拒绝。
严格契约使用 `schema_version = 2`；旧的宽松 v1 清单不会被 v2 verifier 当作同版接受。

## 插件消费字段

`voice-bundle.json` 的稳定接口为：

```text
schema_version = 2
bundle_id
display_name
engine.api = "api_v2"
engine.model_version
assets.gpt/sovits.{path,sha256,bytes}
references.default
references.items.<profile>.{description,auto_enabled,audio,prompt_text,prompt_lang,sha256,bytes}
rendering.{scene_catalog,default_scene,supported_scenes,scene_reference_profiles,profile_sampling_overrides}  # 可选
```

所有资产路径使用 `/` 分隔并相对于语音包根目录。`provenance`、`rights`、
`distribution` 和 `files` 用于审计与完整性校验；插件可以读取，但不应允许用户配置覆盖这些字段。

## 可选场景渲染契约

`--rendering-profile` 接受一个与 manifest `rendering` 字段同形的严格 JSON 对象。省略该
参数时，构建器不会写出 `rendering`，既有 v2 语音包的内容和默认语义不变。提供后，以下
五个字段必须全部存在，未知字段会被拒绝：

- `scene_catalog` 固定为 `vdp-scene-v1`；它标识由消费端实现的场景算法版本；
- `default_scene` 必须属于非空、无重复的 `supported_scenes`；场景只允许
  `speech/singing/audiobook/asmr/stage`；
- `scene_reference_profiles.<scene>.<requested_profile> = <bundled_profile>` 将“场景”与
  已有情绪/参考 profile 保持为两个维度；映射两端都必须存在于 `references.items`。未映射
  的 profile 保持原 profile，由消费端最后回退到 `references.default`；
- `profile_sampling_overrides.<bundled_profile>` 可写 `top_k/top_p/temperature/pace`
  中的一至四项，不能是空对象或显式 `null`。它记录已经评估过的模型/profile 采样建议，
  场景算法仍由 `scene_catalog` 决定；
- `profile_sampling_overrides` 和 `scene_reference_profiles` 没有条目时仍应显式写 `{}`。

该字段只描述可移植的推理意图和能力，不包含 SoX/RVC 可执行文件路径、启用开关或任意
effects/argv。SoX 是否启用、二进制位置以及固定 preset 的具体参数属于消费端本机策略；
严格模型会拒绝 `sox`、`binary`、`enabled`、`effects` 等额外字段。没有专用歌唱参考时，
不要把 `singing` 列入 `supported_scenes`。

仍只理解早期严格 v2 字段且设置 `extra=forbid` 的消费端会拒绝“带 rendering 的 v2”；应先
升级消费端 loader，再启用 `--rendering-profile`。省略 rendering 的 v2 可继续按 `speech`
默认场景、无 bundle 级采样覆盖处理。
