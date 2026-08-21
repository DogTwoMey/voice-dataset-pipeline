"""Narrow Python 3.10-compatible workers for pinned provider environments."""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import sys
from pathlib import Path
from typing import Any


@contextlib.contextmanager
def _working_directory(path: Path):
    previous = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(previous)


def _gpt_sovits(args: argparse.Namespace) -> int:
    import numpy as np
    import soundfile as sf

    repository = args.repository.resolve()
    for candidate in (repository, repository / "GPT_SoVITS"):
        value = str(candidate)
        if value not in sys.path:
            sys.path.insert(0, value)
    from GPT_SoVITS.TTS_infer_pack.TTS import TTS, TTS_Config

    request: dict[str, Any] = json.load(sys.stdin)
    destination = Path(str(request["output"])).expanduser().resolve()
    if destination.exists():
        raise FileExistsError(f"output already exists: {destination}")
    text = str(request["text"]).strip()
    if not text:
        raise ValueError("synthesis text is empty")
    reference = Path(str(request["reference_audio"])).expanduser().resolve()
    if not reference.is_file():
        raise FileNotFoundError(f"reference audio does not exist: {reference}")
    custom = {
        "device": args.device,
        "is_half": args.half == "true",
        "version": args.version,
        "t2s_weights_path": str(args.gpt_weights.resolve()),
        "vits_weights_path": str(args.sovits_weights.resolve()),
        "bert_base_path": str(
            repository / "GPT_SoVITS/pretrained_models/chinese-roberta-wwm-ext-large"
        ),
        "cnhuhbert_base_path": str(repository / "GPT_SoVITS/pretrained_models/chinese-hubert-base"),
    }
    with _working_directory(repository):
        pipeline = TTS(TTS_Config({"custom": custom}))
        pipeline_request = {
            "text": text,
            "text_lang": str(request["text_language"]),
            "ref_audio_path": str(reference),
            "prompt_text": str(request["reference_text"]),
            "prompt_lang": str(request["text_language"]),
            "top_k": int(request["top_k"]),
            "top_p": float(request["top_p"]),
            "temperature": float(request["temperature"]),
            "speed_factor": float(request["pace"]),
            "text_split_method": str(request.get("text_split_method", "cut0")),
            "fragment_interval": float(request.get("fragment_interval", 0.3)),
            "batch_size": 1,
            "seed": int(request["seed"]),
            "parallel_infer": True,
            "repetition_penalty": 1.35,
        }
        chunks: list[Any] = []
        sample_rate = 0
        for rate, audio in pipeline.run(pipeline_request):
            if sample_rate and int(rate) != sample_rate:
                raise ValueError("upstream returned mixed sample rates")
            sample_rate = int(rate)
            chunks.append(np.asarray(audio).reshape(-1))
    if not chunks or sample_rate <= 0:
        raise ValueError("GPT-SoVITS returned no audio")
    waveform = np.concatenate(chunks)
    if not waveform.size or not np.isfinite(waveform).all():
        raise ValueError("GPT-SoVITS returned empty or non-finite audio")
    destination.parent.mkdir(parents=True, exist_ok=True)
    sf.write(destination, waveform, sample_rate, subtype="PCM_16")
    print(
        json.dumps(
            {
                "sample_rate": sample_rate,
                "frames": int(waveform.size),
                "output": str(destination),
            },
            ensure_ascii=False,
        )
    )
    return 0


def _unpack_rvc_payload(status: object, payload: object) -> tuple[int, object]:
    if not isinstance(payload, (tuple, list)) or len(payload) != 2:
        raise ValueError(f"RVC inference failed: {status}")
    sample_rate, waveform = payload
    if sample_rate is None or waveform is None:
        raise ValueError(f"RVC inference failed: {status}")
    return int(sample_rate), waveform


def _rvc(args: argparse.Namespace) -> int:
    import numpy as np
    import soundfile as sf

    repository = args.repository.resolve()
    if str(repository) not in sys.path:
        sys.path.insert(0, str(repository))
    os.environ["weight_root"] = str(args.model.resolve().parent)  # noqa: SIM112
    os.environ.setdefault("rmvpe_root", str(repository / "assets" / "rmvpe"))
    os.environ.setdefault("index_root", str(repository / "logs"))
    os.environ.setdefault("outside_index_root", str(repository / "assets" / "indices"))
    sys.argv = [sys.argv[0]]
    from configs.config import Config
    from infer.vc.modules import VC

    converter = VC(Config())
    converter.get_vc(args.model.name)
    status, payload = converter.vc_single(
        0,
        str(args.input.resolve()),
        args.transpose,
        args.f0_method,
        str(args.index.resolve()),
        args.index_rate,
        0,
        args.rms_mix_rate,
        args.protect,
    )
    sample_rate, waveform = _unpack_rvc_payload(status, payload)
    values = np.asarray(waveform).reshape(-1)
    if not values.size or not np.isfinite(values).all():
        raise ValueError("RVC returned empty or non-finite audio")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    sf.write(args.output, values, sample_rate, subtype="PCM_16")
    print(json.dumps({"sample_rate": sample_rate, "frames": int(values.size)}))
    return 0


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(add_help=False)
    workers = parser.add_subparsers(dest="worker", required=True)
    gpt = workers.add_parser("gpt-sovits", add_help=False)
    gpt.add_argument("--repository", type=Path, required=True)
    gpt.add_argument("--gpt-weights", type=Path, required=True)
    gpt.add_argument("--sovits-weights", type=Path, required=True)
    gpt.add_argument("--version", required=True)
    gpt.add_argument("--device", required=True)
    gpt.add_argument("--half", choices=["true", "false"], required=True)
    rvc = workers.add_parser("rvc", add_help=False)
    rvc.add_argument("--repository", type=Path, required=True)
    rvc.add_argument("--model", type=Path, required=True)
    rvc.add_argument("--index", type=Path, required=True)
    rvc.add_argument("--input", type=Path, required=True)
    rvc.add_argument("--output", type=Path, required=True)
    rvc.add_argument("--f0-method", default="rmvpe")
    rvc.add_argument("--transpose", type=int, default=0)
    rvc.add_argument("--index-rate", type=float, default=0.45)
    rvc.add_argument("--rms-mix-rate", type=float, default=0.25)
    rvc.add_argument("--protect", type=float, default=0.33)
    return parser


def main() -> int:
    args = _parser().parse_args()
    if args.worker == "gpt-sovits":
        return _gpt_sovits(args)
    if args.worker == "rvc":
        return _rvc(args)
    raise ValueError(f"unknown provider worker: {args.worker}")


if __name__ == "__main__":
    raise SystemExit(main())
