"""Generate separate project and sensitive configuration files."""

from __future__ import annotations

import argparse
import importlib
import sys
from pathlib import Path

try:
    _config_module = importlib.import_module("voice_dataset_pipeline.config")
except ModuleNotFoundError:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
    _config_module = importlib.import_module("voice_dataset_pipeline.config")

generate_default_config_layout = _config_module.generate_default_config_layout


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="生成可提交的项目配置和被 Git 忽略的敏感配置",
    )
    parser.add_argument(
        "root",
        nargs="?",
        type=Path,
        default=Path.cwd(),
        help="目标工作区；默认当前目录",
    )
    parser.add_argument("--overwrite-project", action="store_true")
    parser.add_argument("--overwrite-secrets", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    layout = generate_default_config_layout(
        args.root,
        overwrite_project=args.overwrite_project,
        overwrite_secrets=args.overwrite_secrets,
    )
    print(f"项目配置（可提交）: {layout.project}")
    print(f"敏感配置（Git 忽略）: {layout.secrets}")
    print(f"敏感目录规则: {layout.secrets_gitignore}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
