from __future__ import annotations

import ast
from pathlib import Path


def test_provider_worker_bootstrap_is_python_310_and_stdlib_only() -> None:
    source_path = (
        Path(__file__).resolve().parents[1]
        / "src"
        / "voice_dataset_pipeline"
        / "_provider_worker.py"
    )
    source = source_path.read_text(encoding="utf-8")
    tree = ast.parse(source, feature_version=(3, 10))
    imports: set[str] = set()
    for node in tree.body:
        if isinstance(node, ast.Import):
            imports.update(alias.name.split(".", 1)[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            imports.add(node.module.split(".", 1)[0] if node.module else "")
    assert imports <= {
        "__future__",
        "argparse",
        "contextlib",
        "json",
        "os",
        "sys",
        "pathlib",
        "typing",
    }
    assert "numpy" not in imports
    assert "soundfile" not in imports
    assert "voice_dataset_pipeline" not in imports
