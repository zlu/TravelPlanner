from pathlib import Path


def normalize_output_dir(output_dir: str, base_dir: Path | None = None) -> str:
    out = Path(output_dir)
    if not out.is_absolute():
        base = Path(base_dir) if base_dir is not None else Path.cwd()
        out = (base / out).resolve()
    parts = []
    for part in out.parts:
        if parts and part == "evaluation" and parts[-1] == "evaluation":
            continue
        parts.append(part)
    return str(Path(*parts))
