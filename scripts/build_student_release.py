from __future__ import annotations

import shutil
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DESTINATION = ROOT / "dist" / "student_assignment"
FILES = [
    ".gitignore",
    "pyproject.toml",
    "uv.lock",
    "prepare_data.py",
    "train.py",
    "evaluate.py",
    "analyze_results.py",
    "assets/tokenizer.json",
    "assets/tokenizer_metadata.json",
    "docs/ASSIGNMENT.md",
    "docs/REPORT_TEMPLATE.md",
]
TREES = ["src", "tests", "configs"]


def strip_solutions(text: str, source: Path) -> str:
    output: list[str] = []
    inside = False
    marker_indent = ""
    for line in text.splitlines(keepends=True):
        stripped = line.strip()
        if stripped == "# BEGIN SOLUTION":
            if inside:
                raise ValueError(f"nested solution marker in {source}")
            inside = True
            marker_indent = line[: len(line) - len(line.lstrip())]
            output.append(f'{marker_indent}raise NotImplementedError("TODO: implement this exercise")\n')
            continue
        if stripped == "# END SOLUTION":
            if not inside:
                raise ValueError(f"unmatched solution end marker in {source}")
            inside = False
            continue
        if not inside:
            output.append(line)
    if inside:
        raise ValueError(f"unclosed solution marker in {source}")
    return "".join(output)


def copy_file(relative: str, destination_relative: str | None = None) -> None:
    source = ROOT / relative
    destination = DESTINATION / (destination_relative or relative)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if source.suffix == ".py":
        destination.write_text(strip_solutions(source.read_text(), source))
    elif relative == "pyproject.toml":
        destination.write_text(source.read_text().replace('testpaths = ["tests", "instructor_tests"]', 'testpaths = ["tests"]'))
    else:
        shutil.copy2(source, destination)


def main() -> None:
    if DESTINATION.exists():
        shutil.rmtree(DESTINATION)
    for relative in FILES:
        if relative == "docs/ASSIGNMENT.md":
            copy_file(relative, "README.md")
        else:
            copy_file(relative)
    for tree in TREES:
        for source in (ROOT / tree).rglob("*"):
            if not source.is_file() or "__pycache__" in source.parts:
                continue
            relative = source.relative_to(ROOT)
            if relative.parts[:2] == ("configs", "sweeps") or relative == Path("configs/original_composite.yaml"):
                continue
            copy_file(str(relative))

    forbidden = ["BEGIN SOLUTION", "END SOLUTION", "instructor_tests", "cs336_assignment1", "Transformer_from_Scratch"]
    for path in DESTINATION.rglob("*"):
        if not path.is_file():
            continue
        if path.stat().st_size > 5_000_000:
            raise ValueError(f"oversized release file: {path}")
        if path.suffix in {".py", ".md", ".yaml", ".toml"}:
            text = path.read_text(errors="ignore")
            for needle in forbidden:
                if needle in text:
                    raise ValueError(f"forbidden release content {needle!r} in {path}")
    print(f"student release built at {DESTINATION}")


if __name__ == "__main__":
    main()
