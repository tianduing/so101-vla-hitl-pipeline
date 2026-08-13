from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MARKDOWN_ROOTS = (
    ROOT / "README.md",
    ROOT / "docs",
    ROOT / "problem_records",
    ROOT / "sim_mujoco" / "README.md",
    ROOT / "sim_mujoco" / "problem_records",
    ROOT / "sim_mujoco" / "reports",
)


def _markdown_files():
    for root in MARKDOWN_ROOTS:
        if root.is_file():
            yield root
        elif root.is_dir():
            yield from root.rglob("*.md")


def test_github_math_delimiters_are_supported_and_balanced():
    errors = []

    for path in sorted(_markdown_files()):
        in_fence = False
        fence_marker = ""
        display_delimiters = 0

        for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            stripped = line.strip()

            if stripped.startswith(("```", "~~~")):
                marker = stripped[:3]
                if not in_fence:
                    in_fence = True
                    fence_marker = marker
                elif marker == fence_marker:
                    in_fence = False
                    fence_marker = ""
                continue

            if in_fence:
                continue

            for unsupported in (r"\[", r"\]", r"\(", r"\)"):
                if unsupported in line:
                    errors.append(
                        f"{path.relative_to(ROOT)}:{line_number}: "
                        f"GitHub Markdown 不支持公式定界符 {unsupported}"
                    )

            if "$$" in line:
                if stripped != "$$":
                    errors.append(
                        f"{path.relative_to(ROOT)}:{line_number}: "
                        "块公式的 $$ 必须单独占一行"
                    )
                display_delimiters += line.count("$$")

        if in_fence:
            errors.append(f"{path.relative_to(ROOT)}: Markdown 代码块未闭合")
        if display_delimiters % 2:
            errors.append(f"{path.relative_to(ROOT)}: $$ 块公式定界符未成对")

    assert not errors, "\n" + "\n".join(errors)
