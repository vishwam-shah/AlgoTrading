import re
import pathlib

FILES = [
    pathlib.Path('chapters/02_literature_survey.tex'),
    pathlib.Path('chapters/03_methodology.tex'),
    pathlib.Path('chapters/04_results.tex'),
    pathlib.Path('chapters/05_conclusions.tex'),
]

PAT_ADJ = re.compile(r"\\begin{adjustbox}{[^}]*}\\s*\\begin{tabular}{([^}]+)}(.*?)\\end{tabular}\\s*\\end{adjustbox}", re.S)
PAT_TAB = re.compile(r"\\begin{tabular}{([^}]+)}(.*?)\\end{tabular}", re.S)


def transform_block(spec: str, body: str) -> str | None:
    clean_spec = re.sub(r"\\|", " ", spec)
    clean_spec = re.sub(r"\\s+", " ", clean_spec).strip()

    # remove existing rules
    body_no_rules = re.sub(r"^\\s*\\\\(hline|toprule|midrule|bottomrule)\\s*$", "", body, flags=re.M)
    lines = [ln.rstrip() for ln in body_no_rules.splitlines() if ln.strip()]
    if not lines:
        return None

    header, *rest = lines
    new_lines = ["\\toprule", header]
    if rest:
        new_lines.append("\\midrule")
        new_lines.extend(rest)
    new_lines.append("\\bottomrule")
    new_body = "\n".join(new_lines)

    return (
        "\\begin{adjustbox}{max width=\\textwidth}\n"
        f"\\begin{{tabular}}{{{clean_spec}}}\n"
        f"{new_body}\n"
        "\\end{tabular}\n"
        "\\end{adjustbox}"
    )


def process(content: str) -> str:
    def repl_adj(match: re.Match) -> str:
        new_block = transform_block(match.group(1), match.group(2))
        return new_block or match.group(0)

    def repl_tab(match: re.Match) -> str:
        new_block = transform_block(match.group(1), match.group(2))
        return new_block or match.group(0)

    content = PAT_ADJ.sub(repl_adj, content)
    content = PAT_TAB.sub(repl_tab, content)
    return content


def main() -> None:
    for path in FILES:
        original = path.read_text(encoding="utf-8")
        updated = process(original)
        if updated != original:
            path.write_text(updated, encoding="utf-8")
            print(f"Formatted tables in {path}")
        else:
            print(f"No changes in {path}")


if __name__ == "__main__":
    main()
