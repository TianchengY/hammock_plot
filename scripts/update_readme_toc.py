from pathlib import Path
import re


README = Path(__file__).resolve().parents[1] / "README.md"
MAX_LEVEL = 3


def github_slug(title, used):
    slug = title.strip().lower()
    slug = re.sub(r"[^\w\s-]", "", slug)
    slug = re.sub(r"\s+", "-", slug)

    base = slug
    count = used.get(base, 0)
    used[base] = count + 1
    if count:
        return f"{base}-{count}"
    return base


def collect_headings(lines):
    headings = []
    used = {}

    for line in lines:
        match = re.match(r"^(#{2,3})\s+(.+?)\s*$", line)
        if not match:
            continue

        level = len(match.group(1))
        title = match.group(2).strip()
        if level > MAX_LEVEL or title == "Table of Contents":
            continue

        headings.append((level, title, github_slug(title, used)))

    return headings


def build_toc(headings):
    toc = ["## Table of Contents", ""]
    for level, title, slug in headings:
        indent = "  " * (level - 2)
        toc.append(f"{indent}- [{title}](#{slug})")
    toc.append("")
    return toc


def replace_toc(lines, toc):
    start = None
    for i, line in enumerate(lines):
        if line.strip() == "## Table of Contents":
            start = i
            break

    if start is None:
        for i, line in enumerate(lines):
            if line.startswith("## "):
                return lines[:i] + toc + [""] + lines[i:]
        return lines + [""] + toc

    end = start + 1
    while end < len(lines) and not lines[end].startswith("## "):
        end += 1

    return lines[:start] + toc + [""] + lines[end:]


def main():
    text = README.read_text(encoding="utf-8")
    newline = "\r\n" if "\r\n" in text else "\n"
    lines = text.splitlines()

    headings = collect_headings(lines)
    updated = replace_toc(lines, build_toc(headings))
    README.write_text(newline.join(updated) + newline, encoding="utf-8")


if __name__ == "__main__":
    main()
