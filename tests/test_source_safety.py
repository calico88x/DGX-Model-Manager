from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]


def test_no_known_private_fixture_values_in_publishable_text():
    # Assemble development-only fixture values so the regression test does not itself
    # place those complete values into a publishable source file.
    blocked=["192.168."+"3.51", "/home/"+"nova", "zgx-"+"40e6"]
    ignored={ROOT/"tests"/"test_source_safety.py"}
    for p in ROOT.rglob("*"):
        if p in ignored or not p.is_file() or p.suffix.lower() not in {".py",".js",".css",".html",".md",".json",".yaml",".yml",".sh",".txt"}:
            continue
        text=p.read_text(errors="ignore")
        for term in blocked:
            assert term not in text, f"Private development fixture leaked into {p}: {term}"
