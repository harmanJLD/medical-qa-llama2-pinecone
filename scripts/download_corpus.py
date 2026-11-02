"""Download a starter corpus of public-domain patient education PDFs from CDC Stacks.

CDC publications are works of the US federal government and are in the public domain.
Each file is checked to be a real PDF; any that fail to download are skipped.

    python scripts/download_corpus.py            # download into data/
    python scripts/download_corpus.py --list     # show the list only
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import time
from pathlib import Path

# (CDC Stacks id, file name, title)
CORPUS = [
    (55506, "4-steps-manage-diabetes.pdf", "4 Steps to Manage Your Diabetes for Life"),
    (251735, "prediabetes-preventing-type-2.pdf", "On Your Way to Preventing Type 2 Diabetes"),
    (140473, "prediabetes-could-it-be-you.pdf", "Prediabetes: Could It Be You?"),
    (251824, "high-blood-pressure-control-goal.pdf", "High Blood Pressure: How to Make Control Your Goal"),
    (72238, "high-blood-pressure.pdf", "High Blood Pressure"),
    (151351, "asthma-you-can-control.pdf", "You Can Control Your Asthma"),
    (46034, "chronic-kidney-disease-fact-sheet.pdf", "National Chronic Kidney Disease Fact Sheet"),
    (42540, "colorectal-cancer-screening.pdf", "Colorectal Cancer Screening Basic Fact Sheet"),
    (136614, "stroke-signs-and-symptoms.pdf", "Recognizing Signs and Symptoms of Stroke"),
    (48511, "falls-what-you-can-do.pdf", "What You Can Do to Prevent Falls"),
    (148192, "falls-stay-independent.pdf", "Stay Independent: Learn More About Fall Prevention"),
    (59197, "falls-home-safety-checklist.pdf", "Check for Safety: A Home Fall Prevention Checklist"),
    (130141, "concussion-fact-sheet.pdf", "Heads Up: Concussion Fact Sheet"),
    (39499, "arthritis-physical-activity.pdf", "Physical Activity: The Arthritis Pain Reliever"),
    (6868, "flu-vaccine-key-facts.pdf", "Key Facts About Influenza (Flu) Vaccine"),
    (110081, "flu-vaccine-what-you-need-to-know.pdf", "Influenza Vaccine: What You Need to Know"),
    (60428, "antibiotics-aware.pdf", "Be Antibiotics Aware: Smart Use, Best Care"),
]

URL = "https://stacks.cdc.gov/view/cdc/{id}/cdc_{id}_DS1.pdf"
PAGE = "https://stacks.cdc.gov/view/cdc/{id}"


def download(doc_id: int, dest: Path) -> str:
    """Download with curl (the CDC site blocks Python's default HTTP client) and check it is a real PDF."""
    tmp = dest.with_suffix(".part")
    result = subprocess.run(
        ["curl", "-sSL", "--fail", "--max-time", "120", "-o", str(tmp), URL.format(id=doc_id)],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        tmp.unlink(missing_ok=True)
        raise RuntimeError(result.stderr.strip() or f"curl exit code {result.returncode}")
    with open(tmp, "rb") as f:
        is_pdf = f.read(4) == b"%PDF"
    if not is_pdf:
        tmp.unlink(missing_ok=True)
        raise ValueError("response is not a PDF")
    tmp.rename(dest)
    return f"{dest.stat().st_size / 1024:.0f} KB"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path, default=Path("data"))
    parser.add_argument("--list", action="store_true")
    args = parser.parse_args()

    if args.list:
        for doc_id, name, title in CORPUS:
            print(f"{name:42} {URL.format(id=doc_id)}")
        return

    if not shutil.which("curl"):
        sys.exit("curl is required. Install it, or download the PDFs manually from the links in --list.")
    args.out.mkdir(parents=True, exist_ok=True)
    ok, failed, sources = 0, [], []
    for doc_id, name, title in CORPUS:
        dest = args.out / name
        if dest.exists():
            print(f"  exists   {name}")
            ok += 1
        else:
            try:
                size = download(doc_id, dest)
                print(f"  saved    {name} ({size})")
                ok += 1
                time.sleep(0.5)
            except Exception as exc:
                print(f"  skipped  {name}: {exc}")
                failed.append(name)
                continue
        sources.append(f"| {name} | {title} | {PAGE.format(id=doc_id)} |")

    (args.out / "SOURCES.md").write_text(
        "Public-domain CDC publications downloaded by `scripts/download_corpus.py`.\n\n"
        "| File | Title | Source |\n|---|---|---|\n" + "\n".join(sources) + "\n"
    )
    print(f"\n{ok} PDFs ready in {args.out}/, {len(failed)} skipped. Next: python scripts/ingest.py")
    if not ok:
        sys.exit(1)


if __name__ == "__main__":
    main()
