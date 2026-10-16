"""Reproduce the three claims made about task2/adversarial_cv.pdf.

Run:  python task2/verify_adversarial_cv.py

1. The PDF text layer - exactly what the grader's loader (markitdown, no OCR) hands to a
   verifier - contains the truthful title "Manager, AIA" and NOT "Senior Manager, AIA".
2. The word "Senior" is nevertheless visible on the page: there is real black ink immediately
   to the left of the text-layer title, and the page's embedded image sits exactly there.
3. That ink is the word "Senior": the embedded bitmap is compared against reference renderings
   of several candidate words produced by the same code path, and matches "Senior" exactly.

A fourth check confirms the rest of the CV is truthful, so the inflated title is the only
false claim in the document.
"""

import sys
from pathlib import Path

import pymupdf

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from make_adversarial_cv import BOLD_FONT, _word_pixmap  # noqa: E402

PDF = HERE / "adversarial_cv.pdf"
TARGET_TITLE = "Manager, AIA"
INFLATED_TITLE = "Senior Manager, AIA"


def text_layer(page: "pymupdf.Page") -> str:
    return page.get_text("text")


def main() -> int:
    doc = pymupdf.open(PDF)
    page = doc[0]
    text = text_layer(page)
    ok = True

    print("check 1: text layer versus drawn layer")
    print(f"  text layer contains {TARGET_TITLE!r}:  {TARGET_TITLE in text}")
    print(f"  text layer contains {INFLATED_TITLE!r}: {INFLATED_TITLE in text}")
    ok &= TARGET_TITLE in text and INFLATED_TITLE not in text

    print("\ncheck 2: the drawn word is visible, left of the truthful title")
    hits = page.search_for(TARGET_TITLE)
    if not hits:
        print("  FAIL: title not found in the text layer")
        return 1
    title = hits[0]
    clip = pymupdf.Rect(title.x0 - 60, title.y0 - 4, title.x0 + 1, title.y1 + 4)
    scale = 4.0
    pix = page.get_pixmap(matrix=pymupdf.Matrix(scale, scale), clip=clip, colorspace=pymupdf.csGRAY)
    dark = sum(1 for b in pix.samples if b < 128)
    print(f"  dark pixels in the 60pt strip left of x={title.x0:.2f}: {dark}")
    images = page.get_images(full=True)
    print(f"  images on page 1: {len(images)}")
    if images:
        rect = page.get_image_rects(images[0][0])[0]
        print(f"  image rect: ({rect.x0:.1f}, {rect.y0:.1f}) - ({rect.x1:.1f}, {rect.y1:.1f})")
        print(f"  text title starts at x={title.x0:.2f} -> image abuts the title")
    print(f"  contrast on the title row: darkest={min(pix.samples)} lightest={max(pix.samples)}")
    ok &= dark > 0

    print("\ncheck 3: template match against candidate words")
    xref = images[0][0]
    raw = doc.extract_image(xref)
    embedded = pymupdf.Pixmap(raw["image"])
    if embedded.n - embedded.alpha >= 3:
        embedded = pymupdf.Pixmap(pymupdf.csGRAY, embedded)
    scored = []
    for word in ["Senior", "Manager", "Analyst", "Lead", "Head", "Officer"]:
        ref, _, _ = _word_pixmap(word, BOLD_FONT, 11)
        if ref.n - ref.alpha >= 3:
            ref = pymupdf.Pixmap(pymupdf.csGRAY, ref)
        if (ref.width, ref.height) != (embedded.width, embedded.height):
            scored.append((float("inf"), word, "size mismatch"))
            continue
        pairs = list(zip(embedded.samples, ref.samples))
        mean = sum(abs(a - b) for a, b in pairs) / len(pairs)
        scored.append((mean, word, f"{ref.width}x{ref.height}"))
    for mean, word, note in sorted(scored, key=lambda t: t[0]):
        shown = "exact match" if mean == 0 else (f"mean|diff|={mean:.4f}" if mean != float("inf") else note)
        print(f"  {word:<10} {shown}")
    best = min(scored)[1]
    print(f"  best match: {best!r}")
    ok &= best == "Senior"

    print("\ncheck 4: the rest of the visible CV is truthful")
    for expected in ["Analyst, Standard Chartered", "2019 - 2022", "Analyst, HSBC", "2016 - 2019",
                     "BSc in Finance - Graduated 2016",
                     "Hong Kong University of Science and Technology (HKUST)",
                     "Financial Analysis, Excel, SQL, Risk Modeling"]:
        present = expected in text
        ok &= present
        print(f"  {expected!r}: {present}")

    print("\nRESULT:", "all checks passed" if ok else "SOME CHECKS FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
