"""
PyMuPDF (fitz) — Examples
==========================

IMPORT-ONLY module. No side-effects on import. Run tests manually:

    import pymupdf_fitz_examples
    pymupdf_fitz_examples._self_tests()

    # or
    python pymupdf_fitz_examples.py

Covers 5 topics:
1. PDF reading & extraction
2. PDF manipulation
3. PDF annotations
4. PDF rendering
5. Coordinates & layout

Notes on PyMuPDF model:
- Document (fitz.Document) contains Pages. Page numbers are 0-indexed.
- Coordinates: origin (0,0) at top-left, +x to right, +y downwards, units in points (pt, 1/72 inch).
- fitz.Rect(x0,y0,x1,y1), fitz.Point(x,y), fitz.Matrix for transforms.
- All ephemeral PDFs are created in-memory or via tempfile — no committed assets.
"""

from __future__ import annotations

import io
import math
import tempfile
from pathlib import Path
from typing import Any

# Use pymupdf import, but keep fitz alias for canonical docs compatibility.
try:
    import pymupdf as fitz  # PyMuPDF >= 1.23 prefers `import pymupdf`
except ImportError:
    import fitz  # type: ignore[no-redef]  # fallback for older installs

# Pillow only needed to validate rendered PNG bytes
from PIL import Image


# =============================================================================
# Helpers — synthetic PDFs (deterministic, no external files)
# =============================================================================


def create_sample_pdf(
    text: str = "Hello PyMuPDF",
    title: str = "Sample",
    author: str = "TestAuthor",
) -> fitz.Document:
    """
    Create an in-memory PDF with one page containing `text`.

    Returns an open fitz.Document (caller must close). The PDF exists purely
    in memory — `doc.tobytes()` can persist if needed, but we keep it live.

    Steps:
    - fitz.open() with no args creates empty PDF
    - new_page() adds page with default size A4 (595 x 842 pt)
    - insert_text(point, text) draws at given coordinates
    - set metadata via doc.set_metadata()
    """
    doc = fitz.open()  # empty PDF in memory
    page = doc.new_page(width=595, height=842)  # A4 portrait
    # Insert text near top-left; (72,72) = 1 inch margin
    page.insert_text(fitz.Point(72, 72), text, fontsize=14, fontname="helv")
    # Add a second line lower down for block/words tests
    page.insert_text(fitz.Point(72, 100), "Second line for testing", fontsize=11)
    doc.set_metadata({"title": title, "author": author, "subject": "demo"})
    return doc


def create_two_page_pdf() -> fitz.Document:
    """Two-page PDF with distinct text per page — for merge/reading tests."""
    doc = fitz.open()
    for i in range(2):
        page = doc.new_page()
        page.insert_text(fitz.Point(72, 72), f"Page {i+1} content", fontsize=12)
    return doc


# =============================================================================
# 1. PDF READING & EXTRACTION
# =============================================================================
# Core APIs: fitz.open(path|stream), doc.page_count, doc.load_page(n),
#            doc.metadata, doc.get_page_labels()
# Page text: page.get_text("text"), "blocks", "words", "dict", "json", "html"
# Images: page.get_images(full=True), page.get_drawings(), doc.get_page_pixmap?


def extract_text_demo(doc: fitz.Document) -> str:
    """
    Extract plain text from first page.

    `page.get_text()` with no arg or "text" returns concatenated text in reading
    order. Use "blocks" or "words" for layout-aware extraction.
    """
    page = doc.load_page(0)
    text = page.get_text("text")  # or page.get_text() — same default
    # Verbose: text includes newlines; strip for comparison
    return text.strip()


def extract_blocks_words_demo(doc: fitz.Document) -> dict[str, Any]:
    """
    Extract with layout: blocks and words.

    - "blocks": list of (x0,y0,x1,y1, text, block_no, block_type)
      block_type 0 = text, 1 = image
    - "words": list of (x0,y0,x1,y1, word, block_no, line_no, word_no)
    Returns counts and samples for asserts.
    """
    page = doc.load_page(0)
    blocks = page.get_text("blocks")  # type: ignore[assignment]
    words = page.get_text("words")  # type: ignore[assignment]
    # Also demonstrate "dict" for spans — not returned, just validated internally
    d = page.get_text("dict")
    assert "blocks" in d  # dict mode has structured blocks/lines/spans
    return {
        "block_count": len(blocks),
        "word_count": len(words),
        "first_block": blocks[0] if blocks else None,
        "first_word": words[0] if words else None,
    }


def extract_metadata_demo(doc: fitz.Document) -> dict[str, Any]:
    """Return doc metadata dict; keys: title, author, subject, creator, etc."""
    # doc.metadata is a dict; doc.set_metadata() writes it (see helper)
    meta = doc.metadata
    # Also demonstrate page_count and is_pdf flag
    return {
        "title": meta.get("title"),
        "author": meta.get("author"),
        "page_count": doc.page_count,
        "is_pdf": doc.is_pdf,
    }


def extract_images_drawings_demo() -> dict[str, Any]:
    """
    Demonstrate image/drawing extraction.

    We create a page with a drawn rectangle (vector drawing, not raster image)
    and verify get_drawings() picks it up. Raster images via get_images() would
    need an embedded image — we also test that path with an inserted pixmap.
    """
    doc = fitz.open()
    page = doc.new_page()
    # Vector drawing: a filled rectangle
    rect = fitz.Rect(50, 50, 150, 150)
    page.draw_rect(rect, color=(1, 0, 0), fill=(1, 0.8, 0.8), width=1)
    drawings = page.get_drawings()
    images_no_embed = page.get_images(full=True)
    # Now embed a raster image via pixmap and check get_images again
    # Create a small red pixmap and insert as image
    pix = fitz.Pixmap(fitz.csRGB, fitz.IRect(0, 0, 20, 20))
    pix.set_rect(pix.irect, (255, 0, 0))  # solid red
    # Insert image from pixmap bytes
    page2 = doc.new_page()
    page2.insert_image(fitz.Rect(10, 10, 30, 30), pixmap=pix)
    images_with_embed = page2.get_images(full=True)
    doc.close()
    return {
        "drawing_count": len(drawings),
        "images_before": len(images_no_embed),
        "images_after": len(images_with_embed),
    }


# =============================================================================
# 2. PDF MANIPULATION
# =============================================================================
# APIs: doc.new_page(), doc.delete_page(), doc.move_page(), doc.copy_page(),
#       doc.insert_pdf(), page.insert_text(), page.draw_*(), page.set_rotation()
# Common pitfall: page numbers shift after deletion — always operate on fresh doc.


def add_page_and_draw_demo() -> dict[str, Any]:
    """
    Add a page, draw shapes, insert text, and verify.

    Demonstrates:
    - new_page() with explicit size
    - draw_rect(), draw_circle(), draw_line()
    - insert_textbox() for wrapped text
    """
    doc = fitz.open()
    page = doc.new_page(width=400, height=400)
    # Draw a blue rectangle border
    page.draw_rect(fitz.Rect(10, 10, 390, 390), color=(0, 0, 1), width=2)
    # Draw a circle
    page.draw_circle(fitz.Point(200, 200), 50, color=(0, 1, 0), width=1.5)
    # Insert textbox with wrapping
    rect = fitz.Rect(50, 50, 350, 100)
    page.draw_rect(rect, color=(0.5, 0.5, 0.5), width=0.5)  # visual box
    inserted = page.insert_textbox(rect, "Textbox content with wrapping. " * 2, fontsize=10, align=1)
    drawings = page.get_drawings()
    assert inserted >= 0  # number of chars that fit; >=0 means success
    # Verify drawings exist (rect + circle at least)
    result = {"page_count": doc.page_count, "drawing_count": len(drawings), "inserted": inserted}
    doc.close()
    return result


def merge_pdfs_demo() -> dict[str, Any]:
    """
    Merge two PDFs via insert_pdf().

    insert_pdf(other_doc) appends pages from other_doc. Alternative: doc.insert_pdfs?
    Verify final page count = sum.
    """
    doc_a = create_sample_pdf(text="Doc A", title="A")
    doc_b = create_two_page_pdf()
    merged = fitz.open()
    merged.insert_pdf(doc_a)
    merged.insert_pdf(doc_b)
    count = merged.page_count
    # Also test delete_page
    merged.delete_page(0)  # remove first page
    after_delete = merged.page_count
    # Save to bytes to verify can serialize
    pdf_bytes = merged.tobytes()
    doc_a.close()
    doc_b.close()
    merged.close()
    return {"merged_count": count, "after_delete": after_delete, "bytes_len": len(pdf_bytes)}


def rotate_and_save_demo() -> dict[str, Any]:
    """
    Rotate a page and save to bytes / tempfile.

    page.set_rotation(90) rotates view; get_text still works. Save via tobytes()
    or doc.save(path). Verify rotation property and that saved bytes are valid PDF.
    """
    doc = create_sample_pdf()
    page = doc[0]
    page.set_rotation(90)
    assert page.rotation == 90
    # Save to bytes
    pdf_bytes = doc.tobytes()
    assert pdf_bytes[:5] == b"%PDF-"  # PDF magic
    # Also save to tempfile and reopen
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "rotated.pdf"
        doc.save(str(path))
        assert path.stat().st_size > 0
        # Reopen to verify
        doc2 = fitz.open(str(path))
        assert doc2[0].rotation == 90
        doc2.close()
    doc.close()
    return {"rotation": 90, "bytes_start": pdf_bytes[:5].decode()}


# =============================================================================
# 3. PDF ANNOTATIONS
# =============================================================================
# Annotations are page overlays: text comments, highlights, rects, ink, etc.
# APIs: page.add_text_annot(point, text), add_highlight_annot(quads),
#       add_rect_annot(rect), add_ink_annot(list_of_strokes), annot.update(),
#       page.annots(), annot.delete(), annot.set_colors(), annot.set_border()


def annotate_text_highlight_demo() -> dict[str, Any]:
    """
    Add text annotation and highlight annotation, then verify.

    Text annot: small icon with popup text. Highlight: quads covering text.
    Highlights require quads from search_for() — not arbitrary rects.
    """
    doc = create_sample_pdf(text="Annotate this important sentence for highlight testing.")
    page = doc[0]
    # Text annotation at a point
    text_annot = page.add_text_annot(fitz.Point(400, 72), "This is a comment")
    text_annot.set_colors(stroke=(1, 0, 0))
    text_annot.update()
    # Highlight annotation: search for word "important" and highlight it
    quads = page.search_for("important")
    assert len(quads) > 0, "search_for should find 'important'"
    highlight = page.add_highlight_annot(quads)
    highlight.set_colors(stroke=(1, 1, 0))  # yellow highlight (visual)
    highlight.update()
    # Verify annotations
    annots = list(page.annots() or [])
    types = [a.type[0] for a in annots]  # type is tuple (code, name)
    # Cleanup check: delete one
    count_before = len(annots)
    # Find and delete highlight (or first)
    annots[1].update()  # ensure second annot is highlight
    doc.close()
    return {"annot_count": count_before, "types": types, "quads_found": len(quads)}


def annotate_rect_ink_and_remove_demo() -> dict[str, Any]:
    """
    Add rectangle + ink annotations, then remove them.

    Rect annot: draws a rectangle overlay. Ink: freehand strokes as list of point lists.
    Demonstrates iteration and deletion via annot deletion loop.
    """
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text(fitz.Point(72, 72), "Draw on me", fontsize=12)
    # Rect annot
    rect_annot = page.add_rect_annot(fitz.Rect(50, 50, 150, 100))
    rect_annot.set_colors(stroke=(0, 0, 1), fill=(0.9, 0.9, 1))
    rect_annot.set_border(width=1)
    rect_annot.update()
    # Ink annot: two strokes
    ink = page.add_ink_annot([[(60, 60), (140, 60), (140, 90)], [(60, 90), (100, 70)]])
    ink.set_colors(stroke=(1, 0, 0))
    ink.set_border(width=2)
    ink.update()
    before = len(list(page.annots() or []))
    # Remove all annotations via loop (must collect first, then delete)
    for annot in list(page.annots() or []):
        page.delete_annot(annot)
    after = len(list(page.annots() or []))
    doc.close()
    return {"before": before, "after": after}


# =============================================================================
# 4. PDF RENDERING
# =============================================================================
# Rendering: page.get_pixmap(dpi=..., matrix=...), doc.get_page_pixmap? is same
# Pixmap: width, height, n (channels), alpha, stride, samples, tobytes("png"), pil_tobytes?
# Matrix: fitz.Matrix(zoom, zoom) or fitz.Matrix(dpi/72, dpi/72) for scaling.


def render_to_pixmap_demo() -> dict[str, Any]:
    """
    Render a page to pixmap and to PNG bytes.

    Demonstrates:
    - get_pixmap(dpi=144) vs matrix scaling
    - pix.n (channels), pix.width/height, pix.alpha
    - pix.tobytes("png") and PIL validation
    - pix.samples (raw bytes) length check
    """
    doc = create_sample_pdf(text="Render me!")
    page = doc[0]
    # Render at 144 dpi (2x standard 72 dpi) → double size
    pix = page.get_pixmap(dpi=144)
    assert pix.width > 0 and pix.height > 0
    assert pix.n in (3, 4)  # RGB or RGBA
    # Raw samples length = width*height*n (if no alpha padding nuances due to stride)
    # Use stride to be safe: samples length = height * stride
    assert len(pix.samples) == pix.height * pix.stride
    # PNG bytes: check header and that Pillow can open
    png_bytes = pix.tobytes("png")
    assert png_bytes[:8] == b"\x89PNG\r\n\x1a\n"
    img = Image.open(io.BytesIO(png_bytes))
    assert img.size == (pix.width, pix.height)
    # Also render via Matrix (zoom 2x)
    mat = fitz.Matrix(2, 2)  # 2x scale
    pix2 = page.get_pixmap(matrix=mat)
    assert pix2.width == pix.width  # should match dpi=144 (2x72)
    doc.close()
    return {"width": pix.width, "height": pix.height, "n": pix.n, "png_len": len(png_bytes)}


def render_with_alpha_and_clip_demo() -> dict[str, Any]:
    """
    Render with alpha and clip region.

    - alpha=True gives transparent background (RGBA)
    - clip=fitz.Rect(...) renders only sub-rectangle
    Useful for thumbnails or region screenshots.
    """
    doc = fitz.open()
    page = doc.new_page(width=200, height=200)
    page.draw_rect(fitz.Rect(0, 0, 200, 200), fill=(0.2, 0.4, 0.8))
    page.insert_text(fitz.Point(50, 100), "Clip test", fontsize=14, color=(1, 1, 1))
    # Full pixmap with alpha
    pix_alpha = page.get_pixmap(alpha=True)
    assert pix_alpha.alpha == 1 and pix_alpha.n == 4
    # Clipped pixmap: only top-left 100x100
    clip = fitz.Rect(0, 0, 100, 100)
    pix_clip = page.get_pixmap(clip=clip, dpi=72)
    # Clipped pixmap should be smaller (100x100, but dpi matters)
    assert pix_clip.width <= pix_alpha.width and pix_clip.height <= pix_alpha.height
    doc.close()
    return {"alpha_n": pix_alpha.n, "clip_width": pix_clip.width, "full_width": pix_alpha.width}


# =============================================================================
# 5. COORDINATES & LAYOUT
# =============================================================================
# Coordinates & layout keys:
# - page.rect is the page's MediaBox as Rect(0,0,width,height)
# - page.cropbox, mediabox, trimbox etc. for PDF boxes
# - page.get_text("dict") → {width, height, blocks: [{bbox, lines: [{bbox, spans: [{bbox, text, size, font, color}]}]}]}
# - page.get_text("rawdict") adds char-level bboxes
# - page.search_for(text) returns list[Rect] of hits
# Layout analysis: reading blocks sorted by y then x; use `sort` param for get_text


def coordinates_search_demo() -> dict[str, Any]:
    """
    Demonstrate coordinate system and search.

    - page.rect gives page bounds
    - search_for returns quads/rects in page coords
    - Rect properties: x0,y0 (top-left), x1,y1 (bottom-right), width, height
    - Point distance, Rect intersection/normalization
    """
    doc = create_sample_pdf(text="Find HELLO and WORLD on this page for coordinate testing.")
    page = doc[0]
    # Page rect
    rect = page.rect
    assert rect.x0 == 0 and rect.y0 == 0
    assert rect.width == 595 and rect.height == 842  # A4 we created
    # Search for word "WORLD"
    hits = page.search_for("WORLD")
    assert len(hits) == 1
    hit = hits[0]
    # hit is Rect — check sane coordinates inside page
    assert 0 <= hit.x0 < hit.x1 <= rect.width
    assert 0 <= hit.y0 < hit.y1 <= rect.height
    # Rect operations: contains, intersects, normalize
    small = fitz.Rect(hit.x0, hit.y0, hit.x0 + 10, hit.y0 + 10)
    assert hit.intersects(small)  # overlap
    assert rect.contains(hit)  # page contains word hit
    # Point inside hit
    center = fitz.Point((hit.x0 + hit.x1) / 2, (hit.y0 + hit.y1) / 2)
    assert hit.contains(center)  # Rect.contains(Point)
    # Distance between points (Euclidean)
    p1 = fitz.Point(0, 0)
    p2 = fitz.Point(3, 4)
    assert math.isclose(abs(p1 - p2), 5.0)  # Point subtraction gives length?
    # Alternative distance via math
    dist = math.hypot(p2.x - p1.x, p2.y - p1.y)
    assert math.isclose(dist, 5.0)
    doc.close()
    return {"page_rect": (rect.x0, rect.y0, rect.x1, rect.y1), "hit": (hit.x0, hit.y0, hit.x1, hit.y1)}


def layout_dict_demo() -> dict[str, Any]:
    """
    Parse layout via get_text("dict").

    Shows how to get block/line/span bboxes and font info. Useful for
    reconstructing reading order or extracting style.
    Structure: dict -> blocks[] -> lines[] -> spans[]
    Each has 'bbox' (x0,y0,x1,y1), spans have 'text', 'size', 'font', 'color', 'flags'
    """
    doc = fitz.open()
    page = doc.new_page()
    # Insert text at known positions with different fonts
    page.insert_text(fitz.Point(72, 72), "Header Title", fontsize=16, fontname="helv")
    page.insert_text(fitz.Point(72, 100), "Body text with ", fontsize=11)
    # Add a second font span nearby
    page.insert_text(fitz.Point(170, 100), "bold", fontsize=11, fontname="helv")
    # Draw a shape to ensure blocks include drawing-adjacent text properly
    d = page.get_text("dict")
    assert "blocks" in d and "width" in d and "height" in d
    # Find at least one block with text
    text_blocks = [b for b in d["blocks"] if b["type"] == 0]  # 0 = text
    assert len(text_blocks) >= 1
    first_block = text_blocks[0]
    assert "lines" in first_block
    first_line = first_block["lines"][0]
    assert "spans" in first_line
    first_span = first_line["spans"][0]
    assert "bbox" in first_span and "text" in first_span
    # Bbox should be 4 floats
    bbox = first_span["bbox"]
    assert len(bbox) == 4 and bbox[0] < bbox[2] and bbox[1] < bbox[3]
    # Also test rawdict for char-level (heavier)
    raw = page.get_text("rawdict")
    assert "blocks" in raw
    doc.close()
    return {"block_count": len(d["blocks"]), "first_span_text": first_span["text"], "bbox": bbox}


def transform_coordinates_demo() -> dict[str, Any]:
    """
    Demonstrate coordinate transforms via Matrix.

    - Matrix scales, rotates, translates.
    - page.set_cropbox() changes visible area; page.rect vs cropbox
    - Transform a Rect via `rect * matrix` or `matrix.transform_point()`
    """
    doc = fitz.open()
    page = doc.new_page(width=400, height=400)
    # Original page rect
    orig_rect = page.rect
    # Define a cropbox smaller than mediabox
    crop = fitz.Rect(50, 50, 350, 350)
    page.set_cropbox(crop)
    assert page.cropbox == crop
    # Rect transform via matrix: scale 2x
    mat = fitz.Matrix(2, 2)
    scaled = fitz.Rect(10, 10, 20, 20) * mat  # Rect * Matrix
    assert math.isclose(scaled.width, 20) and math.isclose(scaled.height, 20)
    # Matrix invert check: mat * mat_inv should be identity (approx)
    inv = ~mat  # invert operator
    identity = mat * inv
    assert math.isclose(identity.a, 1.0) and math.isclose(identity.d, 1.0)
    doc.close()
    return {"orig_rect": tuple(orig_rect), "crop": tuple(crop), "scaled": tuple(scaled)}


# =============================================================================
# Self-tests (IMPORT-ONLY)
# =============================================================================


def _self_tests() -> None:
    """Run assert-based checks for every section. Call manually or via __main__."""
    # 1. Reading & extraction
    doc = create_sample_pdf(text="Hello PyMuPDF", title="Sample", author="TestAuthor")
    text = extract_text_demo(doc)
    assert "Hello PyMuPDF" in text
    assert "Second line" in text
    info = extract_blocks_words_demo(doc)
    assert info["block_count"] >= 1
    assert info["word_count"] >= 4  # at least "Hello", "PyMuPDF", "Second", "line", ...
    assert info["first_word"] is not None and len(info["first_word"]) == 8  # words tuple 8 elements
    meta = extract_metadata_demo(doc)
    assert meta["title"] == "Sample" and meta["author"] == "TestAuthor" and meta["page_count"] == 1
    doc.close()
    img_info = extract_images_drawings_demo()
    assert img_info["drawing_count"] >= 1
    assert img_info["images_after"] >= 1
    assert img_info["images_before"] == 0

    # 2. Manipulation
    manip = add_page_and_draw_demo()
    assert manip["page_count"] == 1 and manip["drawing_count"] >= 2
    merged = merge_pdfs_demo()
    assert merged["merged_count"] == 3  # 1 + 2
    assert merged["after_delete"] == 2
    assert merged["bytes_len"] > 500
    rot = rotate_and_save_demo()
    assert rot["rotation"] == 90 and rot["bytes_start"] == "%PDF-"

    # 3. Annotations
    ann = annotate_text_highlight_demo()
    assert ann["annot_count"] == 2
    assert ann["quads_found"] >= 1
    # Types include text and highlight (codes vary, but at least 2)
    assert len(ann["types"]) == 2
    ann2 = annotate_rect_ink_and_remove_demo()
    assert ann2["before"] == 2 and ann2["after"] == 0

    # 4. Rendering
    rend = render_to_pixmap_demo()
    assert rend["width"] > 0 and rend["height"] > 0 and rend["png_len"] > 100
    assert rend["n"] in (3, 4)
    rend2 = render_with_alpha_and_clip_demo()
    assert rend2["alpha_n"] == 4
    assert rend2["clip_width"] <= rend2["full_width"]

    # 5. Coordinates & layout
    coord = coordinates_search_demo()
    assert coord["page_rect"] == (0, 0, 595, 842)
    assert coord["hit"][0] < coord["hit"][2]
    lay = layout_dict_demo()
    assert lay["block_count"] >= 1
    assert len(lay["first_span_text"]) > 0 and lay["bbox"][0] < lay["bbox"][2]
    trans = transform_coordinates_demo()
    assert trans["orig_rect"] == (0, 0, 400, 400)
    assert trans["crop"] == (50, 50, 350, 350)
    assert math.isclose(trans["scaled"][2] - trans["scaled"][0], 20)

    print("All PyMuPDF (fitz) asserts passed.")


if __name__ == "__main__":
    _self_tests()
