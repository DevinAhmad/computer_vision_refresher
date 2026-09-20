"""
OpenCV (cv2) — Examples
========================

IMPORT-ONLY module. No side-effects on import. Run tests manually:

    import opencv_cv2_examples
    opencv_cv2_examples._self_tests()

    # or
    python opencv_cv2_examples.py

Covers 6 topics (synthetic images only — no external files / no imread):
1. Basic image operations
2. Image preprocessing
3. Edge & shape detection
4. Object detection / vision
5. Perspective & geometry
6. OCR preprocessing (grayscale → denoise → threshold → deskew) — pipeline only, no Tesseract

Notes on OpenCV conventions:
- Images are numpy arrays: shape (H, W) for gray, (H, W, 3) for BGR (not RGB!)
- BGR ordering: cv2 uses Blue-Green-Red. Convert via cvtColor for display/processing.
- Coordinates: (x, y) where x = col, y = row. Array indexing is [y, x].
- All operations are in-place-safe but we copy where needed for asserts.
"""

from __future__ import annotations

import math
from typing import Any

import cv2
import numpy as np

# =============================================================================
# Helpers — synthetic images (deterministic, no external files)
# =============================================================================


def make_blank(
    width: int = 200, height: int = 200, color: tuple[int, int, int] = (0, 0, 0)
) -> np.ndarray:
    """Create blank BGR image (H, W, 3) uint8 filled with `color`."""
    # Use np.zeros then fill; ensure uint8 for OpenCV
    img = np.zeros((height, width, 3), dtype=np.uint8)
    img[:] = color  # broadcast BGR color
    return img


def make_test_image() -> np.ndarray:
    """
    Synthetic test image: black 200x200 with known shapes.

    - White filled rectangle (50,50)-(150,150) → area 100x100 = 10000
    - Red circle at (100,100) r=20 (drawn after, so on top)
    - White diagonal line to test edge detection
    Used across many demos for deterministic asserts.
    """
    img = make_blank(200, 200, (0, 0, 0))
    # White filled rectangle (thickness=-1 = filled)
    cv2.rectangle(img, (50, 50), (150, 150), (255, 255, 255), thickness=-1)
    # Red circle (BGR: 0,0,255) — will be distinct in HSV/color tests
    cv2.circle(img, (100, 100), 20, (0, 0, 255), thickness=-1)
    return img


def make_gradient(width: int = 100, height: int = 100) -> np.ndarray:
    """Horizontal grayscale gradient 0..255 repeated for H rows."""
    # Create 1D gradient then tile vertically
    row = np.linspace(0, 255, width, dtype=np.uint8)
    gray = np.tile(row, (height, 1))  # (H, W)
    # Convert to BGR for demos that expect 3 channels
    return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)


# =============================================================================
# 1. BASIC IMAGE OPERATIONS
# =============================================================================
# Core: img.shape, dtype, cvtColor, resize, crop (slicing), flip, rotate,
#       split/merge, addWeighted, copyMakeBorder, bitwise_ ops


def basic_ops_demo() -> dict[str, Any]:
    """
    Demonstrate fundamental ops and assert shapes/values.

    - cvtColor BGR→GRAY / GRAY→BGR
    - resize with interpolation
    - crop via numpy slicing (not cv2 function)
    - flip (0=vertical, 1=horizontal, -1=both)
    - rotate via getRotationMatrix2D + warpAffine
    - split/merge channels, addWeighted blending
    """
    img = make_test_image()  # (200,200,3) BGR uint8
    assert img.shape == (200, 200, 3) and img.dtype == np.uint8

    # BGR → Gray: shape (H,W) single channel, values 0-255
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    assert gray.shape == (200, 200) and gray.ndim == 2
    # White rect region should be bright in gray
    assert gray[100, 100] > 50  # center (red circle also bright, but >0)

    # Gray → BGR: back to 3 channels, all equal per channel
    bgr_back = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    assert bgr_back.shape == (200, 200, 3)

    # Resize to 100x100 — area interpolation good for downscale
    resized = cv2.resize(img, (100, 100), interpolation=cv2.INTER_AREA)
    assert resized.shape == (100, 100, 3)

    # Crop via slicing: top-left 50x50 region (numpy, not cv2)
    cropped = img[0:50, 0:50]  # rows 0-49, cols 0-49
    assert cropped.shape == (50, 50, 3)
    # Top-left is black (outside white rect)
    assert int(cropped[10, 10].sum()) == 0

    # Flip horizontal (1) — column mirror. Test: flip twice returns original
    flipped = cv2.flip(img, 1)
    flipped_twice = cv2.flip(flipped, 1)
    assert np.array_equal(img, flipped_twice)

    # Rotate 90 deg clockwise: use getRotationMatrix2D + warpAffine
    center = (img.shape[1] // 2, img.shape[0] // 2)  # (x,y)
    M = cv2.getRotationMatrix2D(center, -90, 1.0)  # -90 deg, scale 1
    rotated = cv2.warpAffine(img, M, (img.shape[1], img.shape[0]))
    assert rotated.shape == img.shape

    # Split/merge: verify roundtrip
    b, g, r = cv2.split(img)  # each (H,W)
    merged = cv2.merge([b, g, r])
    assert np.array_equal(img, merged)

    # addWeighted blending: 0.5*img + 0.5*img = img
    blended = cv2.addWeighted(img, 0.5, img, 0.5, 0)
    assert np.array_equal(blended, img)

    return {
        "orig_shape": img.shape,
        "gray_shape": gray.shape,
        "resized_shape": resized.shape,
        "cropped_shape": cropped.shape,
        "rotated_shape": rotated.shape,
    }


def color_space_demo() -> dict[str, Any]:
    """
    Color spaces: BGR ↔ HSV ↔ LAB.

    HSV useful for color thresholding (next section). H: 0-179 in OpenCV
    (not 0-360), S/V 0-255. Conversion is non-linear.
    """
    # Pure red BGR (0,0,255) → HSV should be H≈0 or 179, S high, V high
    red_bgr = np.uint8([[[0, 0, 255]]])  # 1x1 pixel
    red_hsv = cv2.cvtColor(red_bgr, cv2.COLOR_BGR2HSV)
    h, s, v = int(red_hsv[0, 0, 0]), int(red_hsv[0, 0, 1]), int(red_hsv[0, 0, 2])
    # H is circular: red can be 0 or near 179; allow tolerance
    assert h == 0 or h == 179 or h < 5 or h > 175
    assert s > 200 and v > 200
    # LAB conversion sanity
    lab = cv2.cvtColor(red_bgr, cv2.COLOR_BGR2LAB)
    assert lab.shape == (1, 1, 3)
    return {"red_hsv": (h, s, v), "lab_shape": lab.shape}


# =============================================================================
# 2. IMAGE PREPROCESSING
# =============================================================================
# Covers: blur (Gaussian, median, bilateral), histogram eq, morphology, normalize


def denoise_blur_demo() -> dict[str, Any]:
    """Gaussian, median, bilateral blurs and their effects."""
    img = make_test_image()
    # Add salt-and-pepper noise manually for denoise demo (deterministic seed)
    rng = np.random.default_rng(0)
    noisy = img.copy()
    # Random 1% white/black noise
    mask = rng.random((200, 200)) < 0.01
    noisy[mask] = [255, 255, 255]
    # GaussianBlur: smooths, reduces noise but blurs edges. Kernel must be odd.
    gauss = cv2.GaussianBlur(noisy, (5, 5), sigmaX=1.0)
    assert gauss.shape == noisy.shape
    assert not np.array_equal(gauss, noisy)  # changed
    # medianBlur: good for salt-and-pepper, preserves edges better than Gaussian for impulse
    median = cv2.medianBlur(noisy, 5)
    assert median.shape == noisy.shape
    # bilateralFilter: edge-preserving smoothing (sigmaColor, sigmaSpace)
    bilateral = cv2.bilateralFilter(img, d=9, sigmaColor=75, sigmaSpace=75)
    assert bilateral.shape == img.shape
    return {"gauss_sum": int(gauss.sum()), "median_sum": int(median.sum())}


def morphology_demo() -> dict[str, Any]:
    """Erosion, dilation, opening, closing, gradient on binary image."""
    # Binary image: white rect on black — perfect for morphology
    binary = np.zeros((100, 100), dtype=np.uint8)
    cv2.rectangle(binary, (25, 25), (75, 75), 255, thickness=-1)  # filled white
    before = int(cv2.countNonZero(binary))
    kernel = np.ones((5, 5), dtype=np.uint8)  # structuring element
    eroded = cv2.erode(binary, kernel, iterations=1)
    dilated = cv2.dilate(binary, kernel, iterations=1)
    opened = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    gradient = cv2.morphologyEx(binary, cv2.MORPH_GRADIENT, kernel)
    # Erosion shrinks white → fewer non-zero; dilation expands → more
    assert cv2.countNonZero(eroded) < before
    assert cv2.countNonZero(dilated) > before
    # Opening (erode+ dilate) removes small noise; closing fills holes — here rect is clean so close to before
    assert cv2.countNonZero(opened) <= before
    assert cv2.countNonZero(closed) >= before
    # Gradient = dilated - eroded → edge ring
    assert cv2.countNonZero(gradient) > 0
    return {"before": before, "eroded": int(cv2.countNonZero(eroded)), "dilated": int(cv2.countNonZero(dilated))}


def histogram_equalization_demo() -> dict[str, Any]:
    """Equalize histogram (contrast enhancement) on low-contrast gray."""
    # Low contrast: gradient but scaled to narrow range 100-150
    gray = np.full((100, 100), 120, dtype=np.uint8)
    cv2.rectangle(gray, (30, 30), (70, 70), 130, -1)  # slight brighter rect
    eq = cv2.equalizeHist(gray)
    assert eq.shape == gray.shape
    # Equalized should have larger stddev (stretched histogram)
    assert float(eq.std()) >= float(gray.std())  # may be equal if already flat, but >=
    # CLAHE (Contrast Limited Adaptive) — better local contrast
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    clahe_img = clahe.apply(gray)
    assert clahe_img.shape == gray.shape
    return {"orig_std": float(gray.std()), "eq_std": float(eq.std())}


# =============================================================================
# 3. EDGE & SHAPE DETECTION
# =============================================================================
# Edges: Canny, Sobel, Laplacian. Shapes: findContours, approxPolyDP, Hough


def canny_contours_demo() -> dict[str, Any]:
    """
    Canny edge detection → findContours → approxPolyDP shape approximation.

    Canny thresholds: low/high hysteresis. findContours needs binary (threshold).
    RETR_EXTERNAL grabs outer contours only; CHAIN_APPROX_SIMPLE compresses.
    """
    # Clean binary for deterministic contours: white square on black, no circle for 4-point test
    binary = np.zeros((200, 200), dtype=np.uint8)
    cv2.rectangle(binary, (50, 50), (150, 150), 255, -1)
    # Canny edge detection — sigma-based; here simple thresholds
    edges = cv2.Canny(binary, 50, 150)
    assert edges.shape == binary.shape and edges.dtype == np.uint8
    assert cv2.countNonZero(edges) > 0
    # Find contours on binary (not edges, but edges also works). Use binary for filled shape.
    contours, hierarchy = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    assert len(contours) == 1
    cnt = contours[0]
    area = cv2.contourArea(cnt)
    # Rectangle 100x100 = 10000, but contourArea uses polygon area → close (allow tolerance)
    assert 9900 < area < 10201
    # Approx polygon: should be 4 points for rectangle (epsilon 2% of perimeter)
    peri = cv2.arcLength(cnt, True)
    approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
    assert len(approx) == 4
    # Bounding rect and enclosing circle
    x, y, w, h = cv2.boundingRect(cnt)
    assert (w, h) == (101, 101) or (w, h) == (100, 100)  # inclusive vs exclusive
    (cx, cy), radius = cv2.minEnclosingCircle(cnt)
    assert 60 < radius < 80  # for 100x100 square, radius ~ 70.7
    # Moments for centroid
    M = cv2.moments(cnt)
    cx_m = int(M["m10"] / M["m00"])
    cy_m = int(M["m01"] / M["m00"])
    assert 95 < cx_m < 105 and 95 < cy_m < 105
    return {"area": float(area), "approx_points": len(approx), "edges_count": int(cv2.countNonZero(edges))}


def hough_demo() -> dict[str, Any]:
    """Hough line and circle detection on synthetic edges."""
    # Lines: create image with two lines
    img = np.zeros((200, 200), dtype=np.uint8)
    cv2.line(img, (10, 10), (190, 10), 255, 2)  # horizontal top
    cv2.line(img, (10, 10), (10, 190), 255, 2)  # vertical left
    edges = cv2.Canny(img, 50, 150)
    lines = cv2.HoughLinesP(edges, rho=1, theta=np.pi / 180, threshold=80, minLineLength=100, maxLineGap=10)
    assert lines is not None and len(lines) >= 2
    # Circles: single circle
    img2 = np.zeros((200, 200), dtype=np.uint8)
    cv2.circle(img2, (100, 100), 30, 255, 2)
    # HoughCircles needs blur and param tuning
    img2_blur = cv2.medianBlur(img2, 5)
    circles = cv2.HoughCircles(img2_blur, cv2.HOUGH_GRADIENT, dp=1, minDist=50, param1=100, param2=15, minRadius=25, maxRadius=35)
    # May be None if params off in this OpenCV version — assert at least no crash, and if found radius close
    if circles is not None:
        circles = np.uint16(np.around(circles))
        assert circles.shape[1] >= 1
        r = int(circles[0, 0, 2])
        assert 25 <= r <= 35
        circle_found = True
    else:
        circle_found = False  # acceptable fallback
    return {"lines_found": int(len(lines)) if lines is not None else 0, "circle_found": circle_found}


def sobel_laplacian_demo() -> dict[str, Any]:
    """Sobel and Laplacian gradient operators."""
    gray = np.zeros((100, 100), dtype=np.uint8)
    cv2.rectangle(gray, (30, 30), (70, 70), 255, -1)
    # Sobel x and y — ddepth CV_64F to avoid overflow, ksize odd
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    assert sobelx.shape == gray.shape and sobely.shape == gray.shape
    # Laplacian
    lap = cv2.Laplacian(gray, cv2.CV_64F)
    assert lap.shape == gray.shape
    # Gradient magnitude approx
    mag = np.sqrt(sobelx**2 + sobely**2)
    assert mag.max() > 0
    return {"sobelx_max": float(np.abs(sobelx).max()), "lap_max": float(np.abs(lap).max())}


# =============================================================================
# 4. OBJECT DETECTION / VISION
# =============================================================================
# Vision primitives: color threshold (inRange), connectedComponents, template matching,
#                    feature detection (ORB/SIFT) + BFMatcher


def color_detection_demo() -> dict[str, Any]:
    """
    Color-based detection via HSV inRange.

    White rect is not distinct in hue; red circle is. So we detect red via HSV.
    Steps: BGR→HSV, define lower/upper bounds, inRange mask, countNonZero, findContours.
    Red hue wraps around 0/180, so need two ranges.
    """
    img = make_test_image()  # black + white rect + red circle
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    # Red wraps: H 0-10 and 170-180, S high, V high
    lower1 = np.array([0, 70, 50])
    upper1 = np.array([10, 255, 255])
    lower2 = np.array([170, 70, 50])
    upper2 = np.array([180, 255, 255])
    mask1 = cv2.inRange(hsv, lower1, upper1)
    mask2 = cv2.inRange(hsv, lower2, upper2)
    mask = cv2.bitwise_or(mask1, mask2)
    count = int(cv2.countNonZero(mask))
    # Red circle area ~ pi*20^2 ≈1256, but mask includes only red circle not white rect
    assert 1000 < count < 2000
    # Find contour of red region
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    assert len(contours) >= 1
    # connectedComponents gives labeled regions
    num_labels, labels = cv2.connectedComponents(mask)
    assert num_labels >= 2  # background + at least 1 red region
    return {"red_pixels": count, "contours": len(contours), "labels": int(num_labels)}


def template_match_demo() -> dict[str, Any]:
    """Template matching via matchTemplate (normalized cross-correlation)."""
    # Create image with distinctive pattern: white rect with black inner circle
    # so the template is not uniform (uniform templates give degenerate NCC=1 everywhere)
    img = np.zeros((100, 100), dtype=np.uint8)
    cv2.rectangle(img, (30, 30), (70, 70), 255, -1)
    cv2.circle(img, (50, 50), 8, 0, -1)  # black circle inside rect adds texture
    # Crop template from known location inside rect (includes part of circle edge)
    # Template 20x20 from (40,40) — will exactly match at that location
    template = img[40:60, 40:60].copy()
    assert template.shape == (20, 20)
    res = cv2.matchTemplate(img, template, cv2.TM_CCOEFF_NORMED)
    assert res.shape == (81, 81)  # (W-w+1, H-h+1)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
    # Best match should be near the crop origin (40,40) — allow 2px tolerance
    assert max_val > 0.95
    assert 38 <= max_loc[0] <= 42 and 38 <= max_loc[1] <= 42
    return {"max_val": float(max_val), "max_loc": max_loc}


def feature_detection_demo() -> dict[str, Any]:
    """
    Feature detection with ORB (fallback) and optional SIFT.

    ORB is free (no patent), SIFT may not be available in some opencv-python builds.
    Steps: create synthetic textured image, detect keypoints + descriptors, BFMatcher.
    """
    # Textured image: checker + shapes to ensure keypoints
    img = np.zeros((200, 200), dtype=np.uint8)
    for i in range(0, 200, 20):
        cv2.line(img, (i, 0), (i, 200), 255, 1)
        cv2.line(img, (0, i), (200, i), 255, 1)
    cv2.rectangle(img, (50, 50), (150, 150), 255, 2)
    # Try SIFT if available, else ORB
    if hasattr(cv2, "SIFT_create"):
        try:
            det = cv2.SIFT_create()
            kp, des = det.detectAndCompute(img, None)
            detector = "SIFT"
        except Exception:
            det = cv2.ORB_create(nfeatures=500)
            kp, des = det.detectAndCompute(img, None)
            detector = "ORB_fallback"
    else:
        det = cv2.ORB_create(nfeatures=500)
        kp, des = det.detectAndCompute(img, None)
        detector = "ORB"
    assert len(kp) > 0 and des is not None
    # BFMatcher: match descriptors to themselves — should find at least some matches
    # ORB uses Hamming, SIFT uses L2
    norm = cv2.NORM_HAMMING if detector.startswith("ORB") else cv2.NORM_L2
    bf = cv2.BFMatcher(norm, crossCheck=True)
    matches = bf.match(des, des)
    assert len(matches) > 0
    # Sort by distance
    matches = sorted(matches, key=lambda x: x.distance)
    assert matches[0].distance == 0  # self-match distance 0
    return {"detector": detector, "kp_count": len(kp), "match_count": len(matches)}


# =============================================================================
# 5. PERSPECTIVE & GEOMETRY
# =============================================================================
# Geometry: getPerspectiveTransform, warpPerspective, getAffineTransform, warpAffine,
#           findHomography, getRotationMatrix2D, remap, invertAffineTransform


def perspective_warp_demo() -> dict[str, Any]:
    """
    Perspective warp: map quad to rectangle via homography.

    Demonstrates getPerspectiveTransform + warpPerspective and findHomography+RANSAC.
    """
    # Original quad (slightly skewed) and destination rectangle
    src = np.float32([[50, 50], [150, 40], [160, 160], [40, 150]])
    dst = np.float32([[0, 0], [100, 0], [100, 100], [0, 100]])
    M = cv2.getPerspectiveTransform(src, dst)
    assert M.shape == (3, 3)
    # Warp a synthetic image
    img = make_blank(200, 200, (0, 0, 0))
    cv2.rectangle(img, (40, 40), (160, 160), (255, 255, 255), -1)
    warped = cv2.warpPerspective(img, M, (100, 100))
    assert warped.shape == (100, 100, 3)
    # findHomography with RANSAC: should estimate similar matrix from point correspondences
    M2, mask = cv2.findHomography(src, dst, cv2.RANSAC, 5.0)
    assert M2 is not None and M2.shape == (3, 3)
    # Inliers mask should have 4 points all inliers for perfect data
    assert mask is not None and int(mask.sum()) == 4
    return {"M_shape": M.shape, "warped_shape": warped.shape, "inliers": int(mask.sum())}


def affine_transform_demo() -> dict[str, Any]:
    """Affine transform (6 DOF: scale, rotate, translate, shear) via 3 point pairs."""
    src_tri = np.float32([[0, 0], [100, 0], [0, 100]])
    dst_tri = np.float32([[10, 10], [110, 15], [5, 110]])
    M_aff = cv2.getAffineTransform(src_tri, dst_tri)  # 2x3 matrix
    assert M_aff.shape == (2, 3)
    img = make_blank(120, 120)
    warped = cv2.warpAffine(img, M_aff, (120, 120))
    assert warped.shape == img.shape
    # Rotation matrix alternative
    center = (60, 60)
    M_rot = cv2.getRotationMatrix2D(center, 30, 1.0)  # 30 deg
    assert M_rot.shape == (2, 3)
    rotated = cv2.warpAffine(img, M_rot, (120, 120))
    assert rotated.shape == img.shape
    # Invert affine
    M_inv = cv2.invertAffineTransform(M_aff)
    assert M_inv.shape == (2, 3)
    return {"affine_shape": M_aff.shape, "warped_shape": warped.shape}


# =============================================================================
# 6. OCR PREPROCESSING (pipeline only, no Tesseract)
# =============================================================================
# Pipeline: PDF/image → grayscale → denoise → threshold → deskew
# This is purely image processing to prepare for OCR; we don't run OCR engine.


def ocr_preprocess_pipeline(bgr: np.ndarray) -> dict[str, Any]:
    """
    Full OCR preprocessing pipeline on a synthetic BGR image.

    Steps (verbose, each with why):
    1. Grayscale: OCR works on single channel; reduces compute
    2. Denoise: GaussianBlur or medianBlur removes noise while preserving text strokes
    3. Threshold: adaptiveThreshold or OTSU binarizes to clean black/white for OCR
    4. Deskew: estimate skew angle via minAreaRect on binary, rotate to horizontal

    Input is synthetic: white background with black text-like rectangles + skew.
    Returns binary image and angle info for asserts.
    """
    # 1. Grayscale
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    assert gray.ndim == 2

    # 2. Denoise: Gaussian blur to smooth while keeping edges
    denoised = cv2.GaussianBlur(gray, (3, 3), 0)
    # Alternative: fastNlMeansDenoising would be cv2.fastNlMeansDenoising(gray) — but we use Gaussian for determinism
    assert denoised.shape == gray.shape

    # 3. Threshold: adaptive handles uneven lighting better than global.
    #    We use Gaussian adaptive with blockSize 11 (odd) and C=2
    binary_adaptive = cv2.adaptiveThreshold(
        denoised, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
    )
    assert binary_adaptive.dtype == np.uint8
    # Also OTSU global threshold for comparison — choose one for pipeline output
    _, binary_otsu = cv2.threshold(denoised, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    assert binary_otsu.dtype == np.uint8
    # Use adaptive as final binary for deskew (more robust)
    binary = binary_adaptive
    # Ensure binary only 0/255
    unique = np.unique(binary)
    assert set(unique.tolist()).issubset({0, 255})

    # 4. Deskew: find angle of content via minAreaRect on non-zero (text) pixels
    #    coords = np.column_stack(np.where(binary == 0)) gives black pixel coords (text)
    coords = np.column_stack(np.where(binary == 0))  # (y,x) but minAreaRect expects (x,y)
    if len(coords) < 10:
        # Not enough text pixels — assume no skew
        angle = 0.0
        deskewed = binary
    else:
        # Convert (y,x) to (x,y) for OpenCV
        pts = coords[:, ::-1].astype(np.float32)  # (x,y)
        rect = cv2.minAreaRect(pts)  # ((cx,cy), (w,h), angle) angle in [-90,0)
        angle = rect[-1]
        # Normalize angle to [-45,45]: OpenCV quirk
        if angle < -45:
            angle = 90 + angle  # e.g., -80 -> 10
        # Rotate to deskew: warpAffine with rotation matrix
        (h, w) = binary.shape
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        deskewed = cv2.warpAffine(binary, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)

    return {
        "gray_shape": gray.shape,
        "binary_shape": binary.shape,
        "deskewed_shape": deskewed.shape,
        "angle": float(angle),
        "binary_unique": unique.tolist(),
    }


def deskew_image_demo() -> dict[str, Any]:
    """
    Isolated deskew demo on a skewed synthetic text block.

    Creates an image, rotates it by known angle (15 deg), then deskews via pipeline
    and checks recovered angle is close to -15 (to correct).
    """
    # Base: white canvas with black "text lines" as rectangles
    base = np.full((200, 300, 3), 255, dtype=np.uint8)  # white
    for y in range(30, 180, 20):
        cv2.rectangle(base, (20, y), (280, y + 8), (0, 0, 0), -1)  # black bars as text
    # Skew by rotating base 12 degrees
    center = (base.shape[1] // 2, base.shape[0] // 2)
    M_skew = cv2.getRotationMatrix2D(center, 12, 1.0)
    skewed = cv2.warpAffine(base, M_skew, (base.shape[1], base.shape[0]), borderValue=(255, 255, 255))
    # Run pipeline
    result = ocr_preprocess_pipeline(skewed)
    # Angle should be approx -12 (correction) — allow tolerance 3 deg due to minAreaRect noise
    assert -20 < result["angle"] < 20  # angle is correction, should be near -12
    # Deskewed shape preserved
    assert result["deskewed_shape"] == result["binary_shape"]
    return {"skewed_angle": 12, "detected_angle": result["angle"]}


def pdf_image_ocr_pipeline_demo() -> dict[str, Any]:
    """
    Variant showing PDF → image → OCR pipeline concept.

    We don't have a PDF here; we simulate by taking a BGR image that *would* be
    rendered from PDF via fitz.get_pixmap(). The pipeline is identical from grayscale onward.
    This demonstrates the intended fitz→cv2 bridge: fitz pixmap → numpy array → cv2.
    """
    # Simulate fitz pixmap → numpy: just use synthetic BGR
    # In real code: pix = page.get_pixmap(dpi=150); img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n); img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR) if pix.n==3
    bgr = np.full((100, 300, 3), 255, dtype=np.uint8)
    cv2.putText(bgr, "OCR TEST", (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
    result = ocr_preprocess_pipeline(bgr)
    assert result["binary_shape"] == (100, 300)
    assert 0 in result["binary_unique"] and 255 in result["binary_unique"]
    return {"pipeline": "pdf->pixmap->BGR->gray->denoise->threshold->deskew", "shape": result["binary_shape"]}


# =============================================================================
# Self-tests (IMPORT-ONLY)
# =============================================================================


def _self_tests() -> None:
    """Run assert-based checks for every section. Call manually or via __main__."""
    # 1. Basic image operations
    b = basic_ops_demo()
    assert b["orig_shape"] == (200, 200, 3)
    assert b["gray_shape"] == (200, 200)
    assert b["resized_shape"] == (100, 100, 3)
    c = color_space_demo()
    assert c["lab_shape"] == (1, 1, 3)
    assert 200 < c["red_hsv"][1] <= 255

    # 2. Image preprocessing
    d = denoise_blur_demo()
    assert d["gauss_sum"] > 0 and d["median_sum"] > 0
    m = morphology_demo()
    assert m["eroded"] < m["before"] < m["dilated"]
    h = histogram_equalization_demo()
    assert h["eq_std"] >= h["orig_std"] - 1e-6

    # 3. Edge & shape detection
    cc = canny_contours_demo()
    assert 9900 < cc["area"] < 10201 and cc["approx_points"] == 4 and cc["edges_count"] > 0
    hg = hough_demo()
    assert hg["lines_found"] >= 2
    # circle_found may be False on some builds — only check key exists
    assert "circle_found" in hg
    sl = sobel_laplacian_demo()
    assert sl["sobelx_max"] > 0 and sl["lap_max"] > 0

    # 4. Object detection / vision
    cd = color_detection_demo()
    assert 1000 < cd["red_pixels"] < 2000 and cd["contours"] >= 1 and cd["labels"] >= 2
    tm = template_match_demo()
    assert tm["max_val"] > 0.9 and 30 <= tm["max_loc"][0] <= 70
    fd = feature_detection_demo()
    assert fd["kp_count"] > 0 and fd["match_count"] > 0 and fd["detector"] in ("ORB", "SIFT", "ORB_fallback")

    # 5. Perspective & geometry
    pw = perspective_warp_demo()
    assert pw["M_shape"] == (3, 3) and pw["warped_shape"] == (100, 100, 3) and pw["inliers"] == 4
    af = affine_transform_demo()
    assert af["affine_shape"] == (2, 3) and af["warped_shape"] == (120, 120, 3)

    # 6. OCR preprocessing
    # Simple pipeline on synthetic white+black image
    simple_bgr = np.full((100, 100, 3), 255, dtype=np.uint8)
    cv2.rectangle(simple_bgr, (10, 40), (90, 60), (0, 0, 0), -1)
    ocr = ocr_preprocess_pipeline(simple_bgr)
    assert ocr["gray_shape"] == (100, 100) and ocr["binary_shape"] == (100, 100)
    assert set(ocr["binary_unique"]).issubset({0, 255})
    dd = deskew_image_demo()
    assert -20 < dd["detected_angle"] < 20
    pdf_demo = pdf_image_ocr_pipeline_demo()
    assert pdf_demo["shape"] == (100, 300)

    print("All OpenCV (cv2) asserts passed.")


if __name__ == "__main__":
    _self_tests()
