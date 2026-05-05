#!/usr/bin/env python
# coding: utf-8

# In[12]:


get_ipython().system('pip install opencv-python numpy')

#python apple_quality_checker.py --image path/to/apple.jpg
#python apple_quality_checker.py --camera          # live webcam feed
#python apple_quality_checker.py --folder path/to/folder/  # batch mode

import cv2
import numpy as np
import argparse
import os
import sys
from pathlib import Path


# In[13]:


# ─────────────────────────── Quality Thresholds ───────────────────────────
THRESHOLDS = {
    "min_roundness":      0.70,   # 0–1, how circular the apple is
    "max_defect_ratio":   0.08,   # fraction of apple area that may be defective
    "min_color_score":    0.50,   # uniformity/saturation of expected apple color
    "min_size_px":       4000,    # minimum contour area in pixels
}


# In[11]:


# ─────────────────────────── Core Detection ───────────────────────────────

def segment_apple(image: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Segment the apple from the background.
    Returns (mask, largest_contour).
    """
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Red apples span two hue ranges in HSV
    lower_red1 = np.array([0,   50,  50])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([160, 50,  50])
    upper_red2 = np.array([180, 255, 255])

    # Green apples
    lower_green = np.array([35,  40,  40])
    upper_green = np.array([90, 255, 255])

    # Yellow / golden
    lower_yellow = np.array([15,  50,  50])
    upper_yellow = np.array([35, 255, 255])

    mask_r1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask_r2 = cv2.inRange(hsv, lower_red2, upper_red2)
    mask_g  = cv2.inRange(hsv, lower_green, upper_green)
    mask_y  = cv2.inRange(hsv, lower_yellow, upper_yellow)

    combined = cv2.bitwise_or(mask_r1, mask_r2)
    combined = cv2.bitwise_or(combined, mask_g)
    combined = cv2.bitwise_or(combined, mask_y)

    # Morphological cleanup
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    combined = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, kernel, iterations=3)
    combined = cv2.morphologyEx(combined, cv2.MORPH_OPEN,  kernel, iterations=2)

    contours, _ = cv2.findContours(combined, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return combined, None

    largest = max(contours, key=cv2.contourArea)
    mask = np.zeros_like(combined)
    cv2.drawContours(mask, [largest], -1, 255, cv2.FILLED)
    return mask, largest


def analyze_roundness(contour) -> float:
    """Circularity = 4π·Area / Perimeter²  (1.0 = perfect circle)."""
    if contour is None:
        return 0.0
    area = cv2.contourArea(contour)
    perimeter = cv2.arcLength(contour, True)
    if perimeter == 0:
        return 0.0
    return min(1.0, (4 * np.pi * area) / (perimeter ** 2))


def analyze_color(image: np.ndarray, mask: np.ndarray) -> float:
    """
    Score color quality based on HSV saturation and expected hue coverage.
    Returns 0–1 (higher = better / more vibrant and uniform).
    """
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    apple_pixels = hsv[mask == 255]
    if len(apple_pixels) == 0:
        return 0.0

    saturation = apple_pixels[:, 1].mean() / 255.0   # 0–1
    value      = apple_pixels[:, 2].mean() / 255.0   # brightness

    # Reward high saturation + good brightness
    score = 0.6 * saturation + 0.4 * value
    return float(np.clip(score, 0, 1))


def detect_defects(image: np.ndarray, mask: np.ndarray) -> tuple[float, np.ndarray]:
    """
    Detect bruises, dark spots, and blemishes on the apple surface.
    Returns (defect_ratio, defect_mask).
    """
    apple_area = int(np.sum(mask == 255))
    if apple_area == 0:
        return 1.0, np.zeros_like(mask)

    # Work only within the apple region
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l_channel = lab[:, :, 0]

    # Apply mask
    l_masked = cv2.bitwise_and(l_channel, l_channel, mask=mask)

    # Compute local mean inside the apple
    apple_mean = float(l_channel[mask == 255].mean())

    # Pixels significantly darker than mean → defect
    dark_threshold = apple_mean * 0.65
    defect_mask = np.where((l_masked < dark_threshold) & (mask == 255),
                           np.uint8(255), np.uint8(0))

    # Clean tiny noise
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    defect_mask = cv2.morphologyEx(defect_mask, cv2.MORPH_OPEN, k, iterations=2)

    defect_area = int(np.sum(defect_mask == 255))
    ratio = defect_area / apple_area
    return float(ratio), defect_mask


def compute_quality_score(roundness: float,
                           color_score: float,
                           defect_ratio: float) -> tuple[float, str, tuple]:
    """
    Combine individual metrics into an overall quality score 0–100
    and a grade label.
    """
    roundness_score = roundness * 100
    color_pts       = color_score * 100
    defect_pts      = max(0, (1 - defect_ratio / THRESHOLDS["max_defect_ratio"]) * 100)
    defect_pts      = min(defect_pts, 100)

    overall = 0.35 * roundness_score + 0.30 * color_pts + 0.35 * defect_pts

    if overall >= 80:
        grade, color = "GRADE A  ✅  Excellent", (0, 200, 80)
    elif overall >= 60:
        grade, color = "GRADE B  ⚠️  Acceptable", (0, 180, 220)
    elif overall >= 40:
        grade, color = "GRADE C  ⚠️  Poor",       (0, 140, 255)
    else:
        grade, color = "REJECTED ❌  Defective",  (0, 50, 220)

    return overall, grade, color


# In[4]:


# ─────────────────────────── Visualisation ────────────────────────────────

def draw_overlay(image: np.ndarray,
                 contour,
                 defect_mask: np.ndarray,
                 roundness: float,
                 color_score: float,
                 defect_ratio: float,
                 overall: float,
                 grade: str,
                 grade_color: tuple) -> np.ndarray:
    """Render all annotations on a copy of the image."""
    out = image.copy()

    # Highlight defects in red
    defect_overlay = out.copy()
    defect_overlay[defect_mask == 255] = (0, 0, 200)
    cv2.addWeighted(defect_overlay, 0.45, out, 0.55, 0, out)

    # Draw apple contour
    if contour is not None:
        cv2.drawContours(out, [contour], -1, grade_color, 3)

    # HUD panel
    panel_h, panel_w = 170, 380
    panel = np.zeros((panel_h, panel_w, 3), dtype=np.uint8)
    panel[:] = (20, 20, 20)

    font  = cv2.FONT_HERSHEY_DUPLEX
    font2 = cv2.FONT_HERSHEY_SIMPLEX

    cv2.putText(panel, "APPLE QC REPORT",       (10, 28),  font2, 0.65, (180, 180, 180), 1)
    cv2.putText(panel, f"Overall:  {overall:.1f}/100", (10, 58),  font, 0.6,  (255, 255, 255), 1)
    cv2.putText(panel, f"Shape:    {roundness*100:.1f}",    (10, 86),  font2, 0.55, (180, 220, 255), 1)
    cv2.putText(panel, f"Color:    {color_score*100:.1f}",  (10, 110), font2, 0.55, (180, 220, 255), 1)
    cv2.putText(panel, f"Defects:  {defect_ratio*100:.1f}%", (10, 134), font2, 0.55, (180, 220, 255), 1)

    # Grade label strip
    label_strip = np.zeros((36, panel_w, 3), dtype=np.uint8)
    label_strip[:] = grade_color
    grade_text = grade.split("  ")[0]  # short label
    cv2.putText(label_strip, grade_text, (10, 26), font, 0.75, (255, 255, 255), 2)
    panel[134:, :] = label_strip

    # Paste panel onto image (top-left)
    h, w = out.shape[:2]
    px, py = 10, 10
    if h > panel_h + py and w > panel_w + px:
        out[py:py+panel_h, px:px+panel_w] = panel

    return out


# In[5]:


# ─────────────────────────── Main Pipeline ────────────────────────────────

def process_image(image: np.ndarray, show: bool = True) -> dict:
    """
    Run the full quality-check pipeline on a single BGR image.
    Returns a dict with all metrics and the annotated image.
    """
    if image is None or image.size == 0:
        raise ValueError("Empty or invalid image.")

    mask, contour = segment_apple(image)

    if contour is None or cv2.contourArea(contour) < THRESHOLDS["min_size_px"]:
        result = {
            "grade": "NO APPLE DETECTED",
            "overall": 0,
            "roundness": 0,
            "color_score": 0,
            "defect_ratio": 1,
            "annotated": image.copy(),
        }
        if show:
            cv2.putText(result["annotated"], "No apple detected",
                        (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 200), 3)
        return result

    roundness    = analyze_roundness(contour)
    color_score  = analyze_color(image, mask)
    defect_ratio, defect_mask = detect_defects(image, mask)
    overall, grade, grade_color = compute_quality_score(roundness, color_score, defect_ratio)

    annotated = draw_overlay(image, contour, defect_mask,
                             roundness, color_score, defect_ratio,
                             overall, grade, grade_color)

    if show:
        cv2.imshow("Apple Quality Checker", annotated)

    return {
        "grade": grade,
        "overall": round(overall, 2),
        "roundness": round(roundness, 3),
        "color_score": round(color_score, 3),
        "defect_ratio": round(defect_ratio, 4),
        "annotated": annotated,
    }


def print_report(result: dict, source: str = "") -> None:
    sep = "─" * 44
    print(f"\n{sep}")
    if source:
        print(f"  Source : {source}")
    print(f"  Grade  : {result['grade']}")
    print(f"  Overall: {result['overall']}/100")
    print(f"  Shape  : {result['roundness']*100:.1f}/100")
    print(f"  Color  : {result['color_score']*100:.1f}/100")
    print(f"  Defects: {result['defect_ratio']*100:.2f}%")
    print(sep)


# In[6]:


# ─────────────────────────── Entry Points ─────────────────────────────────

def run_image_mode(path: str) -> None:
    img = cv2.imread(path)
    if img is None:
        sys.exit(f"[ERROR] Cannot open image: {path}")
    result = process_image(img, show=True)
    print_report(result, source=path)
    print("Press any key to exit…")
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def run_folder_mode(folder: str) -> None:
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    paths = [p for p in Path(folder).iterdir() if p.suffix.lower() in exts]
    if not paths:
        sys.exit(f"[ERROR] No images found in {folder}")

    grades = {"GRADE A": 0, "GRADE B": 0, "GRADE C": 0, "REJECTED": 0, "NO APPLE": 0}

    for p in sorted(paths):
        img = cv2.imread(str(p))
        if img is None:
            continue
        result = process_image(img, show=False)
        print_report(result, source=p.name)

        g = result["grade"]
        if   "GRADE A"  in g: grades["GRADE A"]  += 1
        elif "GRADE B"  in g: grades["GRADE B"]  += 1
        elif "GRADE C"  in g: grades["GRADE C"]  += 1
        elif "REJECTED" in g: grades["REJECTED"] += 1
        else:                  grades["NO APPLE"] += 1

    total = sum(grades.values())
    print("\n══════ Batch Summary ══════")
    for k, v in grades.items():
        bar = "█" * v + "░" * (total - v)
        print(f"  {k:<10}: {v:>3}  {bar}")
    print(f"  {'TOTAL':<10}: {total:>3}")
    print("═══════════════════════════\n")


def run_camera_mode(camera_index: int = 0) -> None:
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        sys.exit(f"[ERROR] Cannot open camera index {camera_index}")

    print("Live Apple QC  –  Press 'q' to quit, 's' to save snapshot")
    frame_n = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Process every 3rd frame to keep UI responsive
        frame_n += 1
        if frame_n % 3 == 0:
            result = process_image(frame, show=False)
            annotated = result["annotated"]
        else:
            annotated = frame

        cv2.imshow("Apple Quality Checker — Live", annotated)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        if key == ord("s"):
            fname = f"snapshot_{frame_n:04d}.jpg"
            cv2.imwrite(fname, annotated)
            print(f"Saved {fname}")
            print_report(result)

    cap.release()
    cv2.destroyAllWindows()


# In[7]:


# ─────────────────────────── CLI ──────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Apple Quality Checker — Computer Vision QC Tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    group = p.add_mutually_exclusive_group(required=True)
    group.add_argument("--image",  metavar="PATH",  help="Path to a single apple image")
    group.add_argument("--folder", metavar="PATH",  help="Folder with multiple apple images (batch mode)")
    group.add_argument("--camera", metavar="INDEX", nargs="?", const=0, type=int,
                       help="Live webcam feed (default index 0)")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()

    if args.image:
        run_image_mode(args.image)
    elif args.folder:
        run_folder_mode(args.folder)
    else:
        run_camera_mode(args.camera)


# In[ ]:




