"""
Apple Quality Checker — Streamlit App
======================================
Run with:
    streamlit run apple_quality_app.py

Requires:
    pip install streamlit opencv-python numpy
Place apple_qc_model.pkl in the same directory as this script.
"""

import streamlit as st
import cv2
import numpy as np
import pickle
import time
from pathlib import Path
from typing import Tuple

# ─────────────────────────────────────────────
#  Page config
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Apple QC Inspector",
    page_icon="🍎",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
#  Custom CSS — dark industrial aesthetic
# ─────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Share+Tech+Mono&family=Barlow+Condensed:wght@300;600;800&display=swap');

html, body, [class*="css"] {
    font-family: 'Barlow Condensed', sans-serif;
}

.stApp {
    background: #0d0d0f;
    color: #e8e8e0;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background: #111114;
    border-right: 1px solid #2a2a30;
}

/* Metric cards */
.metric-card {
    background: #16161a;
    border: 1px solid #2a2a30;
    border-radius: 4px;
    padding: 16px 20px;
    text-align: center;
    position: relative;
    overflow: hidden;
}
.metric-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 2px;
    background: var(--accent);
}
.metric-label {
    font-family: 'Share Tech Mono', monospace;
    font-size: 10px;
    letter-spacing: 3px;
    color: #666;
    text-transform: uppercase;
    margin-bottom: 6px;
}
.metric-value {
    font-family: 'Share Tech Mono', monospace;
    font-size: 28px;
    font-weight: 700;
    color: #e8e8e0;
}
.metric-sub {
    font-size: 11px;
    color: #555;
    margin-top: 2px;
}

/* Grade badge */
.grade-badge {
    display: inline-block;
    padding: 10px 28px;
    border-radius: 2px;
    font-family: 'Barlow Condensed', sans-serif;
    font-size: 32px;
    font-weight: 800;
    letter-spacing: 4px;
    text-transform: uppercase;
}

/* Section headers */
.section-header {
    font-family: 'Share Tech Mono', monospace;
    font-size: 11px;
    letter-spacing: 4px;
    color: #555;
    text-transform: uppercase;
    border-bottom: 1px solid #1e1e24;
    padding-bottom: 8px;
    margin: 24px 0 16px 0;
}

/* Progress bar override */
.stProgress > div > div {
    background: #1a1a20;
}

/* Titles */
h1 { font-family: 'Barlow Condensed', sans-serif !important; font-weight: 800 !important; letter-spacing: 2px !important; }
h2, h3 { font-family: 'Barlow Condensed', sans-serif !important; font-weight: 600 !important; }

/* File uploader */
[data-testid="stFileUploader"] {
    background: #111114;
    border: 1px dashed #2a2a30;
    border-radius: 4px;
}

/* Scrollbar */
::-webkit-scrollbar { width: 4px; }
::-webkit-scrollbar-track { background: #0d0d0f; }
::-webkit-scrollbar-thumb { background: #2a2a30; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
#  Load model
# ─────────────────────────────────────────────
@st.cache_resource
def load_model(path: str = "apple_qc_model.pkl") -> dict:
    pkl_path = Path(path)
    if not pkl_path.exists():
        # Try same directory as this script
        alt = Path(__file__).parent / path
        if alt.exists():
            pkl_path = alt
        else:
            st.error(f"Model file not found: {path}")
            st.stop()
    with open(pkl_path, "rb") as f:
        return pickle.load(f)


model = load_model()
THRESHOLDS = model["thresholds"]
WEIGHTS    = model["weights"]
GRADES     = model["grade_thresholds"]


# ─────────────────────────────────────────────
#  CV Pipeline (unchanged logic from original)
# ─────────────────────────────────────────────
def segment_apple(image: np.ndarray) -> Tuple[np.ndarray, any]:
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    masks = [
        cv2.inRange(hsv, np.array([0,   50,  50]), np.array([10, 255, 255])),
        cv2.inRange(hsv, np.array([160, 50,  50]), np.array([180, 255, 255])),
        cv2.inRange(hsv, np.array([35,  40,  40]), np.array([90, 255, 255])),
        cv2.inRange(hsv, np.array([15,  50,  50]), np.array([35, 255, 255])),
    ]
    combined = masks[0]
    for m in masks[1:]:
        combined = cv2.bitwise_or(combined, m)
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
    if contour is None:
        return 0.0
    area = cv2.contourArea(contour)
    perimeter = cv2.arcLength(contour, True)
    if perimeter == 0:
        return 0.0
    return min(1.0, (4 * np.pi * area) / (perimeter ** 2))


def analyze_color(image: np.ndarray, mask: np.ndarray) -> float:
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    pixels = hsv[mask == 255]
    if len(pixels) == 0:
        return 0.0
    saturation = pixels[:, 1].mean() / 255.0
    value      = pixels[:, 2].mean() / 255.0
    return float(np.clip(0.6 * saturation + 0.4 * value, 0, 1))


def detect_defects(image: np.ndarray, mask: np.ndarray) -> Tuple[float, np.ndarray]:
    apple_area = int(np.sum(mask == 255))
    if apple_area == 0:
        return 1.0, np.zeros_like(mask)
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l_channel = lab[:, :, 0]
    l_masked  = cv2.bitwise_and(l_channel, l_channel, mask=mask)
    apple_mean = float(l_channel[mask == 255].mean())
    dark_thr   = apple_mean * 0.65
    defect_mask = np.where((l_masked < dark_thr) & (mask == 255),
                           np.uint8(255), np.uint8(0))
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    defect_mask = cv2.morphologyEx(defect_mask, cv2.MORPH_OPEN, k, iterations=2)
    ratio = int(np.sum(defect_mask == 255)) / apple_area
    return float(ratio), defect_mask


def compute_grade(roundness, color_score, defect_ratio):
    r_pts = roundness * 100
    c_pts = color_score * 100
    d_pts = min(100, max(0, (1 - defect_ratio / THRESHOLDS["max_defect_ratio"]) * 100))
    overall = (WEIGHTS["roundness"] * r_pts +
               WEIGHTS["color"]     * c_pts +
               WEIGHTS["defect"]    * d_pts)
    if overall >= GRADES["A"]:
        return overall, "GRADE A", "Excellent", "#22c55e"
    elif overall >= GRADES["B"]:
        return overall, "GRADE B", "Acceptable", "#f59e0b"
    elif overall >= GRADES["C"]:
        return overall, "GRADE C", "Poor", "#f97316"
    else:
        return overall, "REJECTED", "Defective", "#ef4444"


def build_annotated(image, contour, defect_mask, grade_color_hex):
    """Return annotated BGR image."""
    out = image.copy()
    # Defect overlay (red tint)
    overlay = out.copy()
    overlay[defect_mask == 255] = (0, 0, 180)
    cv2.addWeighted(overlay, 0.45, out, 0.55, 0, out)
    if contour is not None:
        # Convert hex to BGR
        h = grade_color_hex.lstrip("#")
        r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
        cv2.drawContours(out, [contour], -1, (b, g, r), 3)
    return out


def process_image(bgr: np.ndarray) -> dict:
    mask, contour = segment_apple(bgr)
    if contour is None or cv2.contourArea(contour) < THRESHOLDS["min_size_px"]:
        return {"grade": "NO APPLE", "label": "No apple detected",
                "overall": 0, "roundness": 0, "color_score": 0,
                "defect_ratio": 1.0, "color": "#6b7280", "annotated": bgr}

    roundness    = analyze_roundness(contour)
    color_score  = analyze_color(bgr, mask)
    defect_ratio, defect_mask = detect_defects(bgr, mask)
    overall, grade, label, color = compute_grade(roundness, color_score, defect_ratio)
    annotated = build_annotated(bgr, contour, defect_mask, color)

    return {
        "grade": grade, "label": label, "color": color,
        "overall": round(overall, 1),
        "roundness": round(roundness * 100, 1),
        "color_score": round(color_score * 100, 1),
        "defect_ratio": round(defect_ratio * 100, 2),
        "annotated": annotated,
    }


# ─────────────────────────────────────────────
#  Helpers
# ─────────────────────────────────────────────
def bgr_to_rgb(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

def bytes_to_bgr(data: bytes) -> np.ndarray:
    arr = np.frombuffer(data, np.uint8)
    return cv2.imdecode(arr, cv2.IMREAD_COLOR)

def metric_card(label, value, sub="", accent="#22c55e"):
    st.markdown(f"""
    <div class="metric-card" style="--accent:{accent}">
        <div class="metric-label">{label}</div>
        <div class="metric-value">{value}</div>
        <div class="metric-sub">{sub}</div>
    </div>""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
#  Sidebar
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🍎 Apple QC")
    st.markdown(f"<span style='font-family:monospace;font-size:11px;color:#555'>model v{model['version']}</span>",
                unsafe_allow_html=True)
    st.divider()

    st.markdown("### Thresholds")
    min_round = st.slider("Min Roundness", 0.0, 1.0,
                          THRESHOLDS["min_roundness"], 0.01)
    max_defect = st.slider("Max Defect %", 0.0, 0.30,
                           THRESHOLDS["max_defect_ratio"], 0.005,
                           format="%.3f")
    min_color = st.slider("Min Color Score", 0.0, 1.0,
                          THRESHOLDS["min_color_score"], 0.01)

    # Live-patch thresholds
    THRESHOLDS["min_roundness"]    = min_round
    THRESHOLDS["max_defect_ratio"] = max_defect
    THRESHOLDS["min_color_score"]  = min_color

    st.divider()
    st.markdown("### Weights")
    w_shape  = st.slider("Shape weight",  0.0, 1.0, WEIGHTS["roundness"], 0.05)
    w_color  = st.slider("Color weight",  0.0, 1.0, WEIGHTS["color"],     0.05)
    w_defect = st.slider("Defect weight", 0.0, 1.0, WEIGHTS["defect"],    0.05)
    total_w  = w_shape + w_color + w_defect
    if total_w > 0:
        WEIGHTS["roundness"] = w_shape  / total_w
        WEIGHTS["color"]     = w_color  / total_w
        WEIGHTS["defect"]    = w_defect / total_w

    st.caption(f"Normalized → shape {WEIGHTS['roundness']:.2f} | color {WEIGHTS['color']:.2f} | defect {WEIGHTS['defect']:.2f}")
    st.divider()
    mode = st.radio("Input mode", ["📸 Single Image", "📁 Batch Folder"])


# ─────────────────────────────────────────────
#  Main header
# ─────────────────────────────────────────────
st.markdown("""
<div style='margin-bottom:8px'>
<span style='font-family:Share Tech Mono,monospace;font-size:11px;letter-spacing:4px;color:#555'>
COMPUTER VISION // QUALITY CONTROL
</span>
</div>
<h1 style='font-size:52px;letter-spacing:3px;margin:0;line-height:1'>
APPLE QC INSPECTOR
</h1>
<p style='color:#555;font-family:Share Tech Mono,monospace;font-size:12px;margin-top:8px'>
Segmentation · Roundness · Color · Defect Detection
</p>
""", unsafe_allow_html=True)

st.divider()


# ─────────────────────────────────────────────
#  SINGLE IMAGE MODE
# ─────────────────────────────────────────────
if mode == "📸 Single Image":
    upload = st.file_uploader(
        "Drop an apple image here",
        type=["jpg", "jpeg", "png", "bmp", "webp"],
        label_visibility="collapsed",
    )

    if upload:
        bgr = bytes_to_bgr(upload.read())
        if bgr is None:
            st.error("Could not decode image.")
            st.stop()

        with st.spinner("Running QC pipeline…"):
            t0 = time.time()
            result = process_image(bgr)
            elapsed = time.time() - t0

        # Layout
        col_img, col_ann = st.columns(2, gap="medium")
        with col_img:
            st.markdown('<div class="section-header">ORIGINAL</div>', unsafe_allow_html=True)
            st.image(bgr_to_rgb(bgr), use_container_width=True)
        with col_ann:
            st.markdown('<div class="section-header">ANALYSED</div>', unsafe_allow_html=True)
            st.image(bgr_to_rgb(result["annotated"]), use_container_width=True)

        st.divider()

        # Grade badge
        grade_color = result["color"]
        st.markdown(f"""
        <div style='text-align:center;margin:12px 0 24px 0'>
            <div class="grade-badge" style='background:{grade_color}22;
                 color:{grade_color};border:2px solid {grade_color}44'>
                {result["grade"]}
            </div>
            <div style='margin-top:8px;font-family:Share Tech Mono,monospace;
                 font-size:12px;color:#555'>
                {result["label"].upper()} &nbsp;·&nbsp; {elapsed*1000:.0f} ms
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Metrics row
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            metric_card("Overall Score", f"{result['overall']}", "/100", grade_color)
        with c2:
            metric_card("Shape", f"{result['roundness']}", "circularity /100",
                        "#60a5fa" if result["roundness"] >= min_round*100 else "#ef4444")
        with c3:
            metric_card("Color", f"{result['color_score']}", "vibrancy /100",
                        "#60a5fa" if result["color_score"] >= min_color*100 else "#ef4444")
        with c4:
            metric_card("Defects", f"{result['defect_ratio']}%", "of surface",
                        "#60a5fa" if result["defect_ratio"] <= max_defect*100 else "#ef4444")

        # Score bars
        st.markdown('<div class="section-header" style="margin-top:28px">SCORE BREAKDOWN</div>',
                    unsafe_allow_html=True)

        def score_bar(label, val, max_val=100, warn=None):
            pct = val / max_val
            bar_color = "#ef4444" if (warn and val > warn) else (
                        "#22c55e" if pct >= 0.8 else
                        "#f59e0b" if pct >= 0.6 else "#f97316")
            st.markdown(f"""
            <div style='margin-bottom:12px'>
                <div style='display:flex;justify-content:space-between;
                     font-family:Share Tech Mono,monospace;font-size:11px;
                     color:#888;margin-bottom:4px'>
                    <span>{label}</span><span>{val}</span>
                </div>
                <div style='background:#1a1a20;border-radius:2px;height:6px'>
                    <div style='width:{min(pct*100,100):.1f}%;height:6px;
                         background:{bar_color};border-radius:2px;
                         transition:width 0.6s ease'></div>
                </div>
            </div>
            """, unsafe_allow_html=True)

        score_bar("SHAPE (roundness)", result["roundness"])
        score_bar("COLOR (vibrancy)", result["color_score"])
        score_bar("DEFECT SURFACE", result["defect_ratio"], max_val=15,
                  warn=max_defect * 100)

        # Pass/Fail checklist
        st.markdown('<div class="section-header">QUALITY GATES</div>', unsafe_allow_html=True)
        checks = [
            ("Roundness ≥ threshold",
             result["roundness"] / 100 >= min_round),
            ("Color score ≥ threshold",
             result["color_score"] / 100 >= min_color),
            (f"Defects ≤ {max_defect*100:.1f}%",
             result["defect_ratio"] / 100 <= max_defect),
        ]
        for desc, passed in checks:
            icon  = "✅" if passed else "❌"
            color = "#22c55e" if passed else "#ef4444"
            st.markdown(
                f"<span style='font-family:Share Tech Mono,monospace;"
                f"font-size:13px;color:{color}'>{icon} &nbsp; {desc}</span>",
                unsafe_allow_html=True,
            )

        # Download annotated image
        st.divider()
        _, dl_col, _ = st.columns([2, 1, 2])
        with dl_col:
            _, buf = cv2.imencode(".jpg", result["annotated"])
            st.download_button(
                "⬇ Download Annotated Image",
                data=buf.tobytes(),
                file_name=f"qc_{result['grade'].replace(' ', '_')}.jpg",
                mime="image/jpeg",
                use_container_width=True,
            )

    else:
        st.markdown("""
        <div style='text-align:center;padding:80px 0;color:#333;
             font-family:Share Tech Mono,monospace;font-size:13px'>
            ↑ Upload an apple image to begin inspection
        </div>
        """, unsafe_allow_html=True)


# ─────────────────────────────────────────────
#  BATCH MODE
# ─────────────────────────────────────────────
else:
    uploads = st.file_uploader(
        "Upload multiple apple images",
        type=["jpg", "jpeg", "png", "bmp", "webp"],
        accept_multiple_files=True,
        label_visibility="collapsed",
    )

    if uploads:
        st.markdown(f"<div class='section-header'>{len(uploads)} IMAGES QUEUED</div>",
                    unsafe_allow_html=True)

        progress = st.progress(0)
        results  = []

        for i, f in enumerate(uploads):
            bgr = bytes_to_bgr(f.read())
            if bgr is not None:
                r = process_image(bgr)
                r["filename"] = f.name
                r["bgr"] = bgr
                results.append(r)
            progress.progress((i + 1) / len(uploads))

        progress.empty()

        # Summary stats
        grades_count = {"GRADE A": 0, "GRADE B": 0, "GRADE C": 0,
                        "REJECTED": 0, "NO APPLE": 0}
        for r in results:
            for k in grades_count:
                if k in r["grade"]:
                    grades_count[k] += 1
                    break

        total = len(results)
        pass_count = grades_count["GRADE A"] + grades_count["GRADE B"]

        st.markdown('<div class="section-header">BATCH SUMMARY</div>', unsafe_allow_html=True)
        s1, s2, s3, s4, s5 = st.columns(5)
        cols = [s1, s2, s3, s4, s5]
        for col, (grade, count) in zip(cols, grades_count.items()):
            colors_map = {
                "GRADE A": "#22c55e", "GRADE B": "#f59e0b",
                "GRADE C": "#f97316", "REJECTED": "#ef4444",
                "NO APPLE": "#6b7280"
            }
            with col:
                metric_card(grade, str(count),
                            f"{count/total*100:.0f}%" if total else "—",
                            colors_map[grade])

        avg_score = np.mean([r["overall"] for r in results]) if results else 0
        pass_rate = pass_count / total * 100 if total else 0
        st.markdown(f"""
        <div style='font-family:Share Tech Mono,monospace;font-size:12px;
             color:#555;margin:16px 0;text-align:center'>
            AVG SCORE: <span style='color:#e8e8e0'>{avg_score:.1f}/100</span>
            &nbsp;·&nbsp;
            PASS RATE (A+B): <span style='color:#e8e8e0'>{pass_rate:.0f}%</span>
            &nbsp;·&nbsp;
            TOTAL INSPECTED: <span style='color:#e8e8e0'>{total}</span>
        </div>
        """, unsafe_allow_html=True)

        # Results grid
        st.markdown('<div class="section-header">INDIVIDUAL RESULTS</div>',
                    unsafe_allow_html=True)

        cols_per_row = 3
        for row_start in range(0, len(results), cols_per_row):
            row_results = results[row_start:row_start + cols_per_row]
            cols = st.columns(cols_per_row)
            for col, r in zip(cols, row_results):
                with col:
                    st.image(bgr_to_rgb(r["annotated"]), use_container_width=True)
                    st.markdown(f"""
                    <div style='text-align:center;margin-bottom:16px'>
                        <div style='font-family:Share Tech Mono,monospace;
                             font-size:10px;color:#555;margin-bottom:4px'>
                            {r["filename"][:24]}
                        </div>
                        <span style='background:{r["color"]}22;color:{r["color"]};
                             border:1px solid {r["color"]}44;
                             padding:3px 12px;border-radius:2px;
                             font-family:Barlow Condensed,sans-serif;
                             font-size:14px;font-weight:700;letter-spacing:2px'>
                            {r["grade"]}
                        </span>
                        <div style='font-family:Share Tech Mono,monospace;
                             font-size:11px;color:#555;margin-top:6px'>
                            {r["overall"]}/100
                            · shape {r["roundness"]}
                            · color {r["color_score"]}
                            · defect {r["defect_ratio"]}%
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div style='text-align:center;padding:80px 0;color:#333;
             font-family:Share Tech Mono,monospace;font-size:13px'>
            ↑ Upload one or more apple images to run batch inspection
        </div>
        """, unsafe_allow_html=True)

# ─────────────────────────────────────────────
#  Footer
# ─────────────────────────────────────────────
st.divider()
st.markdown(f"""
<div style='font-family:Share Tech Mono,monospace;font-size:10px;
     color:#2a2a30;text-align:center;padding:8px 0'>
    APPLE QC INSPECTOR · MODEL {model['name']} v{model['version']}
    · OpenCV + NumPy · Streamlit
</div>
""", unsafe_allow_html=True)
