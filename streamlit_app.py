import io
import re
import json
import base64
import mimetypes
import tempfile
from datetime import datetime

import streamlit as st
from openai import OpenAI

import cv2
import pandas as pd

from PIL import Image, ImageDraw

# PDF (high quality)
from reportlab.lib.pagesizes import A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors

import numpy as np
import matplotlib.pyplot as plt


# =========================
# App Config
# =========================
st.set_page_config(page_title="VastuSense Demo", layout="wide")

MODEL = "gpt-5.2-pro"
MAX_OUTPUT_TOKENS = 3200
VIDEO_MAX_FRAMES = 8

MAX_FILE_MB = 50
MAX_FILE_BYTES = MAX_FILE_MB * 1024 * 1024

LOADING_GIF_PATH = "assets/vastu_loading.gif"

client = OpenAI()  # OPENAI_API_KEY must be set via env or Streamlit secrets


# =========================
# Styling
# =========================
st.markdown(
    """
    <style>
      .vs-card { padding: 16px 18px; border-radius: 14px; border: 1px solid rgba(49,51,63,.15); background: white; }
      .vs-subtle { color: rgba(49,51,63,.7); font-size: 0.95rem; }
      .vs-hr { margin: 18px 0; }
      .small-note { font-size: 0.9rem; color: rgba(49,51,63,.65); }

      .loading-panel {
        border: 1px solid rgba(49,51,63,.15);
        border-radius: 14px;
        background: white;
        padding: 12px 14px;
      }
      .loading-title { font-weight: 700; margin: 0 0 6px 0; }
      .loading-sub { color: rgba(49,51,63,.7); font-size: 0.92rem; margin: 0 0 8px 0; }
    </style>
    """,
    unsafe_allow_html=True
)

st.markdown("## 🏠 VastuSense — Floor Plan Quick Review")
st.markdown(
    f'<div class="vs-subtle">Upload a floor plan (+ optional walkthrough video). Max upload size: <b>{MAX_FILE_MB} MB</b> per file.</div>',
    unsafe_allow_html=True
)

# Session state
st.session_state.setdefault("out", "")
st.session_state.setdefault("err", "")
st.session_state.setdefault("meta", {})  # parsed JSON blocks


# =========================
# Helpers
# =========================
def bytes_to_mb(n: int) -> float:
    return n / (1024 * 1024)

def validate_file_size(file_obj, label: str):
    if file_obj is None:
        return
    size = getattr(file_obj, "size", None)
    if size is None:
        size = len(file_obj.getvalue())
    if size > MAX_FILE_BYTES:
        raise ValueError(f"{label} is {bytes_to_mb(size):.1f} MB. Maximum allowed is {MAX_FILE_MB} MB.")

def parse_google_maps_link(url: str):
    if not url:
        return None, None
    m = re.search(r"@(-?\d+\.\d+),(-?\d+\.\d+)", url)
    if m:
        return float(m.group(1)), float(m.group(2))
    m = re.search(r"!3d(-?\d+\.\d+)!4d(-?\d+\.\d+)", url)
    if m:
        return float(m.group(1)), float(m.group(2))
    return None, None

def extract_video_frames_as_dataurls(video_bytes: bytes, max_frames: int = 8):
    data_urls = []
    if not video_bytes:
        return data_urls

    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=True) as tmp:
        tmp.write(video_bytes)
        tmp.flush()

        cap = cv2.VideoCapture(tmp.name)
        if not cap.isOpened():
            return data_urls

        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
        if total <= 0:
            cap.release()
            return data_urls

        max_frames = max(1, int(max_frames))
        indices = [int(i * (total - 1) / max_frames) for i in range(max_frames)]
        indices = sorted(set(indices))

        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ok, frame = cap.read()
            if not ok or frame is None:
                continue

            ok2, buf = cv2.imencode(".jpg", frame)
            if not ok2:
                continue

            b64 = base64.b64encode(buf.tobytes()).decode("utf-8")
            data_urls.append(f"data:image/jpeg;base64,{b64}")

        cap.release()

    return data_urls


def sanitize_report_text(text: str) -> str:
    """Remove any accidental code fences like ```plaintext and trailing ```."""
    if not text:
        return ""
    t = text.strip()
    t = re.sub(r"^\s*```[a-zA-Z]*\s*\n", "", t)
    t = re.sub(r"\n\s*```\s*$", "", t)
    t = re.sub(r"^\s*plaintext\s*$", "", t, flags=re.MULTILINE)
    return t.strip()


def extract_json_block(tag: str, text: str):
    """
    Extracts JSON emitted as:
    TAG_JSON: {...}
    Returns dict or None.
    """
    if not text:
        return None
    m = re.search(rf"{re.escape(tag)}_JSON:\s*(\{{.*?\}})\s*$", text, flags=re.DOTALL | re.MULTILINE)
    if not m:
        return None
    raw = m.group(1).strip()
    try:
        return json.loads(raw)
    except Exception:
        return None


def strip_json_blocks(text: str):
    """Remove trailing *_JSON blocks from report before displaying."""
    if not text:
        return ""
    # remove any final lines like TAG_JSON: {...}
    return re.sub(r"(?m)^[A-Z_]+_JSON:\s*\{.*\}\s*$", "", text).strip()


# =========================
# Resident context
# =========================
GOAL_EXPLANATIONS = {
    "Health": "ventilation, dampness control, daylight, kitchen/toilet hygiene",
    "Sleep": "bed placement, mirror control, clutter reduction, calm lighting",
    "Career/Focus": "work/study desk placement, stable back support, distraction control",
    "Relationships": "stable seating, bedroom harmony, balanced zones",
    "Finance": "leakage control, SW stability, north flow, tidy storage",
    "Peace of mind": "NE lightness, open circulation, declutter, calm lighting",
}

def build_resident_context(primary: dict, others: list[dict]) -> str:
    lines = []
    lines.append("Primary Resident (Vastu-only personalization):")
    lines.append(f"- Name: {primary.get('name') or 'Not provided'}")
    lines.append(f"- Age group: {primary.get('age_group')}")
    lines.append(f"- Work/Lifestyle: {primary.get('work_style')}")
    lines.append(f"- Sleep schedule: {primary.get('sleep_schedule')}")
    lines.append(f"- Goals: {', '.join(primary.get('goals', [])) or 'Not specified'}")
    if primary.get("constraints"):
        lines.append(f"- Constraints: {primary.get('constraints')}")
    if primary.get("notes"):
        lines.append(f"- Notes: {primary.get('notes')}")

    if others:
        lines.append("\nAdditional Permanent Residents:")
        for i, r in enumerate(others, start=1):
            lines.append(f"{i}) Relation: {r.get('relation')}, Age group: {r.get('age_group')}, Needs: {r.get('needs') or 'None'}")
    else:
        lines.append("\nAdditional Permanent Residents: None")

    goals = primary.get("goals", [])
    if goals:
        lines.append("\nGoal Interpretation (prioritization guidance):")
        for g in goals:
            expl = GOAL_EXPLANATIONS.get(g)
            if expl:
                lines.append(f"- {g}: focus on {expl}")

    return "\n".join(lines)


# =========================
# Prompt Builder (Markdown tables + JSON outputs)
# =========================
DIRECTION_ELEMENT_CONTROLS = [
    ("North-East", "Water", "Health, clarity, spirituality"),
    ("East", "Air", "Growth, social life"),
    ("South-East", "Fire", "Kitchen, energy, metabolism"),
    ("South", "Fire/Earth", "Fame, strength"),
    ("South-West", "Earth", "Stability, relationships, finance"),
    ("West", "Water", "Gains, opportunities"),
    ("North-West", "Air", "Movement, networking"),
    ("North", "Water", "Career, flow of money"),
]

def build_user_prompt(
    *,
    client_name: str,
    property_type: str,
    entrance_direction: str,
    entrance_movement: str,
    house_alignment: str,
    additional_info: str,
    north_on_plan: str,
    home_type: str,
    advice_style: str,
    lat: float | None,
    lon: float | None,
    primary_resident: dict,
    other_residents: list[dict],
) -> str:
    dir_table = ["| Direction | Element | Controls |", "|---|---|---|"]
    for d, e, c in DIRECTION_ELEMENT_CONTROLS:
        dir_table.append(f"| {d} | {e} | {c} |")
    dir_table_md = "\n".join(dir_table)

    loc_line = ""
    if lat is not None and lon is not None and not (lat == 0.0 and lon == 0.0):
        loc_line = f"- Location: {lat}, {lon} (optional)"

    resident_block = build_resident_context(primary_resident, other_residents)

    prompt = f"""
You are a professional Vastu consultant writing a client-facing report.

STRICT RULES:
- Vastu-only analysis. Do NOT do astrology/birth charts.
- Output MUST be clean Markdown WITHOUT any code fences (no ```plaintext, no ```).
- Match the report alignment of a professional consultant: neat headings, clean bullets, and tables.

GROUND TRUTH INPUTS:
- Property Type: {property_type}
- Client: {client_name}
- Home Type: {home_type}
- North on plan: {north_on_plan}
- Entrance Direction: {entrance_direction} ({entrance_movement})
- House Alignment: {house_alignment}
- Advice Style: {advice_style}
{loc_line}

RESIDENT PROFILE (use ONLY to prioritize recommendations / room allocation):
{resident_block}

ADDITIONAL INFORMATION:
{additional_info.strip() if additional_info and additional_info.strip() else "None"}

OUTPUT FORMAT (STRICT MARKDOWN):

# VASTU SENSE
## Full Vastu Analysis Report

### Property Summary
| Field | Value |
|---|---|
| Property Type | {property_type} |
| Client | {client_name} |
| Entrance Direction | {entrance_direction} ({entrance_movement}) |
| House Alignment | {house_alignment} |
| North on plan | {north_on_plan} |
| Advice Style | {advice_style} |

---

## 1. Overall Energy Map
### Direction Element Controls
{dir_table_md}

**Overall Vastu Balance:** <1–2 lines>  
**Overall Rating:** <X / 10>  
**Vastu Readiness Score (1–100):** <integer 1..100>  
**Confidence:** <High/Medium/Low>

---

## 2. Entrance Analysis
• <3–6 bullets>

## 3. Living Room (mention zone/direction if you can infer)
• <3–6 bullets>

## 4. Dining Area
• <2–4 bullets>

## 5. Kitchen (mention zone/direction if you can infer)
• <4–7 bullets>

## 6. Master Bedroom (mention zone/direction if you can infer)
• <3–6 bullets>

## 7. Other Bedrooms
• <3–6 bullets>

## 8. Toilets
• <3–6 bullets>

## 9. Balconies
• <2–4 bullets>

## 10. Financial & Relationship Energy
• <2–4 bullets>

## 11. Health & Mental Peace
• <2–4 bullets>

## 12. Resident-Based Space Allocation (Personalized Vastu Plan)
• Recommend room allocation for primary resident and others (based on goals + age group).
• If unclear from plan, provide best-guess options + add questions.

## 13. Lifestyle-Compatible Remedies (Non-structural, Practical)
• Remedies aligned to resident priorities.
• No heavy reconstruction.

---

## Major Remedies Summary
• <4–7 bullets>

## Questions for Confirmation
• <3–7 bullets, only if needed>

**Final Conclusion:** <2–3 lines, professional closing>

IMPORTANT (MACHINE-READABLE OUTPUTS):
At the very end, output these three lines EXACTLY (single-line JSON each, no extra text after them):

SCORES_JSON: {{"rating": X, "readiness": Y}}
CONFIDENCE_JSON: {{"level":"High|Medium|Low","score": Z}}
DIRECTION_SCORES_JSON: {{"N":0-100,"NE":0-100,"E":0-100,"SE":0-100,"S":0-100,"SW":0-100,"W":0-100,"NW":0-100,"CENTER":0-100}}

Guidance:
- rating is 0..10, readiness is 1..100
- confidence score Z is 0..100
- direction scores reflect overall directional harmony based on the plan (use 50 if truly unknown)
""".strip()

    return prompt


# =========================
# OpenAI Call
# =========================
def run_openai(floorplan_file, prompt_text: str, video_file=None):
    filename = floorplan_file.name
    file_bytes = floorplan_file.getvalue()
    mime = floorplan_file.type or mimetypes.guess_type(filename)[0] or "application/octet-stream"

    content = [{"type": "input_text", "text": prompt_text}]

    # ---------- Image or PDF ----------
    if filename.lower().endswith(".pdf"):
        f = io.BytesIO(file_bytes)
        f.name = filename
        up = client.files.create(file=f, purpose="user_data")
        content.append({"type": "input_file", "file_id": up.id})
    else:
        b64 = base64.b64encode(file_bytes).decode("utf-8")
        data_url = f"data:{mime};base64,{b64}"
        content.append({
            "type": "input_image",
            "image_url": data_url,
            "detail": "auto"     # quality > cost
        })

    # ---------- Video frames ----------
    if video_file is not None:
        frames = extract_video_frames_as_dataurls(
            video_file.getvalue(),
            max_frames=VIDEO_MAX_FRAMES
        )
        if frames:
            content.append({"type": "input_text", "text": "Additional walkthrough frames:"})
            for f_url in frames:
                content.append({
                    "type": "input_image",
                    "image_url": f_url,
                    "detail": "auto"
                })

    # ---------- Strict schema output ----------
    response_schema = {
        "type": "json_schema",
        "json_schema": {
            "name": "vastu_report",
            "strict": True,
            "schema": {
                "type": "object",
                "properties": {
                    "report_markdown": {"type": "string"},
                    "scores": {
                        "type": "object",
                        "properties": {
                            "rating": {"type": "number"},
                            "readiness": {"type": "integer"}
                        },
                        "required": ["rating", "readiness"]
                    },
                    "confidence": {
                        "type": "object",
                        "properties": {
                            "level": {"type": "string"},
                            "score": {"type": "integer"}
                        },
                        "required": ["level", "score"]
                    },
                    "direction_scores": {
                        "type": "object",
                        "additionalProperties": {"type": "integer"}
                    }
                },
                "required": ["report_markdown","scores","confidence","direction_scores"]
            }
        }
    }

    resp = client.responses.create(
        model=MODEL,
        reasoning={"effort": "high"},   # key for strict logic
        temperature=0,                  # deterministic formatting
        max_output_tokens=MAX_OUTPUT_TOKENS,
        response_format=response_schema,
        input=[{"role": "user", "content": content}],
    )

    return resp



# =========================
# Premium Score UI (1: circular gauge, 4: confidence meter)
# =========================
def grade_label(value_0_100: int | None):
    if value_0_100 is None:
        return "Unknown"
    if value_0_100 >= 85:
        return "Excellent"
    if value_0_100 >= 70:
        return "Very Good"
    if value_0_100 >= 55:
        return "Balanced"
    if value_0_100 >= 40:
        return "Needs Improvement"
    return "Weak Alignment"

def circular_gauge(title: str, value: float | int | None, max_value: float, sublabel: str = ""):
    """
    Pure HTML/SVG gauge, avoids flaky parsing issues.
    """
    if value is None:
        st.warning(f"{title}: not detected")
        return

    v = float(value)
    v = max(0.0, min(max_value, v))
    pct = v / max_value

    # SVG circle math
    r = 46
    c = 2 * 3.14159265 * r
    dash = pct * c
    gap = c - dash

    st.markdown(
        f"""
        <div style="border:1px solid rgba(49,51,63,.15); border-radius:14px; padding:12px 14px; background:white;">
          <div style="font-weight:700; margin-bottom:8px;">{title}</div>
          <div style="display:flex; gap:14px; align-items:center;">
            <svg width="110" height="110" viewBox="0 0 120 120">
              <circle cx="60" cy="60" r="{r}" stroke="rgba(49,51,63,.12)" stroke-width="12" fill="none"/>
              <circle cx="60" cy="60" r="{r}" stroke="rgba(79,70,229,.95)" stroke-width="12" fill="none"
                stroke-linecap="round"
                stroke-dasharray="{dash:.2f} {gap:.2f}"
                transform="rotate(-90 60 60)"/>
              <text x="60" y="60" text-anchor="middle" dominant-baseline="central" style="font-size:22px; font-weight:800; fill:rgba(17,24,39,.92);">
                {v:.0f}
              </text>
              <text x="60" y="82" text-anchor="middle" dominant-baseline="central" style="font-size:11px; fill:rgba(55,65,81,.75);">
                / {max_value:.0f}
              </text>
            </svg>
            <div>
              <div style="font-size:14px; font-weight:800; color:rgba(17,24,39,.92);">{grade_label(int(round(pct*100)))}</div>
              <div style="color:rgba(55,65,81,.75); font-size:12px;">{sublabel}</div>
            </div>
          </div>
        </div>
        """,
        unsafe_allow_html=True
    )

def confidence_meter(conf_level: str | None, conf_score: int | None):
    """
    Confidence card + bar. (4)
    """
    label = conf_level or "Unknown"
    score = conf_score if conf_score is not None else None
    if score is None:
        # fallback from label
        if label.lower() == "high":
            score = 85
        elif label.lower() == "medium":
            score = 60
        elif label.lower() == "low":
            score = 35
        else:
            score = None

    if score is None:
        st.warning("Confidence: not detected")
        return

    score = int(max(0, min(100, score)))

    st.markdown(
        f"""
        <div style="border:1px solid rgba(49,51,63,.15); border-radius:14px; padding:12px 14px; background:white;">
          <div style="font-weight:700; margin-bottom:6px;">AI Confidence</div>
          <div style="color:rgba(55,65,81,.75); font-size:12px; margin-bottom:10px;">
            Level: <b>{label}</b> • Score: <b>{score}/100</b>
          </div>
          <div style="height:14px;background:rgba(49,51,63,.12);border-radius:999px;overflow:hidden;">
            <div style="width:{score}%;height:100%;background:linear-gradient(90deg, rgba(79,70,229,.95), rgba(34,197,94,.85));"></div>
          </div>
          <div style="margin-top:8px; color:rgba(55,65,81,.7); font-size:12px;">
            Higher confidence means room identification/orientation is clearer in the plan. Confirm missing details to improve reliability.
          </div>
        </div>
        """,
        unsafe_allow_html=True
    )


# =========================
# Radar chart (2)
# =========================
def render_direction_radar(direction_scores: dict | None):
    if not direction_scores:
        st.info("Direction radar not available.")
        return

    order = ["N", "NE", "E", "SE", "S", "SW", "W", "NW"]
    vals = []
    for k in order:
        v = direction_scores.get(k, 50)
        try:
            v = float(v)
        except Exception:
            v = 50.0
        vals.append(max(0.0, min(100.0, v)))

    # close loop
    angles = np.linspace(0, 2*np.pi, len(order), endpoint=False).tolist()
    vals2 = vals + [vals[0]]
    angles2 = angles + [angles[0]]

    fig = plt.figure(figsize=(4.8, 4.2))
    ax = plt.subplot(111, polar=True)
    ax.plot(angles2, vals2, linewidth=2)
    ax.fill(angles2, vals2, alpha=0.15)
    ax.set_xticks(angles)
    ax.set_xticklabels(order)
    ax.set_yticks([20, 40, 60, 80, 100])
    ax.set_yticklabels(["20", "40", "60", "80", "100"])
    ax.set_title("Directional Harmony Radar (0–100)", pad=14)
    st.pyplot(fig, clear_figure=True)


# =========================
# Heatmap overlay on floor plan (3)
# =========================
def score_to_color(score: float):
    """
    Map 0..100 to red->yellow->green.
    """
    s = max(0.0, min(100.0, float(score)))
    # simple gradient: red -> yellow -> green
    if s < 50:
        # red to yellow
        t = s / 50.0
        r = 220
        g = int(60 + (180 * t))
        b = 60
    else:
        # yellow to green
        t = (s - 50.0) / 50.0
        r = int(220 - (140 * t))
        g = 220
        b = 80
    return (r, g, b)

def heatmap_overlay_on_image(image_bytes: bytes, direction_scores: dict):
    """
    Creates a 3x3 grid overlay on the image:
      NW  N  NE
       W  C   E
      SW  S  SE
    Each cell colored based on direction score.
    """
    img = Image.open(io.BytesIO(image_bytes)).convert("RGBA")
    w, h = img.size
    overlay = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)

    # map cells
    grid = [
        ["NW", "N", "NE"],
        ["W", "CENTER", "E"],
        ["SW", "S", "SE"],
    ]

    cell_w = w / 3.0
    cell_h = h / 3.0

    for r in range(3):
        for c in range(3):
            key = grid[r][c]
            score = direction_scores.get(key, 50)
            col = score_to_color(score)
            alpha = 85  # transparency
            x0 = int(c * cell_w)
            y0 = int(r * cell_h)
            x1 = int((c + 1) * cell_w)
            y1 = int((r + 1) * cell_h)
            draw.rectangle([x0, y0, x1, y1], fill=(col[0], col[1], col[2], alpha))

    # grid lines
    line_alpha = 110
    for i in [1, 2]:
        draw.line([int(i*cell_w), 0, int(i*cell_w), h], fill=(255, 255, 255, line_alpha), width=3)
        draw.line([0, int(i*cell_h), w, int(i*cell_h)], fill=(255, 255, 255, line_alpha), width=3)

    out = Image.alpha_composite(img, overlay)
    out_buf = io.BytesIO()
    out.save(out_buf, format="PNG")
    out_buf.seek(0)
    return out_buf.getvalue()


# =========================
# High-quality PDF renderer
# =========================
def _esc(s: str) -> str:
    return (s.replace("&", "&amp;")
             .replace("<", "&lt;")
             .replace(">", "&gt;"))

def _is_table_line(line: str) -> bool:
    return line.strip().startswith("|") and line.strip().endswith("|")

def _parse_md_table(lines: list[str], start_idx: int):
    table_lines = []
    i = start_idx
    while i < len(lines) and _is_table_line(lines[i]):
        table_lines.append(lines[i].strip())
        i += 1

    rows = []
    for tl in table_lines:
        parts = [p.strip() for p in tl.strip("|").split("|")]
        rows.append(parts)

    # remove separator row |---|---|
    cleaned = []
    for r in rows:
        if all(set(c) <= set("-:") and "-" in c for c in r):
            continue
        cleaned.append(r)

    return cleaned, i

def make_pdf_bytes(title: str, md_report: str) -> bytes:
    md = (md_report or "").strip()
    md = sanitize_report_text(md)

    lines = md.splitlines()

    buf = io.BytesIO()
    doc = SimpleDocTemplate(
        buf,
        pagesize=A4,
        leftMargin=36,
        rightMargin=36,
        topMargin=36,
        bottomMargin=36
    )

    styles = getSampleStyleSheet()
    h1 = ParagraphStyle("H1", parent=styles["Heading1"], fontSize=16, spaceAfter=10)
    h2 = ParagraphStyle("H2", parent=styles["Heading2"], fontSize=12, spaceBefore=10, spaceAfter=6)
    h3 = ParagraphStyle("H3", parent=styles["Heading3"], fontSize=11, spaceBefore=8, spaceAfter=4)
    body = ParagraphStyle("Body", parent=styles["BodyText"], fontSize=10, leading=13, spaceAfter=3)

    story = []
    story.append(Paragraph(_esc(title), h1))
    story.append(Spacer(1, 6))

    i = 0
    while i < len(lines):
        line = lines[i].rstrip()

        if not line.strip():
            story.append(Spacer(1, 6))
            i += 1
            continue

        if line.strip() == "---":
            story.append(Spacer(1, 8))
            i += 1
            continue

        if _is_table_line(line):
            table_data, next_i = _parse_md_table(lines, i)
            if table_data:
                col_count = max(len(r) for r in table_data)
                for r in table_data:
                    while len(r) < col_count:
                        r.append("")

                table_para = []
                for r in table_data:
                    table_para.append([Paragraph(_esc(c), body) for c in r])

                t = Table(table_para, hAlign="LEFT")
                t.setStyle(TableStyle([
                    ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#f3f4f6")),
                    ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#d1d5db")),
                    ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                    ("VALIGN", (0, 0), (-1, -1), "TOP"),
                    ("LEFTPADDING", (0, 0), (-1, -1), 6),
                    ("RIGHTPADDING", (0, 0), (-1, -1), 6),
                    ("TOPPADDING", (0, 0), (-1, -1), 4),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
                ]))
                story.append(t)
                story.append(Spacer(1, 10))
            i = next_i
            continue

        if line.startswith("# "):
            story.append(Paragraph(_esc(line[2:].strip()), h1))
            i += 1
            continue
        if line.startswith("## "):
            story.append(Paragraph(_esc(line[3:].strip()), h2))
            i += 1
            continue
        if line.startswith("### "):
            story.append(Paragraph(_esc(line[4:].strip()), h3))
            i += 1
            continue

        # bullets
        if line.strip().startswith("• "):
            story.append(Paragraph("• " + _esc(line.strip()[2:]), body))
            i += 1
            continue
        if line.strip().startswith("- "):
            story.append(Paragraph("• " + _esc(line.strip()[2:]), body))
            i += 1
            continue

        # bold inline
        safe = _esc(line)
        safe = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", safe)
        story.append(Paragraph(safe, body))
        i += 1

    doc.build(story)
    buf.seek(0)
    return buf.read()


# =========================
# UI
# =========================
left, right = st.columns([1, 1], gap="large")

with left:
    st.markdown("### 1) Upload floor plan")
    uploaded = st.file_uploader(
        "Upload floor plan (PDF / PNG / JPG / WEBP)",
        type=["pdf", "png", "jpg", "jpeg", "webp"]
    )

    st.markdown("### 1b) Optional: Upload walkthrough video")
    video = st.file_uploader(
        "Upload video (MP4 / MOV / M4V) — optional",
        type=["mp4", "mov", "m4v"]
    )
    st.markdown('<div class="small-note">Tip: Keep video short. We sample a few frames (cost-optimized).</div>',
                unsafe_allow_html=True)

    uploaded_is_pdf = False
    if uploaded:
        uploaded_is_pdf = uploaded.name.lower().endswith(".pdf")
        if uploaded_is_pdf:
            st.info(f"📄 Floor plan PDF uploaded: **{uploaded.name}** ({bytes_to_mb(uploaded.size):.1f} MB)")
        else:
            st.image(uploaded, caption=f"{uploaded.name} ({bytes_to_mb(uploaded.size):.1f} MB)", use_container_width=True)

    if video:
        st.video(video)
        st.caption(f"🎥 Video size: {bytes_to_mb(video.size):.1f} MB")

with right:
    loading_panel = st.empty()

    st.markdown("### 2) Report header inputs")
    h1, h2 = st.columns(2)
    client_name = h1.text_input("Client name", value="Vipul")
    property_type = h2.text_input("Property type", value="Residential Apartment")

    h3, h4 = st.columns(2)
    entrance_direction = h3.selectbox(
        "Entrance direction",
        ["Unknown", "East", "West", "North", "South", "NE", "NW", "SE", "SW"],
        index=2
    )
    entrance_movement = h4.text_input("Entrance movement text", value="movement West → East")
    house_alignment = st.text_input("House alignment text", value="East–South orientation")

    st.markdown("### 3) Quick inputs (recommended)")
    cA, cB, cC, cD = st.columns(4)
    north_on_plan = cA.selectbox("North on plan", ["Unknown", "Top", "Right", "Bottom", "Left"])
    home_type = cB.selectbox("Home type", ["Unknown", "Apartment", "Villa/Bungalow", "Row house"], index=1)
    advice_style = cC.selectbox("Advice style", ["Practical", "Strict"])
    include_video = cD.toggle("Include video frames", value=True)

    st.markdown("### 4) Property location (optional)")
    loc_mode = st.radio("Choose input method", ["Paste Google Maps link", "Enter latitude/longitude"], horizontal=True)
    lat = lon = None
    if loc_mode == "Paste Google Maps link":
        gmap_link = st.text_input("Google Maps link", placeholder="Paste Google Maps share link / URL here")
        lat, lon = parse_google_maps_link(gmap_link)
        if gmap_link and (lat is None or lon is None):
            st.warning("Could not extract latitude/longitude from this link. Try copying the full Maps URL again.")
    else:
        cL1, cL2 = st.columns(2)
        lat = cL1.number_input("Latitude", value=0.0, format="%.6f")
        lon = cL2.number_input("Longitude", value=0.0, format="%.6f")

    if lat is not None and lon is not None and not (lat == 0.0 and lon == 0.0):
        st.success(f"📍 Location captured: {lat:.6f}, {lon:.6f}")
        st.map(pd.DataFrame([{"lat": lat, "lon": lon}]))

    st.markdown("### 5) Additional information (optional)")
    additional_info = st.text_area(
        "Helps improve accuracy",
        placeholder="Example: exact entrance location, constraints (cannot move kitchen), preferences (minimal changes)…",
        height=90
    )

    st.markdown("### 6) Resident profile (Vastu-only personalization)")
    st.caption("Used only to prioritize recommendations and space allocation. No astrology.")

    r1, r2 = st.columns(2)
    primary_resident_name = r1.text_input("Primary resident name (optional)", value="")
    primary_age_group = r2.selectbox("Primary resident age group", ["Adult", "Senior", "Teen", "Child"])

    r3, r4 = st.columns(2)
    primary_work_style = r3.selectbox("Primary work/lifestyle", ["WFH", "Office", "Business", "Student", "Retired", "Other"])
    primary_sleep_schedule = r4.selectbox("Primary sleep schedule", ["Normal", "Early", "Late", "Shift/Irregular"])

    primary_goals = st.multiselect(
        "Primary goals (pick up to 4)",
        ["Health", "Sleep", "Career/Focus", "Relationships", "Finance", "Peace of mind"],
        default=["Peace of mind", "Health"]
    )
    if len(primary_goals) > 4:
        st.warning("Please select up to 4 goals for cleaner prioritization.")

    primary_constraints = st.text_input("Constraints (optional)", placeholder="e.g., No structural changes, Cannot move kitchen")
    primary_notes = st.text_area("Notes (optional)", placeholder="e.g., senior mobility, kids study priority, sleep issues", height=60)

    st.markdown("#### Additional permanent residents (optional)")
    num_residents = st.number_input("How many additional residents?", min_value=0, max_value=5, value=0, step=1)

    other_residents = []
    for i in range(int(num_residents)):
        st.markdown(f"Resident #{i+1}")
        a, b, c = st.columns(3)
        rel = a.selectbox(f"Relation #{i+1}", ["Spouse", "Child", "Parent", "Sibling", "Other"], key=f"rel_{i}")
        ag = b.selectbox(f"Age group #{i+1}", ["Adult", "Senior", "Teen", "Child"], key=f"ag_{i}")
        needs = c.text_input(f"Special needs #{i+1} (optional)", key=f"needs_{i}", placeholder="e.g., study focus, mobility, allergies")
        other_residents.append({"relation": rel, "age_group": ag, "needs": needs})

    primary_resident = {
        "name": primary_resident_name.strip(),
        "age_group": primary_age_group,
        "work_style": primary_work_style,
        "sleep_schedule": primary_sleep_schedule,
        "goals": primary_goals[:4],
        "constraints": primary_constraints.strip(),
        "notes": primary_notes.strip(),
    }

    if uploaded:
        st.info(
            f"✅ Floor plan: {uploaded.name}\n"
            f"✅ Video: {'Included' if (video and include_video) else 'Not included'}\n"
            f"⚙️ Vision detail: LOW (cost-optimized)\n"
            f"📦 Upload limit: {MAX_FILE_MB} MB per file"
        )

    st.markdown("### 7) Generate")
    c1, c2 = st.columns([1, 1])
    show = c1.button("✨ Generate Vastu Report", type="primary", use_container_width=True)
    clear = c2.button("🧹 Clear", use_container_width=True)

if clear:
    st.session_state["out"] = ""
    st.session_state["err"] = ""
    st.session_state["meta"] = {}


# =========================
# Run
# =========================
if show:
    st.session_state["err"] = ""
    st.session_state["out"] = ""
    st.session_state["meta"] = {}

    try:
        if not uploaded:
            st.session_state["err"] = "Please upload a floor plan (PDF or image)."
        else:
            validate_file_size(uploaded, "Floor plan file")
            if include_video and video is not None:
                validate_file_size(video, "Video file")

            prompt_text = build_user_prompt(
                client_name=client_name.strip() or "Client",
                property_type=property_type.strip() or "Residential",
                entrance_direction=entrance_direction,
                entrance_movement=entrance_movement.strip() or "movement not specified",
                house_alignment=house_alignment.strip() or "alignment not specified",
                additional_info=additional_info or "",
                north_on_plan=north_on_plan,
                home_type=home_type,
                advice_style=advice_style,
                lat=lat, lon=lon,
                primary_resident=primary_resident,
                other_residents=other_residents,
            )

            video_to_send = video if (include_video and video is not None) else None

            # Right-side small loading box
            try:
                loading_panel.markdown(
                    "<div class='loading-panel'>"
                    "<div class='loading-title'>Generating report…</div>"
                    "<div class='loading-sub'>Analyzing plan + resident priorities</div>"
                    "</div>",
                    unsafe_allow_html=True
                )
                loading_panel.image(LOADING_GIF_PATH, width=160)
            except Exception:
                loading_panel.info("Generating report…")

            with st.spinner("Generating report..."):
                try:
                    resp = run_openai(uploaded, prompt_text, video_file=video_to_send)
                    raw = sanitize_report_text(resp.output_text or "")

                    # Parse machine-readable blocks
                    scores = extract_json_block("SCORES", raw)
                    conf = extract_json_block("CONFIDENCE", raw)
                    dirs = extract_json_block("DIRECTION_SCORES", raw)

                    st.session_state["meta"] = {
                        "scores": scores or {},
                        "confidence": conf or {},
                        "direction_scores": dirs or {},
                    }

                    # Remove JSON blocks from the visible report
                    cleaned_report = strip_json_blocks(raw)
                    st.session_state["out"] = cleaned_report.strip()
                finally:
                    loading_panel.empty()

    except Exception as e:
        loading_panel.empty()
        msg = str(e)
        if "insufficient_quota" in msg or "exceeded your current quota" in msg:
            st.session_state["err"] = (
                "OpenAI API quota/billing issue (429: insufficient_quota).\n\n"
                "Fix: OpenAI Platform → Billing → add payment method/credits, and check Usage/Limits."
            )
        else:
            st.session_state["err"] = msg


# =========================
# Output
# =========================
st.markdown('<hr class="vs-hr"/>', unsafe_allow_html=True)
st.markdown("## 📄 VastuSense Output")

if st.session_state["err"]:
    st.error(st.session_state["err"])

if st.session_state["out"]:
    report_md = st.session_state["out"]
    meta = st.session_state.get("meta", {}) or {}
    scores = meta.get("scores", {}) or {}
    conf = meta.get("confidence", {}) or {}
    direction_scores = meta.get("direction_scores", {}) or {}

    # ---- Premium score UI (1 + 4) ----
    st.markdown("### 📊 Consultant Scorecard")
    a, b, c = st.columns([1, 1, 1])

    rating = scores.get("rating", None)
    readiness = scores.get("readiness", None)

    # normalize numeric types
    try:
        rating = float(rating) if rating is not None else None
    except Exception:
        rating = None
    try:
        readiness = int(readiness) if readiness is not None else None
    except Exception:
        readiness = None

    conf_level = conf.get("level", None)
    conf_score = conf.get("score", None)
    try:
        conf_score = int(conf_score) if conf_score is not None else None
    except Exception:
        conf_score = None

    with a:
        circular_gauge("Overall Rating", rating, 10, "Overall Vastu balance (0–10)")

    with b:
        circular_gauge("Readiness Score", readiness, 100, "Overall readiness (1–100)")

    with c:
        confidence_meter(conf_level, conf_score)

    # ---- Direction radar (2) ----
    st.markdown("### 🧭 Directional Harmony (Radar)")
    render_direction_radar(direction_scores)

    # ---- Heatmap overlay (3) ----
    st.markdown("### 🗺️ Floor Plan Vastu Heatmap (Directional Overlay)")
    if uploaded and (not uploaded.name.lower().endswith(".pdf")) and direction_scores:
        try:
            img_bytes = uploaded.getvalue()
            overlay_png = heatmap_overlay_on_image(img_bytes, direction_scores)
            st.image(overlay_png, caption="Overlay shows directional harmony (green higher, red lower).", use_container_width=True)
        except Exception as ex:
            st.info(f"Heatmap overlay could not be generated for this image. ({ex})")
    else:
        st.info("Heatmap overlay is available for image uploads (PNG/JPG/WEBP). For PDFs, please upload an image export of the plan.")

    # ---- Report (Markdown with tables) ----
    st.markdown("### 🧾 Full Report")
    st.markdown('<div class="vs-card">', unsafe_allow_html=True)
    st.markdown(report_md)
    st.markdown("</div>", unsafe_allow_html=True)

    # ---- Download PDF ----
    st.markdown("### ⬇️ Download")
    pdf_bytes = make_pdf_bytes("VastuSense Report", report_md)
    ts = datetime.now().strftime("%Y%m%d_%H%M")
    st.download_button(
        label="📄 Download Report as PDF",
        data=pdf_bytes,
        file_name=f"VastuSense_Report_{ts}.pdf",
        mime="application/pdf",
        use_container_width=True
    )


# =========================
# requirements.txt (for your repo)
# =========================
# streamlit
# openai
# reportlab
# opencv-python-headless
# pandas
# pillow
# numpy
# matplotlib
#
# Also enforce 50MB UI/server limit:
# .streamlit/config.toml
# [server]
# maxUploadSize = 50
