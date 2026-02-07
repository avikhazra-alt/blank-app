import io
import re
import base64
import mimetypes
import tempfile
from datetime import datetime

import streamlit as st
from openai import OpenAI

from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from reportlab.lib.units import inch

import cv2
import pandas as pd

st.set_page_config(page_title="VastuSense Demo", layout="wide")

# ---- Fixed settings (hidden) ----
MODEL = "gpt-4o-mini"
MAX_OUTPUT_TOKENS = 2000
VIDEO_MAX_FRAMES = 8                  # keep low for cost/speed
MAX_FILE_MB = 50                      # HARD LIMIT per file
MAX_FILE_BYTES = MAX_FILE_MB * 1024 * 1024

client = OpenAI()  # uses OPENAI_API_KEY from env / Streamlit secrets

# ---- UI styling ----
st.markdown(
    """
    <style>
      .vs-card { padding: 16px 18px; border-radius: 14px; border: 1px solid rgba(49,51,63,.15); background: white; }
      .vs-subtle { color: rgba(49,51,63,.7); font-size: 0.95rem; }
      .vs-hr { margin: 18px 0; }
      .vs-tiles { display:flex; gap:12px; flex-wrap:wrap; }
      .vs-tile { flex: 1 1 260px; padding: 14px 14px; border-radius: 14px; border: 1px solid rgba(49,51,63,.12); background: white; }
      .vs-tile h4 { margin: 0 0 6px 0; font-size: 1.0rem; }
      .vs-tile p { margin: 0; color: rgba(49,51,63,.8); }
      .small-note { font-size: 0.9rem; color: rgba(49,51,63,.65); }
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
st.session_state.setdefault("usage", None)


# ---------- Helpers ----------
def bytes_to_mb(n: int) -> float:
    return n / (1024 * 1024)

def validate_file_size(file_obj, label: str):
    """Hard-block if uploaded file exceeds 50 MB."""
    if file_obj is None:
        return
    size = getattr(file_obj, "size", None)
    if size is None:
        size = len(file_obj.getvalue())
    if size > MAX_FILE_BYTES:
        raise ValueError(f"{label} is {bytes_to_mb(size):.1f} MB. Maximum allowed is {MAX_FILE_MB} MB.")

def parse_google_maps_link(url: str):
    """
    Extract lat/long from common Google Maps URL formats:
      - .../@18.5204,73.8567,17z
      - ...!3d18.5204!4d73.8567
    """
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


def build_user_prompt(additional_info: str,
                      north_on_plan: str,
                      entrance_dir: str,
                      home_type: str,
                      advice_style: str,
                      lat: float | None,
                      lon: float | None) -> str:
    base = f"""
You are a professional Vastu consultant preparing a client-facing mini report for a commercial demo.

Analyze the uploaded floor plan (and optional walkthrough frames if provided). Be practical and specific.
Never guess silently. If something is not visible, say "Not determined from the plan" and ask a question.

Use the following form inputs as ground truth if provided:
- North on plan: {north_on_plan}
- Main entrance direction: {entrance_dir}
- Home type: {home_type}
- Advice style: {advice_style}
""".strip()

    if lat is not None and lon is not None and not (lat == 0.0 and lon == 0.0):
        base += f"""

Property location (optional):
- Latitude: {lat}
- Longitude: {lon}

Use location only to improve orientation context (true north vs magnetic) and solar/daylight considerations.
""".strip()

    base += """

Output in STRICT Markdown with this structure (do not omit any section):

# VastuSense Report

## Vastu Readiness Score (1–100)
- Score: <integer from 1 to 100>
- What this score means (1 short paragraph)

## Confidence & Assumptions
- Confidence: High/Medium/Low
- Assumptions:
  - ...

## Snapshot
- Assumed North:
- Home type:
- Key strengths:
- Key risks:

## Scorecard (0–10)
- Entrance:
- Kitchen:
- Bedrooms:
- Toilets/Baths:
- Living/Dining:
- Overall:

## Top 5 Recommendations (Most Impact)
1. [Impact: High/Medium/Low | Effort: Low/Medium/High] ...
2. [Impact: ... | Effort: ...] ...
3. ...
4. ...
5. ...

## Room-wise Guidance
### Entrance
### Living / Dining
### Kitchen
### Bedrooms
### Toilets / Baths
### Outdoor / Water (patio/pool/terrace)

## Quick Remedies (No structural changes)
- ...

## Action Plan
### Next 7 days (easy fixes)
- ...
### Next 30 days (optional upgrades)
- ...

## Questions to Confirm (to improve accuracy)
- ...
""".strip()

    if additional_info and additional_info.strip():
        base += f"\n\nAdditional user information:\n{additional_info.strip()}"

    return base


def run_openai(floorplan_file, prompt_text: str, video_file=None):
    filename = floorplan_file.name
    file_bytes = floorplan_file.getvalue()
    mime = floorplan_file.type or mimetypes.guess_type(filename)[0] or "application/octet-stream"

    content = [{"type": "input_text", "text": prompt_text}]

    if filename.lower().endswith(".pdf"):
        f = io.BytesIO(file_bytes)
        f.name = filename
        up = client.files.create(file=f, purpose="user_data")
        content.append({"type": "input_file", "file_id": up.id})
    else:
        b64 = base64.b64encode(file_bytes).decode("utf-8")
        data_url = f"data:{mime};base64,{b64}"
        content.append({"type": "input_image", "image_url": data_url, "detail": "low"})  # ✅ LOW detail

    if video_file is not None:
        frames = extract_video_frames_as_dataurls(video_file.getvalue(), max_frames=VIDEO_MAX_FRAMES)
        if frames:
            content.append({"type": "input_text", "text": "Optional walkthrough video frames (additional context):"})
            for f_url in frames:
                content.append({"type": "input_image", "image_url": f_url, "detail": "low"})  # ✅ LOW detail

    resp = client.responses.create(
        model=MODEL,
        max_output_tokens=MAX_OUTPUT_TOKENS,
        input=[{"role": "user", "content": content}],
    )
    return resp


# ---------- Output parsing ----------
def extract_readiness_score_1_100(md: str):
    in_section = False
    for line in md.splitlines():
        if line.strip().startswith("## Vastu Readiness Score"):
            in_section = True
            continue
        if in_section and line.strip().startswith("## "):
            break
        if in_section:
            m = re.match(r"^\s*-\s*Score\s*:\s*([0-9]{1,3})\s*$", line.strip(), flags=re.IGNORECASE)
            if m:
                v = int(m.group(1))
                return max(1, min(100, v))
    return None

def extract_scorecard(md: str) -> dict:
    scores = {}
    in_section = False
    for line in md.splitlines():
        if line.strip().startswith("## Scorecard"):
            in_section = True
            continue
        if in_section and line.strip().startswith("## "):
            break
        if in_section:
            m = re.match(r"^\s*-\s*([^:]+)\s*:\s*([0-9]+(\.[0-9]+)?)", line.strip())
            if m:
                key = m.group(1).strip()
                val = float(m.group(2))
                scores[key] = max(0.0, min(10.0, val))
    return scores

def extract_top_recos(md: str, n=3) -> list:
    recos = []
    in_section = False
    for line in md.splitlines():
        if line.strip().startswith("## Top 5 Recommendations"):
            in_section = True
            continue
        if in_section and line.strip().startswith("## "):
            break
        if in_section:
            m = re.match(r"^\s*\d+\.\s+(.*)", line.strip())
            if m:
                recos.append(m.group(1).strip())
                if len(recos) >= n:
                    break
    return recos


# ---------- PDF ----------
def markdown_to_plain(md: str) -> str:
    text = md
    text = re.sub(r"^##\s*", "\n", text, flags=re.MULTILINE)
    text = re.sub(r"^#\s*", "\n", text, flags=re.MULTILINE)
    text = text.replace("**", "")
    text = re.sub(r"^\s*-\s*", "• ", text, flags=re.MULTILINE)
    return text.strip()

def make_pdf_bytes(title: str, md_report: str) -> bytes:
    buf = io.BytesIO()
    c = canvas.Canvas(buf, pagesize=A4)
    width, height = A4

    margin = 0.7 * inch
    y = height - margin

    c.setFont("Helvetica-Bold", 14)
    c.drawString(margin, y, title)
    y -= 18

    c.setFont("Helvetica", 10)

    plain = markdown_to_plain(md_report)
    lines = []
    for para in plain.split("\n"):
        if not para.strip():
            lines.append("")
            continue

        max_chars = 105
        while len(para) > max_chars:
            cut = para.rfind(" ", 0, max_chars)
            if cut == -1:
                cut = max_chars
            lines.append(para[:cut].rstrip())
            para = para[cut:].lstrip()
        lines.append(para)

    for line in lines:
        if y <= margin:
            c.showPage()
            c.setFont("Helvetica", 10)
            y = height - margin
        c.drawString(margin, y, line[:1400])
        y -= 14

    c.save()
    buf.seek(0)
    return buf.read()


# ================== UI ==================
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

    if uploaded:
        if uploaded.name.lower().endswith(".pdf"):
            st.info(f"📄 Floor plan PDF uploaded: **{uploaded.name}** ({bytes_to_mb(uploaded.size):.1f} MB)")
        else:
            st.image(uploaded, caption=f"{uploaded.name} ({bytes_to_mb(uploaded.size):.1f} MB)",
                     use_container_width=True)

    if video:
        st.video(video)
        st.caption(f"🎥 Video size: {bytes_to_mb(video.size):.1f} MB")

with right:
    st.markdown("### 2) Quick inputs (recommended)")
    cA, cB, cC, cD = st.columns(4)

    north_on_plan = cA.selectbox("North on plan", ["Unknown", "Top", "Right", "Bottom", "Left"])
    entrance_dir = cB.selectbox("Main entrance", ["Unknown", "North", "East", "South", "West", "NE", "NW", "SE", "SW"])
    home_type = cC.selectbox("Home type", ["Unknown", "Apartment", "Villa/Bungalow", "Row house"])
    advice_style = cD.selectbox("Advice style", ["Practical", "Strict"])

    st.markdown("### 3) Property location (optional)")
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

    st.markdown("### 4) Additional information (optional)")
    additional_info = st.text_area(
        "Helps improve accuracy",
        placeholder="Example: exact entrance location, constraints (cannot move kitchen), preferences (minimal changes)…",
        height=120
    )

    include_video = st.toggle("Include video frames in analysis (optional)", value=True)

    if uploaded:
        st.info(
            f"✅ Floor plan: {uploaded.name}\n"
            f"✅ Video: {'Included' if (video and include_video) else 'Not included'}\n"
            f"⚙️ Vision detail: LOW (cost-optimized)\n"
            f"📦 Upload limit: {MAX_FILE_MB} MB per file"
        )

    c1, c2 = st.columns([1, 1])
    show = c1.button("✨ Show Vastu Suggestions", type="primary", use_container_width=True)
    clear = c2.button("🧹 Clear", use_container_width=True)

if clear:
    st.session_state["out"] = ""
    st.session_state["err"] = ""
    st.session_state["usage"] = None


# ---------- Run ----------
if show:
    st.session_state["err"] = ""
    st.session_state["out"] = ""
    st.session_state["usage"] = None

    try:
        if not uploaded:
            st.session_state["err"] = "Please upload a floor plan (PDF or image)."
        else:
            # HARD LIMITS
            validate_file_size(uploaded, "Floor plan file")
            if include_video and video is not None:
                validate_file_size(video, "Video file")

            with st.spinner("Analyzing floor plan..."):
                prompt_text = build_user_prompt(
                    additional_info=additional_info,
                    north_on_plan=north_on_plan,
                    entrance_dir=entrance_dir,
                    home_type=home_type,
                    advice_style=advice_style,
                    lat=lat, lon=lon
                )

                video_to_send = video if (include_video and video is not None) else None
                resp = run_openai(uploaded, prompt_text, video_file=video_to_send)

                st.session_state["out"] = resp.output_text
                st.session_state["usage"] = getattr(resp, "usage", None)

    except Exception as e:
        msg = str(e)
        if "insufficient_quota" in msg or "exceeded your current quota" in msg:
            st.session_state["err"] = (
                "OpenAI API quota/billing issue (429: insufficient_quota).\n\n"
                "Fix: OpenAI Platform → Billing → add payment method/credits, and check Usage/Limits."
            )
        else:
            st.session_state["err"] = msg


# ---------- Output ----------
st.markdown('<hr class="vs-hr"/>', unsafe_allow_html=True)
st.markdown("### 📄 VastuSense Output")

if st.session_state["err"]:
    st.error(st.session_state["err"])

if st.session_state["out"]:
    report_md = st.session_state["out"]

    readiness = extract_readiness_score_1_100(report_md)
    scores_0_10 = extract_scorecard(report_md)
    top3 = extract_top_recos(report_md, n=3)

    # Readiness score meter (1–100)
    st.markdown("#### ✅ Vastu Readiness Score (1–100)")
    if readiness is not None:
        st.progress(readiness / 100.0)
        st.metric("Readiness Score", f"{readiness}/100")
    else:
        st.info("Readiness score not detected. (Model should keep '## Vastu Readiness Score (1–100)' section.)")

    # Optional 0–10 score snapshot
    st.markdown("#### 📊 Score Snapshot (0–10)")
    overall = scores_0_10.get("Overall")
    if overall is not None:
        st.progress(overall / 10.0)
        st.caption(f"Overall (0–10): {overall:.1f} / 10")
        metric_cols = st.columns(5)
        keys = ["Entrance", "Kitchen", "Bedrooms", "Toilets/Baths", "Living/Dining"]
        for i, k in enumerate(keys):
            val = scores_0_10.get(k)
            metric_cols[i].metric(k, f"{val:.1f}/10" if val is not None else "—")
    else:
        st.caption("0–10 scorecard not detected (optional).")

    # Top 3 tiles
    st.markdown("#### ⭐ Top 3 Highlights")
    if top3:
        st.markdown('<div class="vs-tiles">', unsafe_allow_html=True)
        for idx, rec in enumerate(top3, start=1):
            st.markdown(
                f"""
                <div class="vs-tile">
                  <h4>Recommendation #{idx}</h4>
                  <p>{rec}</p>
                </div>
                """,
                unsafe_allow_html=True
            )
        st.markdown("</div>", unsafe_allow_html=True)
    else:
        st.info("Top recommendations not detected. (Keep 'Top 5 Recommendations' in output.)")

    # Full report card
    st.markdown("#### 🧾 Full Report")
    st.markdown('<div class="vs-card">', unsafe_allow_html=True)
    st.markdown(report_md)
    st.markdown("</div>", unsafe_allow_html=True)

    # Copy-ready report
    st.markdown("#### 📋 Copy-ready report")
    st.text_area("Copy/paste this into WhatsApp / Email", report_md, height=220)

    # Download PDF
    st.markdown("#### ⬇️ Download")
    pdf_bytes = make_pdf_bytes("VastuSense Report", report_md)
    ts = datetime.now().strftime("%Y%m%d_%H%M")
    st.download_button(
        label="📄 Download Report as PDF",
        data=pdf_bytes,
        file_name=f"VastuSense_Report_{ts}.pdf",
        mime="application/pdf",
        use_container_width=True
    )
