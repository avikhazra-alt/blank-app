import io
import re
import base64
import mimetypes
import tempfile
from datetime import datetime

import streamlit as st
from openai import OpenAI

# PDF generation
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from reportlab.lib.units import inch

# Video frame extraction
import cv2


st.set_page_config(page_title="VastuSense Demo", layout="wide")

# ---- Fixed settings (hidden) ----
MODEL = "gpt-4o-mini"
MAX_OUTPUT_TOKENS = 1800
VIDEO_MAX_FRAMES = 8  # keep this low for cost/speed

client = OpenAI()  # uses OPENAI_API_KEY from env / Streamlit secrets

# ---- Small UI styling for "commercial" feel ----
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
    '<div class="vs-subtle">Upload a floor plan (+ optional walkthrough video) and get a polished Vastu report.</div>',
    unsafe_allow_html=True
)

# Session state
st.session_state.setdefault("out", "")
st.session_state.setdefault("err", "")
st.session_state.setdefault("usage", None)


# ---------- Helpers ----------
def extract_video_frames_as_dataurls(video_bytes: bytes, max_frames: int = 8):
    """
    Extracts up to max_frames frames uniformly across the video.
    Returns list of base64 data URLs (jpeg).
    Uses OpenCV; writes to temp file because cv2.VideoCapture needs a file path.
    """
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

        # Uniform sampling across the video timeline
        max_frames = max(1, int(max_frames))
        indices = [int(i * (total - 1) / max_frames) for i in range(max_frames)]
        indices = sorted(set(indices))

        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ok, frame = cap.read()
            if not ok or frame is None:
                continue

            # Encode frame as JPEG
            ok2, buf = cv2.imencode(".jpg", frame)
            if not ok2:
                continue

            b64 = base64.b64encode(buf.tobytes()).decode("utf-8")
            data_urls.append(f"data:image/jpeg;base64,{b64}")

        cap.release()

    return data_urls


def build_user_prompt(extra: str) -> str:
    base = """
You are a professional Vastu consultant preparing a client-facing mini report for a commercial demo.

Analyze the uploaded floor plan (and optional walkthrough frames if provided) and produce a polished, easy-to-read report.
Be practical, confident, and specific. If North direction isn't visible, state assumptions clearly.

Output in STRICT Markdown with this structure:

# VastuSense Report
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
1.
2.
3.
4.
5.

## Room-wise Guidance
### Entrance
### Living / Dining
### Kitchen
### Bedrooms
### Toilets / Baths
### Outdoor / Water (patio/pool/terrace)

## Quick Remedies (No structural changes)
- ...

## Questions to Confirm (to improve accuracy)
- ...

Formatting rules:
- Use short bullets, bold keywords, and clear headings.
- Avoid superstition-heavy tone; keep it consultative and modern.
- If any room labels are unclear, say so and provide best-effort guidance.
""".strip()

    if extra and extra.strip():
        base += f"\n\nAdditional user information:\n{extra.strip()}"
    return base


def run_openai(floorplan_file, extra_info: str, video_file=None):
    filename = floorplan_file.name
    file_bytes = floorplan_file.getvalue()
    mime = floorplan_file.type or mimetypes.guess_type(filename)[0] or "application/octet-stream"
    prompt_text = build_user_prompt(extra_info)

    content = [{"type": "input_text", "text": prompt_text}]

    # Attach floor plan
    if filename.lower().endswith(".pdf"):
        f = io.BytesIO(file_bytes)
        f.name = filename
        up = client.files.create(file=f, purpose="user_data")
        content.append({"type": "input_file", "file_id": up.id})
    else:
        b64 = base64.b64encode(file_bytes).decode("utf-8")
        data_url = f"data:{mime};base64,{b64}"
        content.append({"type": "input_image", "image_url": data_url, "detail": "low"})  # ✅ LOW detail

    # Optional video frames
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


# ---------- UI ----------
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
    st.markdown('<div class="small-note">Tip: A short 10–30s walkthrough video is enough. The app samples a few frames.</div>', unsafe_allow_html=True)

    if uploaded:
        name = uploaded.name.lower()
        if name.endswith(".pdf"):
            st.info(f"📄 Floor plan PDF uploaded: **{uploaded.name}**")
        else:
            st.image(uploaded, caption=uploaded.name, use_container_width=True)

    if video:
        st.video(video)

with right:
    st.markdown("### 2) Additional information (optional)")
    additional_info = st.text_area(
        "Helps improve accuracy",
        placeholder=(
            "Examples:\n"
            "• North direction (e.g., 'Top of plan is North')\n"
            "• Main entrance location/direction\n"
            "• Plot facing, city\n"
            "• Constraints (cannot move kitchen, etc.)"
        ),
        height=160
    )

    include_video = st.toggle("Include video frames in analysis (optional)", value=True)

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

    if not uploaded:
        st.session_state["err"] = "Please upload a floor plan (PDF or image)."
    else:
        with st.spinner("Analyzing floor plan..."):
            try:
                video_to_send = video if (include_video and video is not None) else None
                resp = run_openai(uploaded, additional_info, video_file=video_to_send)
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
    scores = extract_scorecard(report_md)
    top3 = extract_top_recos(report_md, n=3)

    # 1) Score gauge
    st.markdown("#### 📊 Score Snapshot")
    overall = scores.get("Overall")
    if overall is not None:
        st.progress(overall / 10.0)
        st.caption(f"Overall Score: {overall:.1f} / 10")

        metric_cols = st.columns(5)
        keys = ["Entrance", "Kitchen", "Bedrooms", "Toilets/Baths", "Living/Dining"]
        for i, k in enumerate(keys):
            val = scores.get(k)
            metric_cols[i].metric(k, f"{val:.1f}/10" if val is not None else "—")
    else:
        st.info("Scorecard not detected in output (the model should keep the Scorecard section).")

    # 2) Top 3 highlights tiles
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
        st.info("Top recommendations not detected (the model should keep 'Top 5 Recommendations').")

    # Pretty report card
    st.markdown("#### 🧾 Full Report")
    st.markdown('<div class="vs-card">', unsafe_allow_html=True)
    st.markdown(report_md)
    st.markdown("</div>", unsafe_allow_html=True)

    # 3) Download as PDF
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

    # Optional debugging
    # if st.session_state["usage"]:
    #     st.caption(f"Usage: {st.session_state['usage']}")
