import io
import re
import base64
import mimetypes
from datetime import datetime

import streamlit as st
from openai import OpenAI

# PDF generation (simple, clean)
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from reportlab.lib.units import inch

st.set_page_config(page_title="VastuSense Demo", layout="wide")

# ---- Fixed settings (hidden) ----
MODEL = "gpt-4o-mini"
MAX_OUTPUT_TOKENS = 1800

client = OpenAI()  # uses OPENAI_API_KEY from env / Streamlit secrets

# ---- Small UI styling for "commercial" feel ----
st.markdown(
    """
    <style>
      .vs-card { padding: 16px 18px; border-radius: 14px; border: 1px solid rgba(49,51,63,.15); background: white; }
      .vs-subtle { color: rgba(49,51,63,.7); font-size: 0.95rem; }
      .vs-pill { display:inline-block; padding:10px 12px; border-radius:14px; border: 1px solid rgba(49,51,63,.12); background: rgba(0,0,0,.02); }
      .vs-hr { margin: 18px 0; }
      .vs-tiles { display:flex; gap:12px; flex-wrap:wrap; }
      .vs-tile { flex: 1 1 260px; padding: 14px 14px; border-radius: 14px; border: 1px solid rgba(49,51,63,.12); background: white; }
      .vs-tile h4 { margin: 0 0 6px 0; font-size: 1.0rem; }
      .vs-tile p { margin: 0; color: rgba(49,51,63,.8); }
    </style>
    """,
    unsafe_allow_html=True
)

st.markdown("## 🏠 VastuSense — Floor Plan Quick Review")
st.markdown(
    '<div class="vs-subtle">Upload a floor plan and get a polished Vastu report + scorecard (demo).</div>',
    unsafe_allow_html=True
)

# Session state
st.session_state.setdefault("out", "")
st.session_state.setdefault("err", "")
st.session_state.setdefault("usage", None)

# ---- Layout ----
left, right = st.columns([1, 1], gap="large")

with left:
    st.markdown("### 1) Upload floor plan")
    uploaded = st.file_uploader(
        "Upload floor plan (PDF / PNG / JPG / WEBP)",
        type=["pdf", "png", "jpg", "jpeg", "webp"]
    )

    if uploaded:
        name = uploaded.name.lower()
        if name.endswith(".pdf"):
            st.info(f"📄 Floor plan PDF uploaded: **{uploaded.name}**")
        else:
            st.image(uploaded, caption=uploaded.name, use_container_width=True)

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

    c1, c2 = st.columns([1, 1])
    show = c1.button("✨ Show Vastu Suggestions", type="primary", use_container_width=True)
    clear = c2.button("🧹 Clear", use_container_width=True)

if clear:
    st.session_state["out"] = ""
    st.session_state["err"] = ""
    st.session_state["usage"] = None


def build_user_prompt(extra: str) -> str:
    # Commercial / report-style output request
    base = """
You are a professional Vastu consultant preparing a client-facing mini report for a commercial demo.

Analyze the uploaded floor plan image/PDF and produce a polished, easy-to-read report.
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


def run_openai(uploaded_file, extra_info):
    filename = uploaded_file.name
    file_bytes = uploaded_file.getvalue()
    mime = uploaded_file.type or mimetypes.guess_type(filename)[0] or "application/octet-stream"
    prompt_text = build_user_prompt(extra_info)

    # PDF: upload then reference via input_file
    if filename.lower().endswith(".pdf"):
        f = io.BytesIO(file_bytes)
        f.name = filename
        up = client.files.create(file=f, purpose="user_data")

        resp = client.responses.create(
            model=MODEL,
            max_output_tokens=MAX_OUTPUT_TOKENS,
            input=[{
                "role": "user",
                "content": [
                    {"type": "input_file", "file_id": up.id},
                    {"type": "input_text", "text": prompt_text},
                ],
            }],
        )
        return resp

    # Image: base64 data URL with LOW detail to reduce tokens/cost
    b64 = base64.b64encode(file_bytes).decode("utf-8")
    data_url = f"data:{mime};base64,{b64}"

    resp = client.responses.create(
        model=MODEL,
        max_output_tokens=MAX_OUTPUT_TOKENS,
        input=[{
            "role": "user",
            "content": [
                {"type": "input_text", "text": prompt_text},
                {"type": "input_image", "image_url": data_url, "detail": "low"},  # ✅ LOW detail
            ],
        }],
    )
    return resp


# ---------- Pretty UI helpers ----------
def extract_scorecard(md: str) -> dict:
    """
    Extracts score lines like:
    - Entrance: 7/10
    - Overall: 8
    Returns dict with keys -> float scores (0..10)
    """
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
                # clamp
                val = max(0.0, min(10.0, val))
                scores[key] = val
    return scores


def extract_top_recos(md: str, n=3) -> list:
    """
    Finds '## Top 5 Recommendations' section and extracts first N numbered items.
    """
    recos = []
    lines = md.splitlines()
    in_section = False
    for line in lines:
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
    """
    Light markdown -> text for PDF export.
    Keeps headings and bullets readable.
    """
    text = md

    # Convert headings to uppercase-ish
    text = re.sub(r"^######\s*", "", text, flags=re.MULTILINE)
    text = re.sub(r"^#####\s*", "", text, flags=re.MULTILINE)
    text = re.sub(r"^####\s*", "", text, flags=re.MULTILINE)
    text = re.sub(r"^###\s*", "", text, flags=re.MULTILINE)
    text = re.sub(r"^##\s*", "\n", text, flags=re.MULTILINE)
    text = re.sub(r"^#\s*", "\n", text, flags=re.MULTILINE)

    # Remove bold markers
    text = text.replace("**", "")

    # Keep bullets readable
    text = re.sub(r"^\s*-\s*", "• ", text, flags=re.MULTILINE)

    return text.strip()


def make_pdf_bytes(title: str, md_report: str) -> bytes:
    """
    Generates a simple one-file PDF with wrapped text.
    """
    buf = io.BytesIO()
    c = canvas.Canvas(buf, pagesize=A4)
    width, height = A4

    margin = 0.7 * inch
    y = height - margin

    # Title
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

        # wrap per line
        max_chars = 105  # rough wrap for A4 with 10pt font
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

        c.drawString(margin, y, line[:1400])  # safe guard
        y -= 14

    c.save()
    buf.seek(0)
    return buf.read()


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
                resp = run_openai(uploaded, additional_info)
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

st.markdown('<hr class="vs-hr"/>', unsafe_allow_html=True)

# ---------- OUTPUT AREA ----------
st.markdown("### 📄 VastuSense Output")

if st.session_state["err"]:
    st.error(st.session_state["err"])

if st.session_state["out"]:
    report_md = st.session_state["out"]

    # Parse scorecard + top recos
    scores = extract_scorecard(report_md)
    top3 = extract_top_recos(report_md, n=3)

    # 1) SCORE GAUGE
    st.markdown("#### 📊 Score Snapshot")
    overall = scores.get("Overall")
    if overall is not None:
        st.progress(overall / 10.0)
        st.caption(f"Overall Score: {overall:.1f} / 10")

        # Optional metrics row
        metric_cols = st.columns(5)
        keys = ["Entrance", "Kitchen", "Bedrooms", "Toilets/Baths", "Living/Dining"]
        for i, k in enumerate(keys):
            val = scores.get(k)
            metric_cols[i].metric(k, f"{val:.1f}/10" if val is not None else "—")
    else:
        st.info("Scorecard not detected in output (ask the model to keep the Scorecard section).")

    # 2) TOP 3 HIGHLIGHTS TILES
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
        st.info("Top recommendations not detected (ask the model to keep the 'Top 5 Recommendations' section).")

    # Pretty report card
    st.markdown("#### 🧾 Full Report")
    st.markdown('<div class="vs-card">', unsafe_allow_html=True)
    st.markdown(report_md)
    st.markdown("</div>", unsafe_allow_html=True)

    # 3) DOWNLOAD AS PDF
    st.markdown("#### ⬇️ Download")
    pdf_bytes = make_pdf_bytes(
        title="VastuSense Report",
        md_report=report_md
    )
    ts = datetime.now().strftime("%Y%m%d_%H%M")
    st.download_button(
        label="📄 Download Report as PDF",
        data=pdf_bytes,
        file_name=f"VastuSense_Report_{ts}.pdf",
        mime="application/pdf",
        use_container_width=True
    )

    # (Optional) Show usage for debugging; comment out if you don't want it visible
    # if st.session_state["usage"]:
    #     st.caption(f"Usage: {st.session_state['usage']}")
