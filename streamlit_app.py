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
    text = re.sub(r"^##\s*", "\n*
