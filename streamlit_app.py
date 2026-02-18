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


# =========================
# App Config
# =========================
st.set_page_config(page_title="VastuSense Demo", layout="wide")

MODEL = "gpt-4o"                # as per your choice
MAX_OUTPUT_TOKENS = 2600        # higher for professional template + resident sections
VIDEO_MAX_FRAMES = 8
MAX_FILE_MB = 50
MAX_FILE_BYTES = MAX_FILE_MB * 1024 * 1024

# Option-2 loading GIF (small panel on right)
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
      .loading-title {
        font-weight: 700;
        margin: 0 0 6px 0;
      }
      .loading-sub {
        color: rgba(49,51,63,.7);
        font-size: 0.92rem;
        margin: 0 0 8px 0;
      }
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


# =========================
# Prompt Builder (PDF-style + resident-aware)
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

GOAL_EXPLANATIONS = {
    "Health": "ventilation, dampness control, clean water areas, kitchen/toilet hygiene, daylight",
    "Sleep": "bed placement, mirror control, clutter reduction, calming light, noise/visual load",
    "Career/Focus": "study/work desk placement, stable back support, north/east light, distraction control",
    "Relationships": "stable seating, bedroom harmony, clutter control, balanced zones",
    "Finance": "leakage control, north flow, SW stability, tidy storage and circulation",
    "Peace of mind": "NE lightness, open circulation, declutter, calm lighting",
}

def build_resident_context(primary: dict, others: list[dict]) -> str:
    lines = []
    lines.append("Primary Resident (for personalization, Vastu-only — no astrology):")
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

    lines.append("\nGoal Interpretation (how to prioritize recommendations):")
    for g in primary.get("goals", []):
        expl = GOAL_EXPLANATIONS.get(g)
        if expl:
            lines.append(f"- {g}: focus on {expl}")

    return "\n".join(lines)

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
    direction_block = "\n".join([f"{d} — {e} — {c}" for d, e, c in DIRECTION_ELEMENT_CONTROLS])

    loc_block = ""
    if lat is not None and lon is not None and not (lat == 0.0 and lon == 0.0):
        loc_block = f"""
Property Location (optional):
- Latitude: {lat}
- Longitude: {lon}
Use this only to improve orientation context and daylight considerations.
""".strip()

    resident_block = build_resident_context(primary_resident, other_residents)

    prompt = f"""
You are a professional Vastu consultant writing a client-facing report.

SCOPE RULES (important):
- This is VASTU analysis only. Do NOT do birth chart / astrology analysis.
- Resident profile is used ONLY to personalize priorities, room allocation, and practical remedies.
- If a detail is not visible in the plan, say "Not determined from the plan" and add it under “Questions for Confirmation”.

INPUTS (treat these as ground truth where provided):
- Property Type: {property_type}
- Client: {client_name}
- Home Type: {home_type}
- North on plan: {north_on_plan}
- Entrance Direction: {entrance_direction}
- Entrance Movement: {entrance_movement}
- House Alignment: {house_alignment}
- Advice Style: {advice_style}
{loc_block}

RESIDENT PROFILE (for personalization; Vastu-only, no astrology):
{resident_block}

ADDITIONAL USER INFORMATION (optional):
{additional_info.strip() if additional_info and additional_info.strip() else "None"}

ANALYSIS SOURCE:
- Use the uploaded floor plan (and optional walkthrough frames if provided).
- Do not ask for higher resolution; operate on the given plan.

OUTPUT FORMAT (STRICT — follow this structure and level of detail):
Use plain text with headings and numbered sections exactly like below.
Use bullet points with "•". Keep it professional and consultant-grade.

VASTU SENSE
Full Vastu Analysis Report
Property Type: {property_type}
Client: {client_name}
Entrance Direction: {entrance_direction} ({entrance_movement})
House Alignment: {house_alignment}

1. Overall Energy Map
Direction Element Controls
{direction_block}
Overall Vastu Balance: <1–2 lines>
Overall Rating: <X / 10>
Vastu Readiness Score (1–100): <integer 1..100>

2. Entrance Analysis
• <3–6 bullets>

3. Living Room (mention zone/direction if you can infer)
• <3–6 bullets>

4. Dining Area
• <2–4 bullets>

5. Kitchen (mention zone/direction if you can infer)
• <4–7 bullets>

6. Master Bedroom (mention zone/direction if you can infer)
• <3–6 bullets>

7. Other Bedrooms
• <3–6 bullets>

8. Toilets
• <3–6 bullets>

9. Balconies
• <2–4 bullets>

10. Financial & Relationship Energy
• <2–4 bullets>

11. Health & Mental Peace
• <2–4 bullets>

12. Resident-Based Space Allocation (Personalized Vastu Plan)
• Clearly recommend which room should be master, which room for children/parents, and where the work/study setup should be.
• Tie recommendations to resident age group + goals (Health/Sleep/Career/Focus/Finance/Peace of mind).
• If room identity is unclear, give best-guess options + ask for confirmation.

13. Lifestyle-Compatible Remedies (Non-structural, Practical)
• Provide remedies aligned to resident priorities (sleep/study/health, etc.)
• Keep remedies feasible (no heavy reconstruction).

Major Remedies Summary
• <4–7 bullets>

Questions for Confirmation
• <3–7 bullets, only if needed>

Final Conclusion: <2–3 lines, professional closing>
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

    if filename.lower().endswith(".pdf"):
        f = io.BytesIO(file_bytes)
        f.name = filename
        up = client.files.create(file=f, purpose="user_data")
        content.append({"type": "input_file", "file_id": up.id})
    else:
        b64 = base64.b64encode(file_bytes).decode("utf-8")
        data_url = f"data:{mime};base64,{b64}"
        content.append({"type": "input_image", "image_url": data_url, "detail": "low"})

    if video_file is not None:
        frames = extract_video_frames_as_dataurls(video_file.getvalue(), max_frames=VIDEO_MAX_FRAMES)
        if frames:
            content.append({"type": "input_text", "text": "Optional walkthrough video frames (additional context):"})
            for f_url in frames:
                content.append({"type": "input_image", "image_url": f_url, "detail": "low"})

    resp = client.responses.create(
        model=MODEL,
        max_output_tokens=MAX_OUTPUT_TOKENS,
        input=[{"role": "user", "content": content}],
    )
    return resp


# =========================
# PDF Export
# =========================
def make_pdf_bytes(title: str, plain_report: str) -> bytes:
    buf = io.BytesIO()
    c = canvas.Canvas(buf, pagesize=A4)
    width, height = A4

    margin = 0.7 * inch
    y = height - margin

    c.setFont("Helvetica-Bold", 14)
    c.drawString(margin, y, title)
    y -= 18

    c.setFont("Helvetica", 10)

    lines = []
    for para in plain_report.split("\n"):
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
    # Option 2: right-side small loading box placeholder
    loading_panel = st.empty()

    st.markdown("### 2) Report header inputs")
    h1, h2 = st.columns(2)
    client_name = h1.text_input("Client name", value="Vipul")
    property_type = h2.text_input("Property type", value="Residential Apartment")

    h3, h4 = st.columns(2)
    entrance_direction = h3.selectbox("Entrance direction", ["Unknown", "East", "West", "North", "South", "NE", "NW", "SE", "SW"], index=1)
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
        height=100
    )

    # -------------------------
    # Resident Profile (Vastu-only)
    # -------------------------
    st.markdown("### 6) Resident profile (Vastu-only personalization)")
    st.caption("Used only to prioritize recommendations and space allocation. No birth chart / astrology.")

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

    primary_constraints = st.text_input("Constraints (optional)", placeholder="e.g., No structural changes, Cannot move kitchen, Budget cap, etc.")
    primary_notes = st.text_area("Notes (optional)", placeholder="e.g., senior mobility, kids study priority, sleep issues, etc.", height=70)

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

    # Build resident objects
    primary_resident = {
        "name": primary_resident_name.strip(),
        "age_group": primary_age_group,
        "work_style": primary_work_style,
        "sleep_schedule": primary_sleep_schedule,
        "goals": primary_goals[:4],
        "constraints": primary_constraints.strip(),
        "notes": primary_notes.strip(),
    }

    # Info box
    if uploaded:
        st.info(
            f"✅ Floor plan: {uploaded.name}\n"
            f"✅ Video: {'Included' if (video and include_video) else 'Not included'}\n"
            f"⚙️ Vision detail: LOW (cost-optimized)\n"
            f"📦 Upload limit: {MAX_FILE_MB} MB per file"
        )

    c1, c2 = st.columns([1, 1])
    show = c1.button("✨ Generate Vastu Report", type="primary", use_container_width=True)
    clear = c2.button("🧹 Clear", use_container_width=True)

if clear:
    st.session_state["out"] = ""
    st.session_state["err"] = ""
    st.session_state["usage"] = None


# =========================
# Run
# =========================
if show:
    st.session_state["err"] = ""
    st.session_state["out"] = ""
    st.session_state["usage"] = None

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

            # Option 2: show GIF in a small right-panel card
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
                    st.session_state["out"] = (resp.output_text or "").strip()
                    st.session_state["usage"] = getattr(resp, "usage", None)
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
st.markdown("### 📄 VastuSense Output")

if st.session_state["err"]:
    st.error(st.session_state["err"])

if st.session_state["out"]:
    report_text = st.session_state["out"]

    st.markdown("#### 🧾 Full Report (Template Style + Resident-Aware)")
    st.markdown('<div class="vs-card">', unsafe_allow_html=True)
    st.text(report_text)
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("#### 📋 Copy-ready report")
    st.text_area("Copy/paste this into WhatsApp / Email", report_text, height=280)

    st.markdown("#### ⬇️ Download")
    pdf_bytes = make_pdf_bytes("VastuSense Report", report_text)
    ts = datetime.now().strftime("%Y%m%d_%H%M")
    st.download_button(
        label="📄 Download Report as PDF",
        data=pdf_bytes,
        file_name=f"VastuSense_Report_{ts}.pdf",
        mime="application/pdf",
        use_container_width=True
    )
