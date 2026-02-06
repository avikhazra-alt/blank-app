import io
import base64
import mimetypes
import streamlit as st
from openai import OpenAI

st.set_page_config(page_title="Vastu Floor Plan Sandbox", layout="centered")

# ---- Fixed settings (not shown in UI) ----
MODEL = "gpt-4o-mini"
MAX_OUTPUT_TOKENS = 1400

st.title("🏠 VastuSense Demo")
st.caption("Upload a floor plan and get Vastu suggestions.")

# Session state
st.session_state.setdefault("out", "")
st.session_state.setdefault("err", "")

client = OpenAI()  # uses OPENAI_API_KEY from environment/secrets

st.subheader("1) Upload floor plan")
uploaded = st.file_uploader("Upload floor plan (PDF / PNG / JPG / WEBP)", type=["pdf", "png", "jpg", "jpeg", "webp"])

if uploaded:
    name = uploaded.name.lower()
    if name.endswith(".pdf"):
        st.info(f"📄 Floor plan PDF uploaded: **{uploaded.name}**")
    else:
        st.image(uploaded, caption=uploaded.name, use_container_width=True)

st.subheader("2) Additional information")
additional_info = st.text_area(
    "Add any context (optional)",
    placeholder="Example: Main entrance direction (if known), city, plot facing, apartment vs bungalow, any constraints/preferences...",
    height=120
)

st.markdown("")

show = st.button("✨ Show Vastu Suggestions", type="primary", use_container_width=True)

def build_user_prompt(extra: str) -> str:
    base = (
        "You are a Vastu consultant. Analyze the uploaded floor plan and provide practical Vastu suggestions.\n\n"
        "Output format:\n"
        "1) Quick Summary (5 bullets)\n"
        "2) Positives (bullets)\n"
        "3) Issues / Risks (bullets)\n"
        "4) Room-wise Suggestions (Kitchen, Bedrooms, Toilets, Living, Pooja, Entrance)\n"
        "5) Easy Remedies (low-cost, non-structural first)\n"
        "6) Questions to confirm (if anything is unclear)\n\n"
        "Be clear, specific, and avoid superstition-heavy language. If directions are unclear, say so and give best-effort guidance."
    )
    if extra and extra.strip():
        base += f"\n\nAdditional user information:\n{extra.strip()}"
    return base

def run_openai(uploaded_file, extra_info):
    filename = uploaded_file.name
    file_bytes = uploaded_file.getvalue()
    mime = uploaded_file.type or mimetypes.guess_type(filename)[0] or "application/octet-stream"
    prompt_text = build_user_prompt(extra_info)

    # PDF: upload to Files API, then reference via input_file
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
        return resp.output_text

    # Image: base64 data URL
    b64 = base64.b64encode(file_bytes).decode("utf-8")
    data_url = f"data:{mime};base64,{b64}"

    resp = client.responses.create(
        model=MODEL,
        max_output_tokens=MAX_OUTPUT_TOKENS,
        input=[{
            "role": "user",
            "content": [
                {"type": "input_text", "text": prompt_text},
                {"type": "input_image", "image_url": data_url},
            ],
        }],
    )
    return resp.output_text

if show:
    st.session_state["err"] = ""
    st.session_state["out"] = ""

    if not uploaded:
        st.session_state["err"] = "Please upload a floor plan (PDF or image)."
    else:
        with st.spinner("Analyzing floor plan..."):
            try:
                st.session_state["out"] = run_openai(uploaded, additional_info)
            except Exception as e:
                st.session_state["err"] = str(e)

st.markdown("---")
st.subheader("Vastu Suggestions")

if st.session_state["err"]:
    st.error(st.session_state["err"])

if st.session_state["out"]:
    st.write(st.session_state["out"])
