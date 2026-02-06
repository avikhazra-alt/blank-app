import io
import base64
import mimetypes
import streamlit as st
from openai import OpenAI

st.set_page_config(page_title="Prompt Sandbox", layout="wide")

# ---- Fixed settings (not shown in UI) ----
MODEL = "gpt-4o-mini"
MAX_OUTPUT_TOKENS = 1200

st.title("📎 Upload → Prompt → Output")
st.caption("Upload a PDF or image, enter a prompt, and get text output on the UI.")

# Session state
st.session_state.setdefault("out", "")
st.session_state.setdefault("err", "")

client = OpenAI()  # uses OPENAI_API_KEY from environment/secrets

col1, col2 = st.columns([1, 1], gap="large")

with col1:
    st.subheader("1) Upload")
    uploaded = st.file_uploader("PDF / PNG / JPG / WEBP", type=["pdf", "png", "jpg", "jpeg", "webp"])

    if uploaded:
        name = uploaded.name.lower()
        if name.endswith(".pdf"):
            st.info(f"📄 PDF uploaded: **{uploaded.name}**")
        else:
            st.image(uploaded, caption=uploaded.name, use_container_width=True)

with col2:
    st.subheader("2) Prompt")
    prompt = st.text_area(
        "What should I do with this file?",
        placeholder="Examples:\n- Summarize this PDF in 5 bullets\n- Extract key data\n- Describe the image and list visible text",
        height=180,
    )

    c1, c2 = st.columns([1, 1])
    run = c1.button("▶️ Run", type="primary", use_container_width=True)
    clear = c2.button("🧹 Clear", use_container_width=True)

if clear:
    st.session_state["out"] = ""
    st.session_state["err"] = ""

def run_openai(uploaded_file, prompt_text):
    filename = uploaded_file.name
    file_bytes = uploaded_file.getvalue()
    mime = uploaded_file.type or mimetypes.guess_type(filename)[0] or "application/octet-stream"

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

    # image
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

if run:
    st.session_state["err"] = ""
    st.session_state["out"] = ""

    if not uploaded:
        st.session_state["err"] = "Please upload a PDF or image."
    elif not prompt.strip():
        st.session_state["err"] = "Please enter a prompt."
    else:
        with st.spinner("Processing..."):
            try:
                st.session_state["out"] = run_openai(uploaded, prompt.strip())
            except Exception as e:
                st.session_state["err"] = str(e)

st.markdown("---")
st.subheader("3) Output")

if st.session_state["err"]:
    st.error(st.session_state["err"])

if st.session_state["out"]:
    st.write(st.session_state["out"])
