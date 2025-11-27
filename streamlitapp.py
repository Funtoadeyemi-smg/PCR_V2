from typing import List

import base64
import os
import tempfile

import streamlit as st
from dotenv import load_dotenv

from utils.dataextractor import ConsolidatedDataExtractor
from utils.powerpointprocessor import PowerPointProcessor

load_dotenv()

st.set_page_config(page_title="Post-Campaign Report Automation", layout="wide")


def load_logo_base64(path: str) -> str:
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode()


def reset_session_state() -> None:
    st.session_state.meta_file = None
    st.session_state.pinterest_file = None
    st.session_state.tiktok_files = []
    st.session_state.media_plan_file = None


if "meta_file" not in st.session_state:
    reset_session_state()
    st.session_state.reset_state = False

logo_base64 = load_logo_base64("smg2.jpeg")

st.markdown(
    """
    <style>
    .stApp { background-color: #000; color: #fff; }
    .block-container { color: #fff; }
    label { background-color: #e1ad01; color: #000 !important; }
    .stButton>button { background-color: #e1ad01; color: #000; font-weight: bold; border: none; }
    .stFileUploader label span { color: #fff; }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    f"""
    <div style="text-align:center;">
        <img src="data:image/png;base64,{logo_base64}" width="110">
    </div>
    """,
    unsafe_allow_html=True,
)

st.title("📊 Post-Campaign Report Automation")
st.markdown(
    "Upload your campaign datasets. Meta data is required; Pinterest, TikTok, and a media-plan file are optional."
)

upload_suffix = f"_{hash(st.session_state.reset_state)}"

meta_file = st.file_uploader(
    "Meta Report (xlsx, required)",
    type=["xlsx"],
    key=f"meta_{upload_suffix}",
)

pinterest_file = st.file_uploader(
    "Pinterest Report (csv/xlsx, optional)",
    type=["csv", "xlsx"],
    key=f"pin_{upload_suffix}",
)

tiktok_files = st.file_uploader(
    "TikTok Files – Audience and/or Ad level (csv/xlsx, optional)",
    type=["csv", "xlsx"],
    accept_multiple_files=True,
    key=f"tik_{upload_suffix}",
)

media_plan_file = st.file_uploader(
    "Media Plan (xlsx, optional)",
    type=["xlsx"],
    key=f"plan_{upload_suffix}",
)

if meta_file:
    st.session_state.meta_file = meta_file

if pinterest_file:
    st.session_state.pinterest_file = pinterest_file

if tiktok_files:
    st.session_state.tiktok_files = tiktok_files

if media_plan_file:
    st.session_state.media_plan_file = media_plan_file

col1, col2 = st.columns([3, 1])

with col1:
    if st.button("Generate PowerPoint Report"):
        if not st.session_state.meta_file:
            st.error("Please upload a Meta report to continue.")
        else:
            with tempfile.TemporaryDirectory() as tmpdir:
                # Persist uploads
                meta_path = os.path.join(tmpdir, st.session_state.meta_file.name)
                st.session_state.meta_file.seek(0)
                with open(meta_path, "wb") as f:
                    f.write(st.session_state.meta_file.read())

                pin_path = None
                if st.session_state.pinterest_file:
                    pin_path = os.path.join(tmpdir, st.session_state.pinterest_file.name)
                    st.session_state.pinterest_file.seek(0)
                    with open(pin_path, "wb") as f:
                        f.write(st.session_state.pinterest_file.read())

                tik_paths: List[str] = []
                for uploaded in st.session_state.tiktok_files or []:
                    path = os.path.join(tmpdir, uploaded.name)
                    uploaded.seek(0)
                    with open(path, "wb") as f:
                        f.write(uploaded.read())
                    tik_paths.append(path)

                plan_path = None
                if st.session_state.media_plan_file:
                    plan_path = os.path.join(tmpdir, st.session_state.media_plan_file.name)
                    st.session_state.media_plan_file.seek(0)
                    with open(plan_path, "wb") as f:
                        f.write(st.session_state.media_plan_file.read())

                prompt_src = "prompt.txt"
                template_src = "consolidated_template.pptx"

                prompt_path = os.path.join(tmpdir, "prompt.txt")
                with open(prompt_src, "rb") as src, open(prompt_path, "wb") as dst:
                    dst.write(src.read())

                template_path = os.path.join(tmpdir, "template.pptx")
                with open(template_src, "rb") as src, open(template_path, "wb") as dst:
                    dst.write(src.read())

                with st.spinner("Generating PowerPoint..."):
                    extractor = ConsolidatedDataExtractor(
                        meta_file=meta_path,
                        pinterest_file=pin_path,
                        tiktok_files=tik_paths,
                        media_plan_file=plan_path,
                        prompt_template=prompt_path,
                        template_path=template_src,
                    )
                    replacements, channels_present = extractor.extract_values()

                    output_path = os.path.join(tmpdir, "post_campaign_report.pptx")
                    ppt = PowerPointProcessor(template_path)
                    ppt.replace_placeholders(replacements, channels_present, output_path)

                with open(output_path, "rb") as file:
                    st.success("✅ Report ready!")
                    st.download_button(
                        "📥 Download Report",
                        file,
                        file_name="post_campaign_report.pptx",
                        mime="application/vnd.openxmlformats-officedocument.presentationml.presentation",
                    )

with col2:
    if st.button("Reset Uploads"):
        st.session_state.reset_state = not st.session_state.reset_state
        reset_session_state()
        st.experimental_rerun()
