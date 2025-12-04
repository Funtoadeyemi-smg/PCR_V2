import base64
import os
import tempfile
from typing import Dict, List

import streamlit as st

from utils.dataextractor import ConsolidatedDataExtractor
from utils.powerpointprocessor import PowerPointProcessor

DEFAULT_CAMPAIGN_OBJECTIVE = (
    ""
)
DEFAULT_PRIMARY_KPIS = ""
DEFAULT_SECONDARY_KPIS = ""

st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

    :root {
        --primary: #2563eb;
        --accent: #22d3ee;
        --surface: rgba(15, 23, 42, 0.6);
        --border: rgba(148, 163, 184, 0.28);
    }

    .stApp {
        background: radial-gradient(circle at 20% 20%, rgba(56, 189, 248, 0.25), transparent 45%),
                    radial-gradient(circle at 80% 0%, rgba(59, 130, 246, 0.22), transparent 40%),
                    #0b1120;
        color: #e2e8f0;
        font-family: 'Inter', sans-serif;
    }

    .block-container {
        padding-top: 2.6rem;
        max-width: 1100px;
    }

    .page-header {
        padding: 2.75rem 2.35rem;
        border-radius: 26px;
        border: 1px solid rgba(148, 197, 255, 0.3);
        background: linear-gradient(135deg, rgba(96, 165, 250, 0.18), rgba(30, 64, 175, 0.25));
        box-shadow: 0 30px 60px rgba(15, 23, 42, 0.35);
        margin-bottom: 2rem;
    }

    .page-header h1 {
        font-size: 2.45rem;
        color: #ffffff;
        font-weight: 700;
        letter-spacing: -0.02em;
        margin-bottom: 0.45rem;
    }

    .page-header p {
        font-size: 1rem;
        color: #cbd5f5;
        margin: 0;
    }

    .pill {
        display: inline-flex;
        align-items: center;
        gap: 0.35rem;
        padding: 0.4rem 0.9rem;
        border-radius: 999px;
        background: rgba(37, 99, 235, 0.2);
        color: #bfdbfe;
        text-transform: uppercase;
        font-size: 0.8rem;
        letter-spacing: 0.08em;
        font-weight: 600;
        margin-bottom: 1.1rem;
    }

    .meta-info {
        display: flex;
        flex-wrap: wrap;
        gap: 1.25rem;
        margin-top: 1.9rem;
    }

    .meta-info .item {
        min-width: 200px;
        padding: 0.9rem 1.15rem;
        border-radius: 14px;
        border: 1px solid rgba(148, 163, 184, 0.35);
        background: rgba(15, 23, 42, 0.55);
        color: #94a3b8;
        font-size: 0.85rem;
    }

    .card {
        background: var(--surface);
        border-radius: 18px;
        border: 1px solid var(--border);
        padding: 1.65rem 1.9rem;
        margin-bottom: 1.9rem;
        box-shadow: 0 22px 40px rgba(15, 23, 42, 0.28);
        backdrop-filter: blur(9px);
    }

    .section-title {
        text-transform: uppercase;
        letter-spacing: 0.24em;
        color: #bfdbfe;
        font-size: 1.05rem;
        font-weight: 700;
        margin-bottom: 1.05rem;
    }

    .upload-card {
        border-radius: 16px;
        padding: 1.1rem 1.35rem;
        border: 1px dashed rgba(148, 163, 184, 0.4);
        background: rgba(15, 23, 42, 0.48);
        margin-bottom: 1.1rem;
        transition: transform 0.2s ease, border-color 0.2s ease;
    }

    .upload-card:hover {
        transform: translateY(-2px);
        border-color: rgba(96, 165, 250, 0.65);
    }

    .upload-card h4 {
        font-size: 0.95rem;
        font-weight: 600;
        margin-bottom: 0.45rem;
        color: #bfdbfe;
    }

    .upload-desc {
        font-size: 0.82rem;
        color: #94a3b8;
        margin-bottom: 0.75rem;
    }

    .stFileUploader label {
        display: none;
    }

    .stFileUploader div[data-testid="stFileUploadDropzone"] {
        border-radius: 12px;
        border: 1px solid rgba(148, 163, 184, 0.35);
        background: rgba(15, 23, 42, 0.65);
    }

    .stButton>button {
        background: linear-gradient(135deg, #2563eb, #22d3ee);
        border: none;
        color: #ffffff;
        font-weight: 600;
        letter-spacing: 0.03em;
        padding: 0.8rem 1.85rem;
        border-radius: 12px;
        transition: transform 0.18s ease, box-shadow 0.18s ease;
    }

    .stButton>button:hover {
        transform: translateY(-1px);
        box-shadow: 0 14px 28px rgba(59, 130, 246, 0.35);
    }
    </style>
""", unsafe_allow_html=True)


def load_logo_base64(file_path):
    with open(file_path, "rb") as f:
        return base64.b64encode(f.read()).decode()

logo_base64 = load_logo_base64("smg2.jpeg")
st.markdown(
    f"""
    <div style="text-align: center;">
        <img src="data:image/png;base64,{logo_base64}" width="100">
    </div>
    """,
    unsafe_allow_html=True,
)



# Initialize session state variables if they don't exist
if "reset_state" not in st.session_state:
    st.session_state.reset_state = False

if "meta_file" not in st.session_state:
    st.session_state.meta_file = None

if "pinterest_file" not in st.session_state:
    st.session_state.pinterest_file = None

if "tiktok_files" not in st.session_state:
    st.session_state.tiktok_files: List = []

if "media_plan_file" not in st.session_state:
    st.session_state.media_plan_file = None

if "campaign_objective" not in st.session_state:
    st.session_state.campaign_objective = DEFAULT_CAMPAIGN_OBJECTIVE

if "primary_kpis" not in st.session_state:
    st.session_state.primary_kpis = DEFAULT_PRIMARY_KPIS

if "secondary_kpis" not in st.session_state:
    st.session_state.secondary_kpis = DEFAULT_SECONDARY_KPIS

if "primary_kpis_selected" not in st.session_state:
    st.session_state.primary_kpis_selected: List[str] = []

if "secondary_kpis_selected" not in st.session_state:
    st.session_state.secondary_kpis_selected: List[str] = []

if "toc_image" not in st.session_state:
    st.session_state.toc_image = None

if "summary_image" not in st.session_state:
    st.session_state.summary_image = None


st.markdown("""
<div class="page-header">
    <div class="pill">PCR Automation</div>
    <h1>Post-Campaign Report Builder</h1>
    <p>Provide the campaign exports, define the objectives, and generate a consolidated post-campaign presentation in minutes.</p>
    <div class="meta-info">
        <div class="item">Required &mdash; Meta performance export (.xlsx)</div>
        <div class="item">Optional &mdash; Pinterest/TikTok performance and media-plan estimates</div>
        <div class="item">Optional &mdash; Table of Contents and Campaign Summary imagery</div>
    </div>
</div>
""", unsafe_allow_html=True)

with st.container():
    st.markdown("<div class='card'><div class='section-title'>Campaign Objectives & KPIs</div>", unsafe_allow_html=True)
    st.markdown(
        "<p style='color:#94a3b8;font-size:0.92rem;margin-bottom:1rem;'>Provide a concise statement of the campaign objective and select the primary and secondary KPIs that will be referenced in the generated commentary.</p>",
        unsafe_allow_html=True,
    )
    st.session_state.campaign_objective = st.text_area(
        "Campaign Objective",
        value=st.session_state.campaign_objective,
        help="Describe the overarching objective for the campaign.",
        placeholder="e.g. Drive awareness of new product launches and generate measurable sales uplift.",
    )

    kpi_options = [
        "Unique Reach",
        "CPM (Cost per Impressions)",
        "Impressions",
        "ROI (Return on Investment)",
        "ROAS (Return on Ad Spend)",
        "Frequency",
        "CTR (Click Through Rate)",
    ]

    primary_selection = st.multiselect(
        "Primary KPI(s)",
        options=kpi_options,
        default=st.session_state.primary_kpis_selected,
        help="Select the KPIs that are considered the primary success measures for this campaign.",
    )
    st.session_state.primary_kpis_selected = primary_selection
    st.session_state.primary_kpis = ", ".join(primary_selection) if primary_selection else ""

    secondary_selection = st.multiselect(
        "Secondary KPI(s)",
        options=kpi_options,
        default=st.session_state.secondary_kpis_selected,
        help="Select supporting KPIs that provide additional context.",
    )
    st.session_state.secondary_kpis_selected = secondary_selection
    st.session_state.secondary_kpis = ", ".join(secondary_selection) if secondary_selection else ""

    st.markdown("</div>", unsafe_allow_html=True)

# Handle file uploads with unique keys that change when reset is pressed
upload_key_suffix = f"_{hash(st.session_state.reset_state)}"

with st.container():
    st.markdown("<div class='card'><div class='section-title'>Upload Performance Data</div>", unsafe_allow_html=True)
    upload_cols = st.columns(2)

    with upload_cols[0]:
        st.markdown("<div class='upload-card'><h4>Meta Campaign (Required)</h4><div class='upload-desc'>Upload the consolidated Meta performance export (.xlsx).</div>", unsafe_allow_html=True)
        meta_file = st.file_uploader(
            "Meta Performance File (Required)",
            type=["xlsx", "csv"],
            key=f"meta_uploader{upload_key_suffix}",
        )
        st.markdown("</div>", unsafe_allow_html=True)

    with upload_cols[1]:
        st.markdown("<div class='upload-card'><h4>Media Plan (Optional)</h4><div class='upload-desc'>Include the planning sheet to compare estimated vs actual performance.</div>", unsafe_allow_html=True)
        media_plan_file = st.file_uploader(
            "Media Plan File (Optional)",
            type=["xlsx", "csv"],
            key=f"media_plan_uploader{upload_key_suffix}",
        )
        st.markdown("</div>", unsafe_allow_html=True)

    upload_cols_2 = st.columns(2)

    with upload_cols_2[0]:
        st.markdown("<div class='upload-card'><h4>Pinterest Performance (Optional)</h4><div class='upload-desc'>CSV export from Pinterest Ads Manager.</div>", unsafe_allow_html=True)
        pinterest_file = st.file_uploader(
            "Pinterest Performance File (Optional)",
            type=["csv", "xlsx"],
            key=f"pinterest_uploader{upload_key_suffix}",
        )
        st.markdown("</div>", unsafe_allow_html=True)

    with upload_cols_2[1]:
        st.markdown("<div class='upload-card'><h4>TikTok Performance (Optional)</h4><div class='upload-desc'>Upload the ad and/or audience level Excel/CSV reports.</div>", unsafe_allow_html=True)
        tiktok_files = st.file_uploader(
            "TikTok Files (Optional – upload ad and/or audience reports)",
            type=["csv", "xlsx"],
            accept_multiple_files=True,
            key=f"tiktok_uploader{upload_key_suffix}",
        )
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)

with st.container():
    st.markdown("<div class='card'><div class='section-title'>Brand Imagery</div>", unsafe_allow_html=True)
    st.markdown(
        "<p style='color:#94a3b8;font-size:0.9rem;margin-bottom:1rem;'>Optional hero images to personalise the Table of Contents and Campaign Summary slides.</p>",
        unsafe_allow_html=True,
    )
    image_cols = st.columns(2)
    with image_cols[0]:
        st.markdown("<div class='upload-card'><h4>Table of Contents image</h4>", unsafe_allow_html=True)
        toc_upload = st.file_uploader(
            "Table of contents image",
            type=["png", "jpg", "jpeg"],
            key=f"toc_image_uploader{upload_key_suffix}",
        )
        if toc_upload is not None:
            st.session_state.toc_image = toc_upload
        st.markdown("</div>", unsafe_allow_html=True)

    with image_cols[1]:
        st.markdown("<div class='upload-card'><h4>Campaign Summary image</h4>", unsafe_allow_html=True)
        summary_upload = st.file_uploader(
            "Campaign summary image",
            type=["png", "jpg", "jpeg"],
            key=f"summary_image_uploader{upload_key_suffix}",
        )
        if summary_upload is not None:
            st.session_state.summary_image = summary_upload
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)

# Store uploaded files in session state
if meta_file is not None:
    st.session_state.meta_file = meta_file

if pinterest_file is not None:
    st.session_state.pinterest_file = pinterest_file

if tiktok_files:
    st.session_state.tiktok_files = tiktok_files

if media_plan_file is not None:
    st.session_state.media_plan_file = media_plan_file

col1, col2, col3, col4 = st.columns(4)

with col1:
    if st.button("Generate PowerPoint Report"):
        with tempfile.TemporaryDirectory() as tmpdir:
            # Resolve Meta file (required)
            meta_path = None
            if st.session_state.meta_file:
                meta_path = os.path.join(tmpdir, st.session_state.meta_file.name)
                st.session_state.meta_file.seek(0)
                with open(meta_path, "wb") as f:
                    f.write(st.session_state.meta_file.read())
            else:
                st.error("Please upload the Meta performance export (.xlsx) before generating the report.")
                st.stop()

            # Save uploaded files temporarily
            pin_path = None
            if st.session_state.pinterest_file:
                pin_path = os.path.join(tmpdir, st.session_state.pinterest_file.name)
                st.session_state.pinterest_file.seek(0)
                with open(pin_path, "wb") as f:
                    f.write(st.session_state.pinterest_file.read())

            tiktok_paths: List[str] = []
            if st.session_state.tiktok_files:
                for uploaded in st.session_state.tiktok_files:
                    file_path = os.path.join(tmpdir, uploaded.name)
                    uploaded.seek(0)
                    with open(file_path, "wb") as f:
                        f.write(uploaded.read())
                    tiktok_paths.append(file_path)

            media_path = None
            if st.session_state.media_plan_file:
                media_path = os.path.join(tmpdir, st.session_state.media_plan_file.name)
                st.session_state.media_plan_file.seek(0)
                with open(media_path, "wb") as f:
                    f.write(st.session_state.media_plan_file.read())

            # Provide internal file paths (ensure these files exist in your project directory)
            prompt = "prompt.txt"
            ppt_template = "consolidated_template.pptx"

            prompt_path = os.path.join(tmpdir, "prompt.txt")
            with open(prompt, "rb") as src, open(prompt_path, "wb") as dst:
                dst.write(src.read())

            ppt_template_path = os.path.join(tmpdir, "template.pptx")
            with open(ppt_template, "rb") as src, open(ppt_template_path, "wb") as dst:
                dst.write(src.read())

            image_placeholders: Dict[str, str] = {}
            if st.session_state.toc_image is not None:
                toc_path = os.path.join(tmpdir, st.session_state.toc_image.name)
                st.session_state.toc_image.seek(0)
                with open(toc_path, "wb") as f:
                    f.write(st.session_state.toc_image.read())
                image_placeholders["{table_of_contents_picture}"] = toc_path
            if st.session_state.summary_image is not None:
                summary_path = os.path.join(tmpdir, st.session_state.summary_image.name)
                st.session_state.summary_image.seek(0)
                with open(summary_path, "wb") as f:
                    f.write(st.session_state.summary_image.read())
                image_placeholders["{campaign_summary_picture}"] = summary_path

            # Run processing
            with st.spinner("Generating your PowerPoint report..."):
                extractor = ConsolidatedDataExtractor(
                    meta_file=meta_path,
                    pinterest_file=pin_path,
                    tiktok_files=tiktok_paths,
                    media_plan_file=media_path,
                    prompt_template=prompt_path,
                    template_path=ppt_template,
                    manual_campaign_objective=st.session_state.campaign_objective,
                    manual_primary_kpis=st.session_state.primary_kpis,
                    manual_secondary_kpis=st.session_state.secondary_kpis,
                )
                replacements, channels_present = extractor.extract_values()

                output_path = os.path.join(tmpdir, "automated_presentation.pptx")
                ppt = PowerPointProcessor(ppt_template_path)
                ppt.replace_placeholders(
                    replacements,
                    channels_present,
                    output_path,
                    image_placeholders=image_placeholders,
                )

            # Download output
            with open(output_path, "rb") as file:
                st.success("✅ PowerPoint report generated!")
                st.download_button(
                    label="📥 Download Report",
                    data=file,
                    file_name="automated_presentation.pptx",
                    mime="application/vnd.openxmlformats-officedocument.presentationml.presentation"
                )
with col4:
    if st.button("Reset Uploads"):
        # Toggle the reset state to force file uploader widgets to create new instances
        st.session_state.reset_state = not st.session_state.reset_state
        # Clear stored files
        st.session_state.meta_file = None
        st.session_state.pinterest_file = None
        st.session_state.tiktok_files = []
        st.session_state.media_plan_file = None
        st.session_state.campaign_objective = DEFAULT_CAMPAIGN_OBJECTIVE
        st.session_state.primary_kpis = DEFAULT_PRIMARY_KPIS
        st.session_state.secondary_kpis = DEFAULT_SECONDARY_KPIS
        st.session_state.primary_kpis_selected = []
        st.session_state.secondary_kpis_selected = []
        st.session_state.toc_image = None
        st.session_state.summary_image = None
        st.rerun()
