import base64
import os
import re
import tempfile
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import pandas as pd
import streamlit as st
from pydantic import ValidationError

from utils.dataextractor import (
    REQUIRED_COLUMNS,
    ConsolidatedDataExtractor,
    detect_tiktok_file_role,
    identify_trailing_summary_row,
)
from utils.powerpointprocessor import PowerPointProcessor

DEFAULT_CAMPAIGN_OBJECTIVE = (
    ""
)
DEFAULT_PRIMARY_KPIS = ""
DEFAULT_SECONDARY_KPIS = ""

ASSET_DIR = Path("prompts_artefacts")
PROMPT_FILE = ASSET_DIR / "prompt.txt"
TEMPLATE_FILE = ASSET_DIR / "powerpoint_template.pptx"
LOGO_FILE = ASSET_DIR / "smg2.jpeg"

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

CHANNEL_LABELS = {"meta": "Meta", "pin": "Pinterest", "tik": "TikTok"}
OPTIONAL_CHANNELS = ["pin", "tik"]
SUMMARY_PAGE_NUMBERS = {"overall": 7, "meta": 13, "pin": 21, "tik": 29}

OVERALL_PREVIEW_FIELDS = [
    ("Gross spend", "gross_spend", "gross_est"),
    ("Impressions", "impressions", "imp_est"),
    ("Reach", "reach", None),
    ("Clicks", "clicks", "click_est"),
    ("CTR", "ctr", "ctr_est"),
    ("Net CPM", "net_cpm", "net_cpm_est"),
    ("Frequency", "freq_est", None),
    ("Brand revenue", "brand_rev", None),
    ("Brand ROAS", "brand_roas", None),
    ("Brand ROI", "brand_roi", None),
]

OVERALL_DELTA_FIELDS = [
    ("Impressions vs plan", "perc_imp"),
    ("Clicks vs plan", "perc_clicks"),
]

CHANNEL_PREVIEW_FIELDS = [
    ("Gross spend", "gross_spend", "est_spend"),
    ("Net spend", "net_spend", None),
    ("Impressions", "impressions", "est_imp"),
    ("Reach", "reach", "est_reach"),
    ("Clicks", "clicks", "est_clicks"),
    ("CTR", "ctr", "est_ctr"),
    ("Net CPM", "net_cpm", "est_cpm"),
    ("Frequency", "freq", "est_freq"),
    ("Brand revenue", "brand_revenue", None),
    ("Brand ROAS", "brand_roas", None),
    ("Brand ROI", "brand_roi", None),
]


def load_dataframe_from_upload(uploaded_file, required_key: Optional[str] = None) -> Optional[pd.DataFrame]:
    if uploaded_file is None:
        return None
    try:
        uploaded_file.seek(0)
    except Exception:
        return None
    suffix = Path(uploaded_file.name).suffix.lower()
    try:
        if suffix in {".csv", ".txt"}:
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_excel(uploaded_file)
    except Exception:
        uploaded_file.seek(0)
        return None
    finally:
        uploaded_file.seek(0)

    if required_key:
        required_columns = {col.lower() for col in REQUIRED_COLUMNS.get(required_key, set())}
        detected = {str(col).strip().lower() for col in df.columns}
        if required_columns and not required_columns.issubset(detected) and suffix not in {".csv", ".txt"}:
            try:
                preview = pd.read_excel(uploaded_file, header=None, nrows=60)
                uploaded_file.seek(0)
                header_row = 0
                for idx, row in preview.iterrows():
                    normalized = {
                        re.sub(r"\s+", " ", str(value).strip().lower())
                        for value in row.tolist()
                        if pd.notna(value) and str(value).strip()
                    }
                    if required_columns.issubset(normalized):
                        header_row = idx
                        break
                df = pd.read_excel(uploaded_file, header=header_row)
            except Exception:
                uploaded_file.seek(0)
                pass
            finally:
                uploaded_file.seek(0)
    return df


def _build_rename_map_local(mapping: Dict[str, str], columns: Iterable[str]) -> Dict[str, str]:
    rename_map: Dict[str, str] = {}
    column_set = set(columns)
    for target, original in mapping.items():
        if original and original in column_set:
            rename_map[original] = target
    return rename_map


def render_preflight_report(
    title: str,
    df: pd.DataFrame,
    required_key: str,
    mapping_key: str,
) -> None:
    mapping = st.session_state.column_mappings.setdefault(mapping_key, {})
    rename_map = _build_rename_map_local(mapping, df.columns)
    df_mapped = df.rename(columns=rename_map)
    renamed_columns = {col.strip() for col in df_mapped.columns}
    required_columns = {col.strip() for col in REQUIRED_COLUMNS.get(required_key, set())}
    missing_columns = sorted(required_columns - renamed_columns)

    with st.expander(f"{title} ({len(df)} rows, {len(df.columns)} columns)", expanded=False):
        st.markdown("**Detected columns**")
        st.dataframe(pd.DataFrame({"Column": df.columns}), use_container_width=True)

        if missing_columns:
            st.warning(
                "Missing required columns after applying mappings: "
                + ", ".join(missing_columns)
            )
        else:
            st.success("All required columns detected.")

        for required_col in sorted(required_columns):
            current_original = mapping.get(required_col, "(Not mapped)")
            options = ["(Not mapped)"] + sorted(df.columns)
            if current_original not in options:
                current_original = "(Not mapped)"
            show_selector = (required_col in missing_columns) or (current_original != "(Not mapped)")
            if show_selector:
                selection = st.selectbox(
                    f"Map column to `{required_col}`",
                    options=options,
                    index=options.index(current_original),
                    key=f"map_{mapping_key}_{required_col}",
                )
                if selection == "(Not mapped)":
                    mapping.pop(required_col, None)
                else:
                    mapping[required_col] = selection

        processing_notes: List[str] = []
        if mapping_key in {"tik_audience", "tik_ad"}:
            key_candidates = ["Ad group name"] if mapping_key == "tik_audience" else ["Ad", "Ad name"]
            summary_label: Optional[str] = None
            trimmed_df = df_mapped
            for candidate in key_candidates:
                summary_info = identify_trailing_summary_row(df_mapped, candidate)
                if summary_info:
                    index, label = summary_info
                    summary_label = label
                    trimmed_df = df_mapped.iloc[:index]
                    break
            if summary_label:
                processing_notes.append(
                    f"Trailing summary row `{summary_label}` will be ignored during aggregation."
                )
            if mapping_key == "tik_ad" and {"Cost", "Impressions"}.issubset(trimmed_df.columns):
                cost = pd.to_numeric(trimmed_df["Cost"], errors="coerce").fillna(0)
                impressions = pd.to_numeric(trimmed_df["Impressions"], errors="coerce").fillna(0)
                skipped_rows = int(((cost <= 0) | (impressions <= 0)).sum())
                if skipped_rows:
                    processing_notes.append(
                        f"{skipped_rows} ad row(s) without spend or impressions will be skipped when identifying leading creatives."
                    )

        st.markdown("**Preview (first 5 rows)**")
        st.dataframe(df.head(), use_container_width=True)
        if processing_notes:
            st.markdown("**Processing notes**")
            for note in processing_notes:
                st.markdown(f"- {note}")


def build_summary_preview(replacements: Dict[str, str], channels: List[str]) -> Dict[str, Dict[str, List[Dict[str, str]]]]:
    def value_for(key: Optional[str]) -> Optional[str]:
        if not key:
            return None
        return replacements.get(f"{{{key}}}")

    def normalise(value: Optional[str], dash_when_missing: bool = False) -> str:
        if value is None or value == "":
            return "—" if dash_when_missing else "N/A"
        return value

    overall_rows: List[Dict[str, str]] = []
    for label, actual_key, plan_key in OVERALL_PREVIEW_FIELDS:
        actual_val = value_for(actual_key)
        plan_val = value_for(plan_key) if plan_key else None
        if actual_val is None and plan_val is None:
            continue
        overall_rows.append(
            {
                "Metric": label,
                "Actual": normalise(actual_val),
                "Plan": normalise(plan_val, dash_when_missing=plan_key is not None),
            }
        )
    for label, delta_key in OVERALL_DELTA_FIELDS:
        delta_val = value_for(delta_key)
        if delta_val:
            overall_rows.append(
                {
                    "Metric": label,
                    "Actual": normalise(delta_val),
                    "Plan": "—",
                }
            )

    channel_rows: Dict[str, List[Dict[str, str]]] = {}
    for prefix in channels:
        rows: List[Dict[str, str]] = []
        for label, actual_key, plan_key in CHANNEL_PREVIEW_FIELDS:
            actual_val = replacements.get(f"{{{prefix}_{actual_key}}}")
            plan_val = replacements.get(f"{{{prefix}_{plan_key}}}") if plan_key else None
            if actual_val is None and plan_val is None:
                continue
            rows.append(
                {
                    "Metric": label,
                    "Actual": normalise(actual_val),
                    "Plan": normalise(plan_val, dash_when_missing=plan_key is not None),
                }
            )
        channel_rows[prefix] = rows

    return {"overall": overall_rows, "channels": channel_rows}


def render_report_preview(preview: Optional[Dict[str, Dict[str, List[Dict[str, str]]]]], channels: List[str]) -> None:
    st.markdown("<div class='card'><div class='section-title'>Report Preview</div>", unsafe_allow_html=True)
    if not preview:
        st.info("Generate a report to preview the summary slides.")
        st.markdown("</div>", unsafe_allow_html=True)
        return

    overall_rows = preview.get("overall", [])
    if overall_rows:
        page = SUMMARY_PAGE_NUMBERS.get("overall")
        st.markdown(f"**Overall summary (page {page})**" if page else "**Overall summary**")
        st.table(pd.DataFrame(overall_rows))

    for prefix in channels:
        channel_rows = preview.get("channels", {}).get(prefix, [])
        if not channel_rows:
            continue
        page = SUMMARY_PAGE_NUMBERS.get(prefix)
        channel_name = CHANNEL_LABELS.get(prefix, prefix.title())
        title = f"**{channel_name} summary (page {page})**" if page else f"**{channel_name} summary**"
        st.markdown(title)
        st.table(pd.DataFrame(channel_rows))

    st.markdown("</div>", unsafe_allow_html=True)


def determine_tiktok_role(filename: str, df: pd.DataFrame) -> str:
    try:
        return detect_tiktok_file_role(df)
    except ValueError:
        audience_map = _build_rename_map_local(
            st.session_state.column_mappings.get("tik_audience", {}),
            df.columns,
        )
        if audience_map:
            try:
                return detect_tiktok_file_role(df.rename(columns=audience_map))
            except ValueError:
                pass
        ad_map = _build_rename_map_local(
            st.session_state.column_mappings.get("tik_ad", {}),
            df.columns,
        )
        if ad_map:
            try:
                return detect_tiktok_file_role(df.rename(columns=ad_map))
            except ValueError:
                pass

        role_labels = {
            "audience": "Audience-level report (Ad group performance)",
            "ad": "Ad-level report",
        }
        default_choice = st.session_state.tiktok_file_roles.get(filename, "audience")
        options = list(role_labels.keys())
        index = options.index(default_choice) if default_choice in options else 0
        selection = st.selectbox(
            f"Select TikTok file type for `{filename}`",
            options=options,
            index=index,
            format_func=lambda key: role_labels[key],
            key=f"tik_role_{filename}",
        )
        return selection


def load_logo_base64(file_path: Path) -> Optional[str]:
    try:
        with file_path.open("rb") as f:
            return base64.b64encode(f.read()).decode()
    except FileNotFoundError:
        return None


logo_base64 = load_logo_base64(LOGO_FILE)
if not logo_base64:
    st.warning("Logo asset not found in `prompts_artefacts`. The header logo will be hidden.")
else:
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

if "generated_report" not in st.session_state:
    st.session_state.generated_report = None

if "primary_kpi_widget_version" not in st.session_state:
    st.session_state.primary_kpi_widget_version = 0

if "secondary_kpi_widget_version" not in st.session_state:
    st.session_state.secondary_kpi_widget_version = 0

if "column_mappings" not in st.session_state:
    st.session_state.column_mappings = {
        "meta": {},
        "pinterest": {},
        "tik_audience": {},
        "tik_ad": {},
        "media_plan": {},
    }

if "tiktok_file_roles" not in st.session_state:
    st.session_state.tiktok_file_roles = {}

if "selected_optional_channels" not in st.session_state:
    st.session_state.selected_optional_channels = []

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

    primary_key = f"primary_kpi_selector_{st.session_state.primary_kpi_widget_version}"
    secondary_key = f"secondary_kpi_selector_{st.session_state.secondary_kpi_widget_version}"

    if primary_key not in st.session_state:
        st.session_state[primary_key] = st.session_state.primary_kpis_selected
    if secondary_key not in st.session_state:
        st.session_state[secondary_key] = st.session_state.secondary_kpis_selected

    def _on_primary_change(key: str = primary_key):
        selected = st.session_state.get(key, [])
        st.session_state.primary_kpis_selected = selected
        st.session_state.primary_kpis = ", ".join(selected) if selected else ""
        st.session_state.primary_kpi_widget_version += 1
        st.session_state.pop(key, None)
        st.rerun()

    def _on_secondary_change(key: str = secondary_key):
        selected = st.session_state.get(key, [])
        st.session_state.secondary_kpis_selected = selected
        st.session_state.secondary_kpis = ", ".join(selected) if selected else ""
        st.session_state.secondary_kpi_widget_version += 1
        st.session_state.pop(key, None)
        st.rerun()

    st.multiselect(
        "Primary KPI(s)",
        options=kpi_options,
        key=primary_key,
        help="Select the KPIs that are considered the primary success measures for this campaign.",
        on_change=_on_primary_change,
    )

    st.multiselect(
        "Secondary KPI(s)",
        options=kpi_options,
        key=secondary_key,
        help="Select supporting KPIs that provide additional context.",
        on_change=_on_secondary_change,
    )

    # Ensure textual values stay in sync even if no change callback fired during this run
    st.session_state.primary_kpis = ", ".join(st.session_state.primary_kpis_selected) if st.session_state.primary_kpis_selected else ""
    st.session_state.secondary_kpis = (
        ", ".join(st.session_state.secondary_kpis_selected) if st.session_state.secondary_kpis_selected else ""
    )

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
if st.session_state.tiktok_files:
    current_names = {uploaded.name for uploaded in st.session_state.tiktok_files}
    st.session_state.tiktok_file_roles = {
        name: role
        for name, role in st.session_state.tiktok_file_roles.items()
        if name in current_names
    }

if media_plan_file is not None:
    st.session_state.media_plan_file = media_plan_file

with st.container():
    st.markdown("<div class='card'><div class='section-title'>Data Pre-flight Checks</div>", unsafe_allow_html=True)

    meta_source = st.session_state.meta_file
    if meta_source:
        df_meta = load_dataframe_from_upload(meta_source, required_key="meta")
        if df_meta is not None:
            render_preflight_report("Meta Performance", df_meta, "meta", "meta")
    else:
        st.info("Upload the Meta performance file to view validation results.")

    if st.session_state.pinterest_file:
        df_pin = load_dataframe_from_upload(st.session_state.pinterest_file, required_key="pinterest")
        if df_pin is not None:
            render_preflight_report("Pinterest Performance", df_pin, "pinterest", "pinterest")

    if st.session_state.tiktok_files:
        for uploaded in st.session_state.tiktok_files:
            df_tik = load_dataframe_from_upload(uploaded)
            if df_tik is None:
                continue
            role = determine_tiktok_role(uploaded.name, df_tik)
            st.session_state.tiktok_file_roles[uploaded.name] = role
            mapping_key = "tik_audience" if role == "audience" else "tik_ad"
            title = f"TikTok ({'Audience' if role == 'audience' else 'Ad'}): {uploaded.name}"
            render_preflight_report(title, df_tik, mapping_key, mapping_key)

    if st.session_state.media_plan_file:
        df_media = load_dataframe_from_upload(st.session_state.media_plan_file, required_key="media_plan")
        if df_media is not None:
            render_preflight_report("Media Plan", df_media, "media_plan", "media_plan")

    st.markdown("</div>", unsafe_allow_html=True)

with st.container():
    st.markdown("<div class='card'><div class='section-title'>Report Sections</div>", unsafe_allow_html=True)
    available_optional = [
        ch
        for ch in OPTIONAL_CHANNELS
        if (ch == "pin" and st.session_state.pinterest_file)
        or (ch == "tik" and st.session_state.tiktok_files)
    ]
    if available_optional:
        if not st.session_state.selected_optional_channels:
            st.session_state.selected_optional_channels = available_optional.copy()
        else:
            st.session_state.selected_optional_channels = [
                ch for ch in st.session_state.selected_optional_channels if ch in available_optional
            ]
            if not st.session_state.selected_optional_channels:
                st.session_state.selected_optional_channels = available_optional.copy()
        if len(available_optional) > 1:
            selection = st.multiselect(
                "Optional channel sections to include in the PowerPoint",
                options=available_optional,
                default=st.session_state.selected_optional_channels,
                format_func=lambda ch: CHANNEL_LABELS.get(ch, ch.title()),
            )
            st.session_state.selected_optional_channels = selection
        else:
            st.session_state.selected_optional_channels = available_optional.copy()
            channel_name = CHANNEL_LABELS.get(available_optional[0], available_optional[0].title())
            st.markdown(f"<p style='color:#94a3b8;'>Including {channel_name} section automatically.</p>", unsafe_allow_html=True)
    else:
        st.markdown("<p style='color:#94a3b8;'>Only Meta data provided; optional channel sections are not available.</p>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

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
            prompt = PROMPT_FILE
            ppt_template = TEMPLATE_FILE

            missing_assets = [str(path.name) for path in (prompt, ppt_template) if not path.exists()]
            if missing_assets:
                st.error(
                    "Missing required template assets in `prompts_artefacts`: "
                    + ", ".join(missing_assets)
                )
                st.stop()

            prompt_path = os.path.join(tmpdir, "prompt.txt")
            with prompt.open("rb") as src, open(prompt_path, "wb") as dst:
                dst.write(src.read())

            ppt_template_path = os.path.join(tmpdir, "template.pptx")
            with ppt_template.open("rb") as src, open(ppt_template_path, "wb") as dst:
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
            processing_warnings: List[str] = []
            with st.spinner("Generating your PowerPoint report..."):
                extractor = ConsolidatedDataExtractor(
                    meta_file=meta_path,
                    pinterest_file=pin_path,
                    tiktok_files=tiktok_paths,
                    media_plan_file=media_path,
                    prompt_template=prompt_path,
                    template_path=str(ppt_template),
                    manual_campaign_objective=st.session_state.campaign_objective,
                    manual_primary_kpis=st.session_state.primary_kpis,
                    manual_secondary_kpis=st.session_state.secondary_kpis,
                    column_mappings=st.session_state.column_mappings,
                )
                try:
                    replacements, channels_present = extractor.extract_values()
                    processing_warnings = extractor.warnings.copy()
                except (ValueError, ValidationError) as exc:
                    st.error(f"Data validation failed: {exc}")
                    st.info("Please correct the highlighted file(s) and re-upload to continue.")
                    st.stop()

                optional_selection = set(st.session_state.selected_optional_channels or [])
                channels_to_include: List[str] = []
                for channel in channels_present:
                    if channel in OPTIONAL_CHANNELS:
                        if channel in optional_selection:
                            channels_to_include.append(channel)
                    else:
                        channels_to_include.append(channel)
                if "meta" not in channels_to_include and "meta" in channels_present:
                    channels_to_include.append("meta")

                output_path = os.path.join(tmpdir, "automated_presentation.pptx")
                ppt = PowerPointProcessor(ppt_template_path)
                ppt.replace_placeholders(
                    replacements,
                    channels_to_include,
                    output_path,
                    image_placeholders=image_placeholders,
                )

            with open(output_path, "rb") as generated_file:
                ppt_bytes = generated_file.read()

            excluded_channels = {ch for ch in channels_present if ch not in channels_to_include}
            if processing_warnings and excluded_channels:
                def _warning_relates_to_excluded(message: str) -> bool:
                    for channel in excluded_channels:
                        if f"prefix '{channel}'" in message:
                            return True
                        channel_label = CHANNEL_LABELS.get(channel, channel.title())
                        if channel_label in message:
                            return True
                    return False

                processing_warnings = [
                    msg for msg in processing_warnings if not _warning_relates_to_excluded(msg)
                ]

            preview_payload = build_summary_preview(replacements, channels_to_include)
            st.session_state.generated_report = {
                "bytes": ppt_bytes,
                "filename": "automated_presentation.pptx",
                "warnings": processing_warnings,
                "preview": preview_payload,
                "channels": channels_to_include,
            }
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
        st.session_state.column_mappings = {
            "meta": {},
            "pinterest": {},
            "tik_audience": {},
            "tik_ad": {},
            "media_plan": {},
        }
        st.session_state.tiktok_file_roles = {}
        st.session_state.selected_optional_channels = []
        st.session_state.generated_report = None
        st.rerun()

report_state = st.session_state.get("generated_report")
if report_state:
    st.markdown("<div class='card'><div class='section-title'>Report Output</div>", unsafe_allow_html=True)
    st.success("✅ PowerPoint report generated!")
    warnings = report_state.get("warnings") or []
    if warnings:
        st.warning(
            "Data issues detected during processing:\n"
            + "\n".join(f"- {message}" for message in warnings)
        )
    st.download_button(
        label="📥 Download Report",
        data=report_state.get("bytes"),
        file_name=report_state.get("filename", "automated_presentation.pptx"),
        mime="application/vnd.openxmlformats-officedocument.presentationml.presentation",
        key="download_report",
    )
    render_report_preview(report_state.get("preview"), report_state.get("channels", []))
    st.markdown("</div>", unsafe_allow_html=True)
