# Post-Campaign Report Generator

This project is a Streamlit application that produces PowerPoint post-campaign reports for paid media activity. By ingesting channel performance exports and media plan estimates, the app consolidates metrics, runs validation, generates optional commentary with OpenAI’s GPT models, and emits a templated deck that mirrors the A360 reporting format.

---

## Highlights

- **Multi-channel ingestion**: Works with Meta, Pinterest, TikTok, and media plan exports. TikTok supports both CSV and Excel audience/ad reports with column mapping.
- **Pre-flight validation**: Displays detected columns, missing fields, and optional column mapping selectors before you generate a report.
- **User-controlled sections**: You can choose which optional channel summaries appear in the final deck; the app adjusts slide placeholders accordingly.
- **AI commentary**: When an OpenAI API key is present, GPT-4 produces narrative commentary for each channel, KPI deltas, and the closing summary.
- **PowerPoint templating**: Replaces text placeholders, injects summary charts, and supports optional image swaps for the table of contents and summary slides.
- **Report preview**: After generation, the app shows a table of the metrics that populate the overall summary slide and each channel summary slide so you can review before downloading.
- **Warning surfacing**: Any gaps (for example, missing ad-level data) are raised prominently without blocking report creation.

---

## Data Requirements

| File | Required? | Notes |
| --- | --- | --- |
| Meta performance export (`.xlsx` or `.csv`) | Yes | Must include the columns listed in `REQUIRED_COLUMNS["meta"]`. |
| Pinterest performance export (`.csv` / `.xlsx`) | Optional | Enable the Pinterest section when supplied. |
| TikTok audience / ad exports (`.csv` or `.xlsx`) | Optional | Upload either/both; column mappings can be applied in the UI. |
| Media plan (`.xlsx` / `.csv`) | Optional | Used for estimate comparisons on summary slides. |

Only the Meta file is mandatory. All other files can be omitted; the app simply hides the corresponding sections and placeholders are filled with `N/A`.

---

## Getting Started

### 1. Clone and install

```bash
git clone https://github.com/Funtoadeyemi-smg/PCR_V2.git
cd PCR_V2
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 2. Configure environment variables

Create a `.env` file in the project root:

```
OPENAI_API_KEY=sk-xxxxxxxxxxxxxxxx
```

If you do not provide an API key the application still runs; commentary placeholders will contain a fallbacdk message.

### 3. Launch the Streamlit app

```bash
streamlit run streamlitapp2.py
```

The original `streamlitapp.py` is kept as a legacy version but the current feature set lives in `streamlitapp2.py`.

---

## Using the App

1. **Campaign details**: Enter the campaign objective and select primary/secondary KPIs. The dropdown closes after each selection to streamline the flow.
2. **Upload data**: Add the available channel exports and media plan files. TikTok uploads can be multiple files (audience and ad level).
3. **Pre-flight checks**: Expand each report to review detected columns, apply column mappings, and note any processing flags (for example, trailing “Total” rows that will be ignored).
4. **Choose channel sections**: When more than one optional channel is available, pick which ones to include in the deck. If only one optional channel exists it is automatically included.
5. **Images (optional)**: Supply replacement images for the table of contents and campaign summary slides if desired.
6. **Generate report**: Click “Generate PowerPoint Report”. A spinner appears while the extractor validates, aggregates, and populates the template.
7. **Review output**: After generation the app displays:
   - A persistent download button for `automated_presentation.pptx`.
   - Any warnings relevant to the channels you selected.
   - A preview table summarising the metrics that populate the overall summary slide (page 7) and each channel summary slide (Meta page 13, Pinterest page 21, TikTok page 29).

Warnings highlight issues such as “No ad-level data found for prefix 'tik'. Creative slides will show N/A.” They do not stop the report—use them as prompts to upload richer data if needed.

---

## PowerPoint Template

The application expects its assets in `prompts_artefacts/`. Ensure `powerpoint_template.pptx`, `prompt.txt`, `commentary_metrics_map.txt`, and `smg2.jpeg` live inside that folder. Placeholders use braces (for example, `{meta_gross_spend}`). If you modify the template ensure the placeholder names align with those defined in `utils/dataextractor.py` and `utils/powerpointprocessor.py`. Replacement images map to placeholders `{table_of_contents_picture}` and `{campaign_summary_picture}`.

---

## Streamlit Cloud Deployment

This repository is compatible with Streamlit Community Cloud:

1. Push your changes to GitHub.
2. Connect the repository and branch in the Streamlit dashboard.
3. On every push Streamlit Cloud rebuilds the environment, runs `pip install -r requirements.txt`, and restarts the app automatically.

---

## Contributing

Bug reports, enhancement ideas, and pull requests are welcome. Please open an issue describing the change before submitting a large feature so we can align on scope.

---

## License

This project is distributed under the MIT License. See `LICENSE` for details.
