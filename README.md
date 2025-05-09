README
# Post-Campaign Report Generator
This application is a Streamlit-powered web app that automates the generation of post-campaign summary PowerPoint presentations based on uploaded marketing data. It integrates Meta, Pinterest, and Media Plan data with GPT-powered commentary to create dynamic, data-driven reports.

---

## Features

- Upload campaign data files (Meta, Pinterest, Media Plan).
- Flexibility in report based on variety of data provided: It works just with what you have provided, with only the meta file upload being compulsory.
- Automatically processes, analyzes, and compares performance against estimates.
- GPT-4 integration for generating natural language commentary on KPIs.
- Creates customized PowerPoint slides by replacing placeholders with computed metrics.
- Sleek dark-themed interface with logo and styling.

---

## Required Inputs

- `Meta Excel File (.xlsx)` — Required
- `Pinterest CSV File (.csv)` — Optional
- `Media Plan Excel File (.xlsx)` — Optional

---

## Technologies Used

- **Python 3.13
- `streamlit` for UI
- `pandas` for data manipulation
- `python-pptx` for PowerPoint file creation
- `openai` for GPT-based text generation
- `dotenv` for managing API keys
- `PIL` and `base64` for logo rendering

---

## Setup Instructions

1. Clone the repository.
2. Create a `.env` file and define:
    OPENAI_API_KEY=your_openai_key_here

3. Install dependencies:
    ```
    pip install -r requirements.txt
    ```
4. Run the app:
    ```
    streamlit run your_script_name.py
    ```

---

##  Output

- A downloadable PowerPoint report customized with the uploaded campaign data and AI-generated insights.

---

## Screenshot

![!(image.png)](#)

---

## Notes

- Make sure all placeholders in the PowerPoint template match the ones defined in the code.
- The app uses basic regex and pandas operations to group and compare metrics across platforms and planned values.

---

## 🤝 Contributions

Contributions, suggestions, or pull requests are welcome!

---

