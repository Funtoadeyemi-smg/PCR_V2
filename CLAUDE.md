# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a Streamlit-based Post-Campaign Report Generator that automates PowerPoint presentation creation for marketing campaigns. The app processes uploaded campaign data files (Meta, Pinterest, Media Plan) and uses OpenAI GPT-4 to generate natural language commentary on KPIs, creating customized PowerPoint reports with data-driven insights.

## Development Commands

**Running the application:**
```bash
streamlit run streamlitapp.py
```

**Installing dependencies:**
```bash
pip install -r requirements.txt
```

**Environment setup:**
Create a `.env` file with:
```
OPENAI_API_KEY=your_openai_key_here
CONNECTION_STRING=your_connection_string_here
```

## Architecture Overview

### Core Components

**Main Application (`streamlitapp.py`)**
- Streamlit web interface with file upload capabilities
- Handles Meta Excel files (required), Pinterest CSV files (optional), and Media Plan Excel files (optional)
- Integrates with OpenAI for GPT-powered commentary generation
- Contains embedded PowerPointProcessor class for template modification

**Utility Classes (`utils/`)**
- `DataExtractor`: Processes uploaded campaign data files, extracts metrics, and prepares data for analysis
- `PowerPointProcessor`: Handles PowerPoint template manipulation, placeholder replacement in slides and tables

### Data Flow

1. **File Upload**: Users upload campaign data files through Streamlit interface
2. **Data Processing**: DataExtractor parses files and calculates KPIs (ROAS, CTR, engagement metrics)
3. **AI Commentary**: OpenAI GPT-4 generates insights based on structured prompts from `prompt.txt`
4. **Report Generation**: PowerPointProcessor replaces placeholders in template with computed metrics and AI commentary
5. **Output**: Downloadable PowerPoint report with charts and data-driven insights

### Key Features

- **Multi-platform data integration**: Supports Meta, Pinterest, and Media Plan data sources
- **Flexible input handling**: Works with whatever data is provided (only Meta file is mandatory)
- **AI-powered insights**: Uses structured prompts to generate performance commentary
- **Template-based reporting**: Modifies PowerPoint templates by replacing placeholders
- **Chart generation**: Creates matplotlib visualizations embedded in PowerPoint slides

### Template System

The app uses PowerPoint templates (`automation_template_v3.pptx`) with placeholder text that gets replaced with:
- Calculated metrics (impressions, reach, ROAS, etc.)
- AI-generated commentary from GPT-4
- Generated charts and visualizations

### OpenAI Integration

Uses structured prompts from `prompt.txt` to generate:
- Overall engagement performance analysis
- Sales, revenue, and ROI performance commentary
- Reach performance insights
- Channel comparison analysis
- Audience performance breakdowns
- Campaign recommendations

## File Structure Notes

- `streamlitapp.py`: Main application (currently being modified)
- `streamlitapp2.py`: Appears to be a development version
- `test.ipynb`: Jupyter notebook for testing
- PowerPoint templates are stored in root directory
- Virtual environments in `fresh_env/` and `myvenv/`
- Generated charts saved as PNG files in root directory