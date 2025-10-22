# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a Streamlit-based energy monitoring dashboard application that visualizes energy consumption and production data from Snowflake. The application provides two main dashboards:

1. **Bubble Budget Scrubber** ([streamlit_app.py](streamlit_app.py)) - A weekly energy tracking dashboard with gamification elements (scoring system, visual dials) that helps users monitor their energy consumption against dynamic targets
2. **Bayou Energy Dashboard** ([bayou_dashboard.py](bayou_dashboard.py) - A monthly billing and interval data dashboard that fetches customer data from the Bayou Energy API

## Running the Application

### Development Setup
```bash
# Install dependencies
pip install -r requirements.txt

# Run the main energy tracking dashboard
streamlit run streamlit_app.py

# Run the Bayou billing dashboard
streamlit run bayou_dashboard.py
```

### Dev Container
The project includes a devcontainer configuration that:
- Uses Python 3.11
- Auto-installs dependencies from requirements.txt
- Automatically runs `streamlit run streamlit_app.py` on attach
- Exposes port 8501 for the Streamlit app

## Architecture

### Data Flow (streamlit_app.py)

1. **Data Source**: Snowflake tables via Snowpark
   - `ENERGY_SHARED.TRUNKS_HELIOS.HOURLY_CONSUMPTION` - hourly consumption data
   - `ENERGY_SHARED.TRUNKS_HELIOS.HOURLY_PRODUCTION` - hourly production (solar) data
   - `ENERGY_SHARED.TRUNKS_HELIOS.PROPERTIES` - site metadata with HELIOS_SITE_IDS array

2. **Session Management**: Uses `@st.cache_resource` for Snowpark session (see [streamlit_app.py:24](streamlit_app.py#L24))

3. **Data Processing Pipeline**:
   - Load week data → compute target from previous week → calculate scores → generate visualizations
   - Timezone-aware processing using `timezonefinder` to convert UTC data to local time
   - Net energy calculation: `net_kwh = consumption_kwh - production_kwh`

4. **Visualization Components**:
   - **Dial visualization**: SVG-based radial progress indicator using wedge paths from [svg_wedges.py](svg_wedges.py)
   - **Bubble grid** (legacy): 10x10 grid showing progress with color-coded bubbles

### Scoring Algorithm

The weekly scoring system ([streamlit_app.py:354](streamlit_app.py#L354)) awards 0-300 points based on:
- Target (T): Previous week's net energy (with floor of ±25 kWh)
- Current net consumption (W): positive = importing, negative = exporting
- Different scoring curves for importers vs exporters with bonus zones

### Insights Generation

Uses OpenAI GPT-4 to generate weekly energy insights:
1. Data analysis functions ([streamlit_app.py:400-483](streamlit_app.py#L400)) compute stats, anomalies, hourly profiles
2. Markdown report generated with structured sections
3. OpenAI API called with prompt template from [prompts.py](prompts.py) to create homeowner-friendly summary
4. Supports "Energy Coach" persona (friendly, practical, competitive tone)

### Bayou Dashboard (bayou_dashboard.py)

- REST API client for Bayou Energy platform (staging/production)
- Basic auth using API key from secrets
- Endpoints: `/api/v2/customers`, `/api/v2/customers/{id}/bills`, `/api/v2/customers/{id}/intervals`
- Displays current month's billing and daily interval charts per meter

## Configuration Requirements

### Streamlit Secrets (.streamlit/secrets.toml)

```toml
[snowflake]
account = "..."
user = "..."
password = "..."
warehouse = "..."
database = "ENERGY_SHARED"
schema = "TRUNKS_HELIOS"

[openai]
api_key = "sk-..."

[push]
endpoint = "https://..."
token = "..."

BAYOU_API_KEY = "..."  # For bayou_dashboard.py
```

## Key Constants

### Scoring Parameters ([streamlit_app.py:341-351](streamlit_app.py#L341))
- `FLOOR_ABS_KWH = 25` - Minimum target threshold
- `DELTA_BAND = 20` - Bootstrap scoring band
- `MAX_POINTS = 300` - Maximum weekly score
- `PCT_ZERO = 120` - Import threshold for zero points
- `EXPORT_CAP = 150` - Maximum export percentage for scoring

### Timezone Handling
- All Snowflake data stored in UTC
- Site timezone determined from coordinates using `timezonefinder`
- Week calculations use Monday 00:00 local time as start

## Code Patterns

### Caching Strategy
- `@st.cache_resource` for Snowpark session (persists across reruns)
- `@st.cache_data(ttl=900)` for API calls in bayou_dashboard (15 min TTL)
- `@st.cache_data` for data fetching and computation functions (no expiration)

### Data Fetching Pattern
```python
@st.cache_data
def load_data(site_id: str, week_start: date, tz_str: str) -> Optional[pd.DataFrame]:
    session = create_snowpark_session()
    # ... query and process data
    return df
```

### Column Name Normalization
After loading from Snowflake, always normalize column names to lowercase:
```python
df.columns = [c.lower() for c in df.columns]
```

### Time Window Pattern
For weekly data, build local time windows then convert to UTC for Snowflake queries ([streamlit_app.py:198-205](streamlit_app.py#L198))

## Dependencies of Note

- `streamlit` - Web framework
- `snowflake-snowpark-python` - Snowflake data access
- `pandas` - Data manipulation
- `openai` - AI-powered insights
- `timezonefinder` - Coordinate → timezone conversion
- `h3` - Likely for geospatial operations (installed but not actively used)
