# bayou_energy_tracker.py
import os
import datetime as dt
from datetime import datetime, timedelta, date
from typing import List, Dict, Optional, Any
from zoneinfo import ZoneInfo
import re

import pandas as pd
import numpy as np
import requests
import streamlit as st
import streamlit.components.v1 as components
from timezonefinder import TimezoneFinder
from openai import OpenAI
from svg_wedges import WEDGE_DATA_JS
from prompts import template_weekly_insights_prompt

# -----------------------------------------------------------------------------
# Page Config & Global Setup
# -----------------------------------------------------------------------------
st.set_page_config(page_title="Bayou Energy Tracker", layout="wide")
st.title("🔋 Bayou Energy Tracker – Weekly Insights")

# Initialize OpenAI client
client = OpenAI(api_key=st.secrets["openai"]["api_key"])

# Timezone lookup helper
TF = TimezoneFinder()

# -----------------------------------------------------------------------------
# 1. Bayou API Client Functions
# -----------------------------------------------------------------------------
default_env = "production"
bayou_domain = st.sidebar.selectbox("Bayou environment", ["staging", "production"],
                                   index=0 if default_env == "staging" else 1)
bayou_domain = f"{bayou_domain}.bayou.energy" if bayou_domain == "staging" else "bayou.energy"

api_key = (
    st.secrets.get("BAYOU_API_KEY")
    or os.getenv("BAYOU_API_KEY")
)
if not api_key:
    st.error("Add your Bayou API key to run the dashboard.")
    st.stop()

AUTH = (api_key, "")

@st.cache_data(ttl=900)  # 15 min cache
def fetch_customers() -> List[Dict]:
    """Fetch all customers, handling pagination if present."""
    all_customers = []
    url = f"https://{bayou_domain}/api/v2/customers"

    while url:
        resp = requests.get(url, auth=AUTH, timeout=30)
        resp.raise_for_status()
        data = resp.json()

        # Handle both list response and paginated response
        if isinstance(data, list):
            all_customers.extend(data)
            break
        elif isinstance(data, dict):
            # Check for common pagination patterns
            customers = data.get('data') or data.get('customers') or data.get('results', [])
            all_customers.extend(customers)

            # Check for next page
            url = data.get('next') or data.get('next_url') or data.get('links', {}).get('next')
        else:
            break

    st.info(f"Loaded {len(all_customers)} customers from Bayou API")
    return all_customers

@st.cache_data(ttl=900)
def fetch_intervals(cust_id: str) -> Dict:
    url = f"https://{bayou_domain}/api/v2/customers/{cust_id}/intervals"
    resp = requests.get(url, auth=AUTH, timeout=60)
    resp.raise_for_status()
    return resp.json()

@st.cache_data(ttl=900)
def fetch_customer_detail(cust_id: str) -> Dict:
    """Fetch detailed customer info including connection status."""
    url = f"https://{bayou_domain}/api/v2/customers/{cust_id}"
    resp = requests.get(url, auth=AUTH, timeout=30)
    resp.raise_for_status()
    return resp.json()

def delete_customer(cust_id: str) -> tuple[bool, str]:
    """Delete a customer via the Bayou API. Returns (success, message)."""
    url = f"https://{bayou_domain}/api/v2/customers/{cust_id}"
    try:
        st.info(f"Attempting to delete: DELETE {url}")
        resp = requests.delete(url, auth=AUTH, timeout=30)

        # Log the response
        st.info(f"Response status: {resp.status_code}")

        if resp.status_code in [200, 204]:
            # Clear the cache so the customer list refreshes
            fetch_customers.clear()
            return True, f"Successfully deleted customer {cust_id}"
        else:
            return False, f"Delete failed with status {resp.status_code}: {resp.text}"
    except requests.exceptions.HTTPError as e:
        error_msg = f"HTTP Error: {e.response.status_code} - {e.response.text}"
        return False, error_msg
    except Exception as e:
        return False, f"Failed to delete customer: {str(e)}"

# -----------------------------------------------------------------------------
# 2. Scoring System Constants & Functions
# -----------------------------------------------------------------------------
FLOOR_ABS_KWH = 25
DELTA_BAND = 20
BOOTSTRAP_T = 150
MAX_POINTS = 300
PCT_ZERO = 120
EXPORT_CAP = 150
SLOPE_0_50 = 1
SLOPE_50_100 = 2
SLOPE_100_150 = 3

def score_week(W: float, T: float) -> int:
    """
    Given:
      W = cumulative net kWh so far (positive = imported, negative = exported)
      T = target net kWh for the week (positive = import target, negative = export target)
    Return a score from 0 to MAX_POINTS following the specified rules.
    """
    if abs(T) <= FLOOR_ABS_KWH:
        d = W - T
        if d > DELTA_BAND:
            return 0
        if d < -DELTA_BAND:
            return MAX_POINTS
        return round(BOOTSTRAP_T - 2.5 * d)

    if T > 0:
        p = 100 * W / T
        if p >= PCT_ZERO:
            pts = 0
        elif p >= 100:
            pts = 100 * (PCT_ZERO - p) / (PCT_ZERO - 100)
        elif p >= 0:
            pts = (100 - p) * 2 + 100
        elif p >= -50:
            pts = (100 - p) * 2
        else:
            pts = MAX_POINTS
        return round(pts)

    if W > 0:
        return 0

    exp_pct = 100 * abs(W) / abs(T)
    if exp_pct < 50:
        pts = SLOPE_0_50 * exp_pct
    elif exp_pct < 100:
        pts = 50 + SLOPE_50_100 * (exp_pct - 50)
    elif exp_pct < EXPORT_CAP:
        pts = 150 + SLOPE_100_150 * (exp_pct - 100)
    else:
        pts = MAX_POINTS
    return round(pts)

# -----------------------------------------------------------------------------
# 3. Data Transformation Layer
# -----------------------------------------------------------------------------
def transform_bayou_intervals_to_df(intervals_json: Dict, meter_id: str, tz_str: str = "UTC") -> Optional[pd.DataFrame]:
    """
    Transform Bayou intervals JSON to DataFrame matching streamlit_app format.
    Returns DataFrame with columns: ts, cons_kwh, prod_kwh, kwh
    """
    meters = intervals_json.get("meters", [])
    meter = next((m for m in meters if m.get("id") == meter_id), None)

    if not meter:
        return None

    intervals = meter.get("intervals", [])
    if not intervals:
        return None

    # Build DataFrame
    records = []
    for interval in intervals:
        ts = pd.to_datetime(interval["start"])
        net_kwh = interval.get("net_electricity_consumption", 0)
        cons_kwh = interval.get("electricity_consumption", None)
        prod_kwh = interval.get("generated_electricity", None)

        # If consumption/production not provided, derive from net
        # Assume: net = consumption - production
        # If net is positive (importing), consumption >= net, production = 0 (if not provided)
        # If net is negative (exporting), production >= abs(net), consumption = 0 (if not provided)
        if cons_kwh is None:
            cons_kwh = max(0, net_kwh) if net_kwh is not None else 0
        if prod_kwh is None:
            prod_kwh = max(0, -net_kwh) if net_kwh is not None else 0

        # Bayou API returns values in Wh (watt-hours), convert to kWh
        # Values like 180, 190, 200 Wh = 0.18, 0.19, 0.20 kWh
        records.append({
            "ts": ts,
            "cons_kwh": (cons_kwh or 0) / 1000.0,
            "prod_kwh": (prod_kwh or 0) / 1000.0,
            "kwh": (net_kwh or 0) / 1000.0
        })

    df = pd.DataFrame(records)

    # Convert timestamps to timezone-aware
    df["ts"] = pd.to_datetime(df["ts"], utc=True).dt.tz_convert(tz_str)
    df = df.sort_values("ts").reset_index(drop=True)

    return df

@st.cache_data
def load_week_data_bayou(cust_id: str, meter_id: str, week_start: date, tz_str: str = "UTC") -> Optional[pd.DataFrame]:
    """
    Load one week of data from Bayou API for a given customer/meter.
    From Monday 00:00 local through Sunday 23:59:59 local.
    """
    # Fetch all intervals
    intervals_json = fetch_intervals(cust_id)

    # Transform to DataFrame
    df = transform_bayou_intervals_to_df(intervals_json, meter_id, tz_str)

    if df is None or df.empty:
        st.warning(f"No data returned from API for meter {meter_id}")
        return None

    # Debug: Show data range
    st.info(f"Total data available: {len(df)} intervals from {df['ts'].min()} to {df['ts'].max()}")

    # Filter to week range
    local_tz = ZoneInfo(tz_str)
    local_start = datetime.combine(week_start, datetime.min.time()).replace(tzinfo=local_tz)
    local_end = local_start + timedelta(days=7)

    st.info(f"Filtering for week: {local_start} to {local_end}")

    mask = (df["ts"] >= local_start) & (df["ts"] < local_end)
    df_week = df[mask].copy()

    if df_week.empty:
        st.warning(f"No data found for week {week_start}. Try selecting a week between {df['ts'].min().date()} and {df['ts'].max().date()}")
        return None

    return df_week

@st.cache_data
def compute_target(cust_id: str, meter_id: str, week_start: date, tz_str: str = "UTC") -> float:
    """Compute the target T for the current week based on last week's net."""
    prev_start = week_start - timedelta(days=7)
    df_prev = load_week_data_bayou(cust_id, meter_id, prev_start, tz_str)

    T = df_prev["kwh"].sum() if (df_prev is not None) else 0.0

    if abs(T) <= FLOOR_ABS_KWH:
        T = FLOOR_ABS_KWH if T >= 0 else -FLOOR_ABS_KWH

    return T

@st.cache_data
def compute_grid_target(cust_id: str, meter_id: str, week_start: date, tz_str: str = "UTC") -> float:
    """Return total grid import from the previous week."""
    prev_start = week_start - timedelta(days=7)
    df_prev = load_week_data_bayou(cust_id, meter_id, prev_start, tz_str)
    return df_prev["cons_kwh"].sum() if df_prev is not None else 0.0

# -----------------------------------------------------------------------------
# 4. Visualization Components
# -----------------------------------------------------------------------------
BUBBLE_CSS = """
<style>
  .b{width:14px;height:14px;border-radius:50%;display:inline-block;margin:1px;}
  .lo{background:#FBD1B5;}    /* light-orange */
  .do{background:#F47B60;}    /* dark-orange */
  .green{background:#A7E299;} /* export */
  .empty{background:#F0F0F0;border:1px solid #e5e5e5;}
</style>
"""

def bubble_grid(W: float, T: float) -> str:
    """
    Render a 10x10 grid of 'bubbles' to visualize progress towards T.
    """
    cells = ["lo"] * 100
    pct = (W / T * 100) if T else 0.0

    if W >= 0:
        steps = min(110, int(pct))
        for i in range(steps):
            cells.pop()
            cells.insert(0, "empty" if i < 100 else "do")
    else:
        for _ in range(min(100, int(abs(pct)))):
            cells.pop(0)
            cells.append("green")

    html = BUBBLE_CSS + '<div style="line-height:0.7">'
    for idx, cls in enumerate(cells):
        html += f"<span class='b {cls}'></span>"
        if (idx + 1) % 10 == 0:
            html += '<br>'

    if W < 0 and abs(pct) > 100:
        bonus = int(min(50, abs(pct) - 100))
        html += '<div style="line-height:0.7; margin-top:4px;">'
        html += ''.join(["<span class='b green'></span>" for _ in range(bonus)])
        html += '</div>'

    html += '</div>'
    return html

def _dial_html(W: float, T: float, pct: float) -> str:
    """
    Build the radial dial visualization.
    """
    mode = "export" if T < 0 else "import"

    text = (
        f"{pct:.0f}% complete to export goal"
        if T < 0
        else f"{pct:.0f}% of energy goal consumed"
    )

    return f"""
<div id='dial-container' style='position:relative;width:303px;height:296px;background:#DFF4AE;'>
  <div id='dial-root'></div>
  <div style='position:absolute;top:50%;left:50%;transform:translate(-50%,-50%);font-size:20px;font-weight:bold;text-align:center;color:#000;background:#DFF4AE;'>
    {text}
  </div>
</div>
<script>
{WEDGE_DATA_JS}

const filledPercentage = {pct:.2f};
const mode             = "{mode}";

const greyWedges   = Math.max(0, Math.floor(filledPercentage));
const purpleWedges = Math.max(0, Math.floor(-filledPercentage));
const purpleStart  = Math.max(1, 101 - purpleWedges);

const svgNS = "http://www.w3.org/2000/svg";
const svg   = document.createElementNS(svgNS,"svg");
svg.setAttribute("viewBox","0 0 303 296");
svg.setAttribute("width",303); svg.setAttribute("height",296);
document.getElementById("dial-root").appendChild(svg);

svg.setAttribute(
  "style",
  "transform:rotate(-90deg);transform-origin:50% 50%;"
);

wedgeData.forEach(w => {{
  const p = document.createElementNS(svgNS, "path");
  p.setAttribute("d", w.path);

  let color = "#1B240F";

  if (purpleWedges > 0 && w.id >= purpleStart) {{
    color = "#815ED5";
  }} else if (greyWedges > 0 && w.id <= greyWedges) {{
    color = "#ACACAC";
  }}

  p.setAttribute("fill", color);
  svg.appendChild(p);
}});
</script>
"""

# -----------------------------------------------------------------------------
# 5. Insights & Analysis Engine
# -----------------------------------------------------------------------------
def validate_and_clean(df: pd.DataFrame, tz_str: str = "UTC") -> pd.DataFrame:
    assert {"ts", "cons_kwh"}.issubset(df.columns), "Missing ts or cons_kwh"
    df2 = df.rename(columns={"ts": "timestamp", "cons_kwh": "import_kwh"}).copy()
    df2 = df2.sort_values("timestamp")
    df2["timestamp"] = pd.to_datetime(df2["timestamp"], utc=True).dt.tz_convert(tz_str)
    df2["import_kwh"] = df2["import_kwh"].fillna(0)
    df2["export_kwh"] = df2.get("prod_kwh", 0).fillna(0)
    df2["net_kwh"] = df2["import_kwh"] - df2["export_kwh"]
    df2["day"] = df2["timestamp"].dt.date
    df2["hour"] = df2["timestamp"].dt.hour

    iv = df2["timestamp"].diff().dropna().dt.total_seconds().mode().iloc[0] / 3600
    df2.attrs["interval_hours"] = iv
    return df2

def high_level_stats(df: pd.DataFrame) -> Dict[str, Any]:
    total = df["import_kwh"].sum()
    total_grid = (df["import_kwh"] - df["export_kwh"]).sum()
    daily = df.groupby("day")["import_kwh"].sum()
    avg = daily.mean()
    peak_idx = df["import_kwh"].idxmax()
    peak = df.loc[peak_idx]
    dayname = peak["timestamp"].day_name()
    base = df[df["hour"].between(2, 5)]["import_kwh"].mean()
    cv = df["import_kwh"].std() / df["import_kwh"].mean()

    return {
        "total_consumption": total,
        "total_grid": total_grid,
        "daily_avg": avg,
        "peak_day": peak["timestamp"].date(),
        "peak_day_name": dayname,
        "peak_hour_ts": peak["timestamp"],
        "peak_hour_kwh": peak["import_kwh"],
        "base_load": base,
        "cv": cv
    }

def weekday_heaviest_3h(df_clean: pd.DataFrame) -> Dict[str, Any]:
    df_wd = df_clean[df_clean["timestamp"].dt.dayofweek < 5]
    hourly_avg = (
        df_wd
        .groupby(df_wd["timestamp"].dt.hour)["import_kwh"]
        .mean()
        .reindex(range(24), fill_value=0)
    )

    block_avg = {}
    for h in range(24):
        hrs = [(h + i) % 24 for i in range(3)]
        block_avg[h] = np.mean([hourly_avg.loc[i] for i in hrs])

    best_h = max(block_avg, key=block_avg.get)
    return {"start_hour": best_h, "avg_kwh": block_avg[best_h]}

def hourly_profile(df: pd.DataFrame) -> pd.DataFrame:
    prof = df.groupby("hour")["import_kwh"].mean().reset_index()
    prof.columns = ["hour", "avg_import_kwh"]
    return prof

def anomaly_scan(df: pd.DataFrame, z_thresh: float = 3.0) -> pd.DataFrame:
    mu, sd = df["import_kwh"].mean(), df["import_kwh"].std()
    df2 = df.copy()
    df2["z"] = (df2["import_kwh"] - mu) / sd
    return df2[abs(df2["z"]) >= z_thresh][["timestamp", "import_kwh", "z"]]

def opportunity_report(df: pd.DataFrame, stats: Dict[str, Any]) -> Dict[str, Any]:
    m17 = df[df["hour"].between(17, 20)]["import_kwh"].mean()
    m10 = df[df["hour"].between(10, 15)]["import_kwh"].mean()
    shift = max(0, m17 - m10)
    p95 = np.percentile(df[df["hour"].between(17, 20)]["import_kwh"], 95)
    bat = max(0, p95 - stats["base_load"])
    note = "No solar data."
    if df["export_kwh"].sum() > 0:
        sc = df["import_kwh"].sum() / df["export_kwh"].sum()
        note = f"Self-consumption ratio ~{sc:.2f}. <0.8 = room to shift more consumption to solar hours, >2 = consider battery or larger solar system."
    return {"shift_savings": shift, "battery_kwh": bat, "solar_note": note}

def render_markdown(parts: Dict[str, Any]) -> str:
    s = parts["stats"]
    common = parts["common"]
    anom = parts["anomalies"]
    opp = parts["opportunity"]

    md = []
    md.append("### 1  Big-picture numbers")
    md.append("| Metric | Value | Comment |")
    md.append("| --- | --- | --- |")
    md.append(f"| Total energy consumed | **{s['total_consumption']:.0f} kWh** | Entire week |")
    md.append(f"| Total grid energy | **{s['total_grid']:.0f} kWh** | What you pay for |")
    md.append(f"| Daily average consumption | {s['daily_avg']:.1f} kWh/day | Peak day = {s['peak_day_name']} ({s['peak_day']}) |")
    md.append(f"| Highest consumption day | {s['peak_day_name']} | Day with max import |")
    ts_str = s["peak_hour_ts"].strftime("%Y-%m-%d %H:%M")
    md.append(f"| Highest single hour | {s['peak_hour_kwh']:.2f} kWh ({ts_str}) | |")
    md.append(f"| Overnight baseload (02–05) | {s['base_load']:.2f} kWh/h | Avg import 2–5 AM |")
    md.append(f"| Hour-to-hour variability (CV) | {s['cv']:.2f} | Coefficient of variation |")
    md.append("")

    sh = common["start_hour"]
    eh = (sh + 2) % 24
    md.append("### 2  Heaviest 3-hour window (weekdays)")
    md.append(
        f"* Heaviest window: **{sh:02d}:00 – {eh:02d}:59** "
        f"(on average, **{common['avg_kwh']:.1f} kWh**)."
    )
    md.append("")

    md.append("### 3  One-off anomalies")
    md.append("| Timestamp | kWh | z-score |")
    md.append("| --- | --- | --- |")
    for _, r in anom.iterrows():
        ts = r["timestamp"].strftime("%Y-%m-%d %H:%M")
        md.append(f"| {ts} | {r['import_kwh']:.2f} | {r['z']:.2f} |")
    md.append("")

    md.append("### 4  Opportunities")
    md.append(f"* **Load-shifting:** ~{opp['shift_savings']:.1f} kWh/day from 17–20 h.")
    md.append(f"* **Battery sizing:** ~{opp['battery_kwh']:.1f} kWh covers peak.")
    md.append(f"* **Solar check:** {opp['solar_note']}")
    return "\n".join(md)

@st.cache_data
def compute_insights_report(df: pd.DataFrame, tz_str: str = "UTC") -> str:
    df2 = validate_and_clean(df, tz_str)
    stats = high_level_stats(df2)
    common = weekday_heaviest_3h(df2)
    anom = anomaly_scan(df2)
    opp = opportunity_report(df2, stats)

    parts = {
        "stats": stats,
        "common": common,
        "hourly": hourly_profile(df2),
        "anomalies": anom,
        "opportunity": opp
    }
    return render_markdown(parts)

def summarize_for_owner(markdown: str, user_name: Optional[str] = None) -> str:
    """Send the markdown report to OpenAI and return a short owner-friendly summary."""
    if "openai" not in st.secrets or not st.secrets["openai"].get("api_key"):
        st.error("OpenAI API key not configured in st.secrets")
        return ""

    first_name = None
    if user_name:
        first_name = user_name.split()[0] if user_name else None

    prompt = template_weekly_insights_prompt(markdown, first_name)

    try:
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=800
        )
        return resp.choices[0].message.content.strip()
    except Exception as e:
        st.error(f"OpenAI request failed: {e}")
        return ""

def compute_so_far_insights(df: pd.DataFrame, idx: int, T_net: float, T_grid: float,
                            show_grid: bool, tz_str: str = "UTC") -> str:
    """Return a markdown table with quick insights up to the given hour."""
    df_part = df.iloc[: idx + 1].copy()

    total_hours = len(df)
    elapsed_hours = idx + 1

    W_net = df_part["kwh"].sum()
    cons_so_far = df_part["cons_kwh"].sum()
    prod_so_far = df_part["prod_kwh"].sum()

    if show_grid:
        W_display = cons_so_far
        T_display = T_grid
    else:
        W_display = W_net
        T_display = T_net

    budget_used = abs(W_display)
    budget_left = abs(T_display) - abs(W_display)
    pct_goal = abs(W_display) / abs(T_display) * 100 if T_display else 0
    pct_time = elapsed_hours / total_hours * 100
    pace_ratio = pct_goal / pct_time if pct_time else 0

    if pace_ratio > 1.05:
        pace_note = "Behind pace"
    elif pace_ratio < 0.95:
        pace_note = "On track to beat last week"
    else:
        pace_note = "On track"

    mean_imp = df_part["cons_kwh"].mean()
    std_imp = df_part["cons_kwh"].std()
    thresh = mean_imp + 2 * std_imp
    spikes = df_part[df_part["cons_kwh"] > thresh]
    top_spikes = spikes.sort_values("cons_kwh", ascending=False).head(3)
    spike_str = ", ".join(
        [
            f"{r['ts'].tz_convert(tz_str).strftime('%a %H:%M')} ({r['cons_kwh']:.1f} kWh)"
            for _, r in top_spikes.iterrows()
        ]
    )
    spike_str = spike_str if spike_str else "None"

    peak_idx = df_part["cons_kwh"].idxmax()
    peak_ts = df_part.loc[peak_idx, "ts"].tz_convert(tz_str)
    peak_val = df_part.loc[peak_idx, "cons_kwh"]

    df_part["sc_ratio"] = (
        df_part[["cons_kwh", "prod_kwh"]].min(axis=1) / df_part["cons_kwh"].replace(0, np.nan)
    )
    best_sc_idx = df_part["sc_ratio"].idxmax()
    best_sc_ts = df_part.loc[best_sc_idx, "ts"].tz_convert(tz_str)
    best_sc_ratio = df_part.loc[best_sc_idx, "sc_ratio"]

    self_cons = (df_part[["cons_kwh", "prod_kwh"]].min(axis=1).sum())
    sc_ratio = self_cons / prod_so_far if prod_so_far > 0 else 0

    cons_rate = pct_goal / elapsed_hours if elapsed_hours else 0
    hours_per_pct = elapsed_hours / pct_goal if pct_goal else 0

    md = []
    md.append("### Insights So Far")
    md.append("| Metric | Value | Comment |")
    md.append("| --- | --- | --- |")
    md.append(f"| Energy budget used | {budget_used:.1f} kWh | {budget_left:.1f} kWh left |")
    md.append(f"| Peak consumption hour | {peak_val:.1f} kWh | {peak_ts.strftime('%a %H:%M')} |")
    md.append(f"| Best self-consumption hour | {best_sc_ratio:.1%} | {best_sc_ts.strftime('%a %H:%M')} |")
    rate_comment = f"{cons_rate:.2f}%/h" if cons_rate else "0%/h"
    if cons_rate > 0:
        rate_comment += f" (~1% every {hours_per_pct:.1f} h)"
    md.append(f"| Consumption rate | {rate_comment} | Percent of goal per hour |")
    md.append(f"| Pace vs goal | {pct_goal:.1f}% used, {pct_time:.1f}% time | {pace_note}, {pace_ratio:.2f} ratio |")
    md.append(f"| Spikes so far | {len(spikes)} | {spike_str} |")
    md.append(f"| Self consumption ratio | {sc_ratio:.1%} | Portion of solar used onsite |")
    return "\n".join(md)

# -----------------------------------------------------------------------------
# 6. Main App UI
# -----------------------------------------------------------------------------
def main():
    # Sidebar: Customer/Meter selection
    customers = fetch_customers()
    if not customers:
        st.warning("No customers found for this key/environment.")
        st.stop()

    cust_options = {
        f"{c.get('external_id') or c['id']} – {c.get('email', '<no-email>')}": c["id"]
        for c in customers
    }
    cust_label = st.sidebar.selectbox("Customer", sorted(cust_options))
    cust_id = cust_options[cust_label]

    # Customer Details Section
    st.sidebar.markdown("---")
    with st.sidebar.expander("🔍 Customer Details"):
        try:
            customer_detail = fetch_customer_detail(cust_id)

            # Ensure customer_detail is a dict
            if not isinstance(customer_detail, dict):
                st.error(f"Unexpected response type: {type(customer_detail)}")
                st.code(str(customer_detail))
            else:
                # Display key information
                st.caption(f"**Customer ID:** {customer_detail.get('id', 'N/A')}")
                st.caption(f"**External ID:** {customer_detail.get('external_id', 'N/A')}")
                st.caption(f"**Email:** {customer_detail.get('email', 'N/A')}")

                # Connection status indicators
                utility_info = customer_detail.get('utility', {})
                if utility_info:
                    if isinstance(utility_info, dict):
                        st.metric("Utility", utility_info.get('name', 'Unknown'))
                    else:
                        st.metric("Utility", str(utility_info))

                # Check for various status fields
                status_fields = ['connection_status', 'credentials_status', 'authentication_status',
                               'has_active_connection', 'needs_reauthentication', 'onboarding_status']

                st.markdown("**Status Fields:**")
                for field in status_fields:
                    if field in customer_detail:
                        value = customer_detail[field]
                        if field == 'needs_reauthentication' and value:
                            st.error(f"⚠️ {field}: {value}")
                        elif isinstance(value, bool):
                            st.info(f"{'✅' if value else '❌'} {field}: {value}")
                        else:
                            st.info(f"{field}: {value}")

                # Onboarding link if available
                if 'onboarding_link' in customer_detail:
                    st.markdown("**Onboarding Link:**")
                    st.code(customer_detail['onboarding_link'], language=None)

                # Show full JSON (toggle with checkbox instead of expander)
                st.markdown("---")
                show_json = st.checkbox("Show Full Customer JSON", key="show_customer_json")
                if show_json:
                    st.json(customer_detail)

        except Exception as e:
            st.error(f"Failed to fetch customer details: {e}")

    # Delete customer button with warnings
    st.sidebar.markdown("---")
    with st.sidebar.expander("⚠️ Delete Customer (CAUTION)"):
        st.error("**WARNING:** This endpoint is undocumented and may timeout or fail.")
        st.warning(f"Customer to delete: **{cust_label}**")
        st.caption(f"Customer ID: {cust_id}")
        st.info("⏱️ This operation may take 30+ seconds or timeout. Contact Bayou support if you need to delete customers reliably.")

        confirm_text = st.text_input("Type DELETE to confirm", key="delete_confirm")
        if st.button("🗑️ Delete Customer", type="primary", key="delete_btn"):
            if confirm_text == "DELETE":
                with st.spinner("Attempting to delete (this may timeout)..."):
                    success, message = delete_customer(cust_id)
                    if success:
                        st.success(message)
                        st.balloons()
                        import time
                        time.sleep(1)
                        st.rerun()
                    else:
                        st.error(message)
                        st.info("💡 If this persists, contact Bayou support to delete this customer.")
            else:
                st.error("Please type DELETE (all caps) to confirm deletion")
    st.sidebar.markdown("---")

    # Fetch intervals to get meter list
    try:
        intervals_json = fetch_intervals(cust_id)
        meters = intervals_json.get("meters", [])
        if not meters:
            st.warning("No meters found for this customer.")
            st.stop()

        meter_ids = [m["id"] for m in meters]
        meter_id = st.sidebar.selectbox("Meter", meter_ids)
    except Exception as e:
        st.error(f"Failed to fetch meters: {e}")
        st.stop()

    # Timezone selector
    tz_str = st.sidebar.selectbox("Timezone", ["UTC", "America/New_York", "America/Chicago", "America/Denver", "America/Los_Angeles"], index=0)

    # Week selection (Monday default)
    weekday_num = date.today().weekday()
    default_mon = date.today() - timedelta(days=weekday_num)
    week_start = st.sidebar.date_input("Week start (Monday)", default_mon)

    if st.sidebar.button("Load Week"):
        df = load_week_data_bayou(cust_id, meter_id, week_start, tz_str)
        if df is None or df["kwh"].abs().sum() == 0:
            st.error("No data for selected week.")
            return

        T = compute_target(cust_id, meter_id, week_start, tz_str)
        T_grid = compute_grid_target(cust_id, meter_id, week_start, tz_str)
        st.session_state.df = df
        st.session_state.T = T
        st.session_state.T_grid = T_grid
        st.session_state.tz = tz_str
        st.session_state.idx = 0
        st.session_state.cust_id = cust_id
        st.session_state.meter_id = meter_id

    if "df" in st.session_state:
        df = st.session_state.df
        T_net = st.session_state.T
        T_grid = st.session_state.T_grid
        tz_str = st.session_state.tz
        idx = st.session_state.idx

        # Show data range
        first_ts = df["ts"].min()
        last_ts = df["ts"].max()
        c_start, c_end = st.columns(2)
        c_start.metric("First timestamp", first_ts.strftime("%Y-%m-%d %H:%M"))
        c_end.metric("Last timestamp", last_ts.strftime("%Y-%m-%d %H:%M"))
        st.markdown("---")

        # Interactive slider
        idx = st.slider("Hour of Week", min_value=0, max_value=len(df) - 1, value=idx)
        st.session_state.idx = idx

        # Calculate current metrics
        W_net = df["kwh"].iloc[: idx + 1].sum()
        pts = score_week(W_net, T_net)

        show_grid = st.checkbox("Show Grid Import (vs Last Week)")
        if show_grid:
            W_display = df["cons_kwh"].iloc[: idx + 1].sum()
            T_display = T_grid
            pct = (W_display / T_display) * 100
        else:
            W_display = W_net
            T_display = T_net
            pct = (W_display / T_display) * 100

        # Metrics row
        c1, c2, c3 = st.columns(3)
        c1.metric("Target", f"{T_display:.1f} kWh")
        c2.metric("So far", f"{W_display:.1f} kWh")
        c3.metric("Points", pts)

        if T_display > 0:
            st.progress(min(abs(pct) / 100, 1.0), text=f"{pct:.1f}% of goal consumed")
        else:
            st.progress(min(abs(pct) / 100, 1.0), text=f"{abs(pct):.1f}% of goal exported")

        # Dial visualization
        components.html(_dial_html(W_display, T_display, pct), height=310, width=310, scrolling=False)

        # Hourly chart
        st.subheader("Hourly Consumption vs Production")
        chart = df.set_index("ts")[['cons_kwh', 'prod_kwh']]
        chart.index = pd.to_datetime(chart.index, utc=True).tz_convert(tz_str)
        chart.columns = ["Consumption", "Production"]
        st.line_chart(chart)

        # Insights buttons
        if st.button("Generate Weekly Insights"):
            md = compute_insights_report(df, tz_str)
            st.markdown(md, unsafe_allow_html=True)

            # Get customer name for personalization
            customer = next((c for c in customers if c["id"] == cust_id), None)
            user_name = customer.get("name") if customer else None

            with st.spinner("Creating summary for homeowner..."):
                summary = summarize_for_owner(md, user_name)
            if summary:
                st.info(summary)

            df2 = validate_and_clean(df, tz_str)
            st.subheader("Average Daily Consumption Profile (kWh/hour)")
            st.line_chart(hourly_profile(df2).set_index("hour")["avg_import_kwh"])

        if st.button("Generate Insights So Far"):
            md_so_far = compute_so_far_insights(df, idx, T_net, T_grid, show_grid, tz_str)
            st.markdown(md_so_far, unsafe_allow_html=True)

        if st.button("Download Raw Interval Data CSV"):
            raw = df[["ts", "cons_kwh", "prod_kwh"]].copy()
            raw["net_kwh"] = raw["cons_kwh"] - raw["prod_kwh"]
            st.download_button(
                "Download CSV",
                data=raw.to_csv(index=False),
                file_name="raw_data.csv",
                mime="text/csv",
            )

if __name__ == "__main__":
    main()
