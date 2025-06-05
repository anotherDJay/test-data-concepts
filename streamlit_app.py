import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, date
from typing import Optional, Dict, Any
import re
from zoneinfo import ZoneInfo
from timezonefinder import TimezoneFinder
from svg_wedges import WEDGE_DATA_JS
import streamlit.components.v1 as components

# Snowpark imports
from snowflake.snowpark.session import Session
from snowflake.snowpark.functions import col, lit, array_contains

# --------------------------------------------------
# Function to create and cache Snowflake session
# --------------------------------------------------
@st.cache_resource  # Cache the session across reruns
def create_snowpark_session() -> Optional[Session]:
    """
    Creates a Snowpark session using credentials from st.secrets.
    Returns None if creation fails or secrets are missing.
    """
    try:
        if "snowflake" in st.secrets:
            connection_parameters = st.secrets["snowflake"]
            session = Session.builder.configs(connection_parameters).create()
            st.success("Snowpark session created successfully!")
            return session
        else:
            st.error("Snowflake connection details not found in st.secrets.")
            return None
    except Exception as e:
        st.error(f"Error creating Snowpark session: {e}")
        return None

# --------------------------------------------------
# Bubble/Grid Visualization
# --------------------------------------------------
BUBBLE_CSS = """
<style>
  .b{width:14px;height:14px;border-radius:50%;display:inline-block;margin:1px;}
  .lo{background:#FBD1B5;}    /* light-orange */
  .do{background:#F47B60;}    /* dark-orange */
  .green{background:#A7E299;} /* export */
  .empty{background:#F0F0F0;border:1px solid #e5e5e5;}
</style>
"""

# Timezone lookup helper
TF = TimezoneFinder()

def bubble_grid(W: float, T: float) -> str:
    """
    Render a 10x10 grid of 'bubbles' to visualize progress towards T.
    - If W >= 0: bubbles start as 'lo' and turn 'empty' up to 100%; beyond 100% they turn 'do'.
    - If W < 0: bubbles fill from the top as 'green' (exporting), plus a bonus row of up to 50.
    """
    cells = ["lo"] * 100
    pct = (W / T * 100) if T else 0.0

    if W >= 0:
        # “Importers”: up to 110% of T turns bubbles empty (first 100) then 'do'
        steps = min(110, int(pct))
        for i in range(steps):
            cells.pop()
            cells.insert(0, "empty" if i < 100 else "do")
    else:
        # “Exporters”: each kWh exported (up to 100%) removes a 'lo' bubble and appends a 'green'
        for _ in range(min(100, int(abs(pct)))):
            cells.pop(0)
            cells.append("green")

    html = BUBBLE_CSS + '<div style="line-height:0.7">'
    for idx, cls in enumerate(cells):
        html += f"<span class='b {cls}'></span>"
        if (idx + 1) % 10 == 0:
            html += '<br>'

    # If exporting beyond 100%, draw a bonus row (max 50 green bubbles)
    if W < 0 and abs(pct) > 100:
        bonus = int(min(50, abs(pct) - 100))
        html += '<div style="line-height:0.7; margin-top:4px;">'
        html += ''.join(["<span class='b green'></span>" for _ in range(bonus)])
        html += '</div>'

    html += '</div>'
    return html

# ──────────────────────────────────────────────────────────────
# 2.  DIAL HTML GENERATOR  ── paste anywhere after imports
# ──────────────────────────────────────────────────────────────
def _dial_html(W: float, T: float, pct: float) -> str:              # ▶ ADD
    """
    Build the exact same radial dial used in the React prototype.
    • Black base ring
    • Purple wedges fill CCW when exporting (W < 0)
    • Grey  wedges fill  CW when importing (W ≥ 0)
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
{WEDGE_DATA_JS}   /* ← 100 SVG wedges injected verbatim */

const filledPercentage = {pct:.2f};
const mode             = "{mode}";
const filledWedges     = Math.floor(filledPercentage);
const startWedge       = Math.max(1, 101 - filledWedges);   // export fill start

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
  const p = document.createElementNS(svgNS,"path");
  p.setAttribute("d", w.path);

  const filled = (mode==="export")
       ? w.id >= startWedge        // CCW fill
       : w.id <= filledWedges;     //  CW fill

  p.setAttribute("fill",
       filled ? (mode==="export" ? "#815ED5" : "#ACACAC") : "#1B240F");

  svg.appendChild(p);
}});
</script>
"""

# --------------------------------------------------
# Data Fetchers (using create_snowpark_session())
# --------------------------------------------------
@st.cache_data
def get_sites() -> list:
    """
    Return a sorted list of all distinct HELIOS_SITE_ID values
    from the HOURLY_CONSUMPTION table in Snowflake.
    """
    session = create_snowpark_session()
    if session is None:
        return []

    try:
        df = (
            session
            .table("ENERGY_SHARED.TRUNKS_HELIOS.HOURLY_CONSUMPTION")
            .select("HELIOS_SITE_ID")
            .distinct()
            .to_pandas()
        )
        return sorted(df["HELIOS_SITE_ID"].astype(str).tolist())
    except Exception as e:
        st.error(f"Error fetching site list: {e}")
        return []


@st.cache_data
def load_week_data(site_id: str, week_start: date, tz_str: str = "UTC") -> Optional[pd.DataFrame]:
    """
    Load one week of consumption + production data for a given site.
    Returns a pandas DataFrame with columns:
      ['ts', 'cons_kwh', 'prod_kwh', 'kwh']  (all lowercase).
    If no rows, returns None.
    """
    session = create_snowpark_session()
    if session is None:
        return None

    start = datetime.combine(week_start, datetime.min.time())
    end = start + timedelta(days=7)

    try:
        cons = (
            session
            .table("ENERGY_SHARED.TRUNKS_HELIOS.HOURLY_CONSUMPTION")
            .filter((col("HELIOS_SITE_ID") == site_id) & 
                    (col("CONSUMPTION_ON") >= lit(start)) & 
                    (col("CONSUMPTION_ON") < lit(end)))
            .select(
                col("CONSUMPTION_ON").alias("ts"),
                col("CONSUMPTION_KWH").alias("cons_kwh")
            )
        )

        prod = (
            session
            .table("ENERGY_SHARED.TRUNKS_HELIOS.HOURLY_PRODUCTION")
            .filter((col("HELIOS_SITE_ID") == site_id) & 
                    (col("PRODUCTION_ON") >= lit(start)) & 
                    (col("PRODUCTION_ON") < lit(end)))
            .select(
                col("PRODUCTION_ON").alias("ts"),
                col("PRODUCTION_KWH").alias("prod_kwh")
            )
        )

        # Full outer join on 'ts'
        joined = cons.join(prod, on="ts", how="full")
        df = joined.to_pandas()
    except Exception as e:
        st.error(f"Error loading week data: {e}")
        return None

    if df.empty:
        return None

    # Lowercase column names
    df.columns = [c.lower() for c in df.columns]

    # Convert timestamps from UTC to local timezone and sort
    df["ts"] = pd.to_datetime(df["ts"], utc=True).dt.tz_convert(tz_str)
    df = df.sort_values("ts").reset_index(drop=True)

    # Fill missing consumption/production with 0
    df[["cons_kwh", "prod_kwh"]] = df[["cons_kwh", "prod_kwh"]].fillna(0)

    # Compute net kWh (import minus export)
    df["kwh"] = df["cons_kwh"] - df["prod_kwh"]
    return df


@st.cache_data
def compute_target(site_id: str, week_start: date, tz_str: str = "UTC") -> float:
    """
    Compute the target T for the current week based on last week's net.
    If last-week’s |T| <= FLOOR_ABS_KWH, clamp to ±FLOOR_ABS_KWH.
    """
    prev_start = week_start - timedelta(days=7)
    df_prev = load_week_data(site_id, prev_start, tz_str)

    T = df_prev["kwh"].sum() if (df_prev is not None) else 0.0

    if abs(T) <= FLOOR_ABS_KWH:
        T = FLOOR_ABS_KWH if T >= 0 else -FLOOR_ABS_KWH

    return T


@st.cache_data
\
def get_site_info(site_id: str) -> Optional[Dict[str, Any]]:
    """
    Fetch the STREET_ADDRESS, CITY, and ZIP_CODE for a given site_id from
    ENERGY_SHARED.TRUNKS_HELIOS.PROPERTIES using LATERAL FLATTEN on HELIOS_SITE_IDS.
    Debug-print the resulting DataFrame so we can verify the query output.
    """
    session = create_snowpark_session()
    if session is None:
        return None

    query = f"""
    SELECT
      p.STREET_ADDRESS AS address,
      p.CITY AS city,
      p.STATE as state,
      p.COORDINATES as coordinates,
      p.ZIP_CODE AS zip
    FROM ENERGY_SHARED.TRUNKS_HELIOS.PROPERTIES p,
         LATERAL FLATTEN(input => p.HELIOS_SITE_IDS) f
    WHERE f.value = '{site_id}'
    LIMIT 1;
    """

    try:
        df = session.sql(query).to_pandas()
        # Normalize column names to lowercase
        df.columns = [c.lower() for c in df.columns]
    except Exception as e:
        st.error(f"Error fetching site info: {e}")
        return None

    if df.empty:
        st.write("No matching site_info rows returned.")
        return None

    row = df.iloc[0]
    tz = None
    coords = row.get("coordinates")
    if coords is not None:
        lon = lat = None
        if isinstance(coords, (list, tuple)) and len(coords) >= 2:
            lat, lng = float(coords[0]), float(coords[1])
        elif isinstance(coords, str):
            m = re.search(r"(-?\d+(?:\.\d+)?)[^\d-]+(-?\d+(?:\.\d+)?)", coords)
            if m:
                lat, lng = float(m.group(1)), float(m.group(2))
        if lat is not None and lng is not None:
            tz = TF.timezone_at(lng=lng, lat=lat)

    return {
        "address": row["address"],
        "city": row["city"],
        "state": row["state"],
        "zip": row["zip"],
        "timezone": tz if tz else "UTC",
    }

# --------------------------------------------------
# Weekly Score Algorithm Constants
# --------------------------------------------------
FLOOR_ABS_KWH = 25
DELTA_BAND    = 20
BOOTSTRAP_T   = 150
MAX_POINTS    = 300
PCT_ZERO      = 120
EXPORT_CAP    = 150
SLOPE_0_50    = 1
SLOPE_50_100  = 2
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
        if d >  DELTA_BAND:
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

# --------------------------------------------------
# Insight Helpers
# --------------------------------------------------
def validate_and_clean(df: pd.DataFrame, tz_str: str = "UTC") -> pd.DataFrame:
    assert {"ts", "cons_kwh"}.issubset(df.columns), "Missing ts or cons_kwh"
    df2 = df.rename(columns={"ts": "timestamp", "cons_kwh": "import_kwh"}).copy()
    df2 = df2.sort_values("timestamp")
    # Ensure timestamp is timezone-aware
    df2["timestamp"] = pd.to_datetime(df2["timestamp"], utc=True).dt.tz_convert(tz_str)
    df2["import_kwh"] = df2["import_kwh"].fillna(0)
    df2["export_kwh"] = df2.get("prod_kwh", 0).fillna(0)
    df2["net_kwh"]    = df2["import_kwh"] - df2["export_kwh"]
    df2["day"]        = df2["timestamp"].dt.date
    df2["hour"]       = df2["timestamp"].dt.hour

    iv = df2["timestamp"].diff().dropna().dt.total_seconds().mode().iloc[0] / 3600
    df2.attrs["interval_hours"] = iv
    return df2


def high_level_stats(df: pd.DataFrame) -> Dict[str, Any]:
    total = df["import_kwh"].sum()
    total_grid = (df["import_kwh"] - df["export_kwh"]).sum()
    daily = df.groupby("day")["import_kwh"].sum()
    avg   = daily.mean()
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


def anomaly_scan(df: pd.DataFrame, z_thresh: float=3.0) -> pd.DataFrame:
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
        note = f"Self-consumption ratio ~{sc:.2f}."
    return {"shift_savings": shift, "battery_kwh": bat, "solar_note": note}


def render_markdown(parts: Dict[str, Any], site_info: Optional[Dict[str, Any]]) -> str:
    s      = parts["stats"]
    common = parts["common"]
    anom   = parts["anomalies"]
    opp    = parts["opportunity"]

    md = []
    if site_info:
        md.append("### 0  Site Information")
        md.append(f"- **Address:** {site_info['address']}")
        md.append(f"- **City:** {site_info['city']}")
        md.append(f"- **State:** {site_info['state']}")
        md.append(f"- **ZIP:** {site_info['zip']}")
        # Add time zone info if available
        if "timezone" in site_info and site_info["timezone"]:
            md.append(f"- **Time Zone:** {site_info['timezone']}")
        md.append("")

    md.append("### 1  Big-picture numbers")
    md.append("| Metric | Value | Comment |")
    md.append("| --- | --- | --- |")
    md.append(f"| Total energy consumed | **{s['total_consumption']:.0f} kWh** | Entire week |")
    md.append(f"| Total  grid energy | **{s['total_grid']:.0f} kWh** | What you pay for |")
    md.append(f"| Daily average consumption | {s['daily_avg']:.1f} kWh/day | Peak day = {s['peak_day_name']} ({s['peak_day']}) |")
    md.append(f"| Highest consumption day | {s['peak_day_name']} | Day with max import |")
    ts0    = s["peak_hour_ts"]
    ts_str = ts0.strftime("%Y-%m-%d %H:%M")
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
def compute_insights_report(df: pd.DataFrame, site_id: str, site_info: Optional[Dict[str, Any]], tz_str: str = "UTC") -> str:
    df2    = validate_and_clean(df, tz_str)
    stats  = high_level_stats(df2)
    common = weekday_heaviest_3h(df2)
    anom   = anomaly_scan(df2)
    opp    = opportunity_report(df2, stats)

    parts = {
        "stats": stats,
        "common": common,
        "hourly": hourly_profile(df2),
        "anomalies": anom,
        "opportunity": opp
    }
    return render_markdown(parts, site_info)

# --------------------------------------------------
# Streamlit App Main
# --------------------------------------------------

def main():
    st.title("🔋 Bubble Budget Scrubber + Insights")

    sites = get_sites()
    if not sites:
        st.warning("No sites available. Check your Snowflake connection.")
        return

    site = st.selectbox("Helios Site", sites)

    weekday_num = date.today().weekday()
    days_since_sunday = (weekday_num + 1) % 7
    default_sun = date.today() - timedelta(days=days_since_sunday)

    week_start = st.date_input("Week start (Sunday)", default_sun)

    if st.button("Load Week"):
        site_info = get_site_info(site)
        tz_str = site_info.get("timezone", "UTC") if site_info else "UTC"
        df = load_week_data(site, week_start, tz_str)
        if df is None or df["kwh"].abs().sum() == 0:
            st.error("No data for selected week.")
            return

        T = compute_target(site, week_start, tz_str)
        st.session_state.df = df
        st.session_state.T = T
        st.session_state.site_info = site_info
        st.session_state.tz = tz_str
        st.session_state.idx = 0

    if "df" in st.session_state:
        df, T, idx, site_info, tz_str = (
            st.session_state.df,
            st.session_state.T,
            st.session_state.idx,
            st.session_state.site_info,
            st.session_state.tz,
        )

        idx = st.slider("Hour of Week", min_value=0, max_value=len(df) - 1, value=idx)
        st.session_state.idx = idx

        W = df["kwh"].iloc[: idx + 1].sum()
        pts = score_week(W, T)
        pct = (W/T)*100 if T!=0 else 0

        c1, c2, c3 = st.columns(3)
        c1.metric("Target", f"{T:.1f} kWh")
        c2.metric("Net so far", f"{W:.1f} kWh")
        c3.metric("Points", pts)
        if T > 0:
            st.progress(min(abs(pct)/100,1.0), text=f"{pct:.1f}% of goal consumed")
        else:
            st.progress(min(abs(pct)/100,1.0), text=f"{abs(pct):.1f}% of goal exported")

        # st.markdown(bubble_grid(W, T), unsafe_allow_html=True)
        components.html(_dial_html(W, T, pct), height=310, width=310, scrolling=False)

        st.subheader("Hourly Consumption vs Production")
        chart = df.set_index("ts")[['cons_kwh', 'prod_kwh']]
        chart.index = pd.to_datetime(chart.index, utc=True).tz_convert(tz_str)  # Ensure tz-aware
        chart.columns = ["Consumption", "Production"]
        st.line_chart(chart)

        if st.button("Generate Weekly Insights"):
            md = compute_insights_report(df, site, site_info, tz_str)
            st.markdown(md, unsafe_allow_html=True)

            df2 = validate_and_clean(df, tz_str)
            st.subheader("Average Daily Consumption Profile (kWh/hour)")
            st.line_chart(hourly_profile(df2).set_index("hour")["avg_import_kwh"])

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

