"""Energy insights generation logic."""

import pandas as pd
import numpy as np
from datetime import timedelta, date
from typing import Dict, Any, Optional
from zoneinfo import ZoneInfo
from openai import OpenAI

# Scoring constants
FLOOR_ABS_KWH = 25
DELTA_BAND = 20
BOOTSTRAP_T = 150
MAX_POINTS = 300
PCT_ZERO = 120
EXPORT_CAP = 150
SLOPE_0_50 = 1
SLOPE_50_100 = 2
SLOPE_100_150 = 3


def compute_target(df_prev: Optional[pd.DataFrame]) -> float:
    """
    Compute the target T for the current week based on last week's net.
    If last-week's |T| <= FLOOR_ABS_KWH, clamp to ±FLOOR_ABS_KWH.
    """
    T = df_prev["kwh"].sum() if (df_prev is not None) else 0.0

    if abs(T) <= FLOOR_ABS_KWH:
        T = FLOOR_ABS_KWH if T >= 0 else -FLOOR_ABS_KWH

    return T


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


def validate_and_clean(df: pd.DataFrame, tz_str: str = "UTC") -> pd.DataFrame:
    """Validate and clean dataframe for insights generation."""
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
    """Compute high-level statistics from the data."""
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
    """Find the heaviest 3-hour window on weekdays."""
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
    """Generate hourly consumption profile."""
    prof = df.groupby("hour")["import_kwh"].mean().reset_index()
    prof.columns = ["hour", "avg_import_kwh"]
    return prof


def anomaly_scan(df: pd.DataFrame, z_thresh: float = 3.0) -> pd.DataFrame:
    """Detect anomalies in consumption using z-score."""
    mu, sd = df["import_kwh"].mean(), df["import_kwh"].std()
    df2 = df.copy()
    df2["z"] = (df2["import_kwh"] - mu) / sd
    return df2[abs(df2["z"]) >= z_thresh][["timestamp", "import_kwh", "z"]]


def opportunity_report(df: pd.DataFrame, stats: Dict[str, Any]) -> Dict[str, Any]:
    """Generate opportunity analysis."""
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


def render_markdown(parts: Dict[str, Any], site_info: Optional[Dict[str, Any]]) -> str:
    """Render insights as markdown report."""
    s = parts["stats"]
    common = parts["common"]
    anom = parts["anomalies"]
    opp = parts["opportunity"]

    md = []
    if site_info:
        md.append("### 0  Site Information")
        # Only include city and state for privacy (not full address or ZIP)
        md.append(f"- **City:** {site_info['city']}")
        md.append(f"- **State:** {site_info['state']}")
        if "timezone" in site_info and site_info["timezone"]:
            md.append(f"- **Time Zone:** {site_info['timezone']}")
        md.append("")

    md.append("### 1  Big-picture numbers")
    md.append("| Metric | Value | Comment |")
    md.append("| --- | --- | --- |")
    md.append(f"| Total energy consumed | **{s['total_consumption']:.0f} kWh** | Entire week |")
    md.append(f"| Total grid energy | **{s['total_grid']:.0f} kWh** | What you pay for |")
    md.append(f"| Daily average consumption | {s['daily_avg']:.1f} kWh/day | Peak day = {s['peak_day_name']} ({s['peak_day']}) |")
    md.append(f"| Highest consumption day | {s['peak_day_name']} | Day with max import |")
    ts0 = s["peak_hour_ts"]
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


def compute_insights_report(df: pd.DataFrame, site_info: Optional[Dict[str, Any]], tz_str: str = "UTC") -> str:
    """Generate complete insights report."""
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
    return render_markdown(parts, site_info)


def summarize_for_owner(
    markdown: str,
    openai_api_key: str,
    user_name: Optional[str] = None,
    format: str = "json"
) -> Dict[str, Any] | str:
    """
    Send the markdown report to OpenAI and return an owner-friendly summary.

    Args:
        markdown: The markdown report to summarize
        openai_api_key: OpenAI API key
        user_name: Optional user's name (first name will be extracted)
        format: "json" returns structured dict, "text" returns formatted string

    Returns:
        If format="json": dict with keys: weekly_insight, headline, quick_wins, push_notification, hacker_hints
        If format="text": formatted string with all content
    """
    import json
    from prompts import template_weekly_insights_prompt

    client = OpenAI(api_key=openai_api_key)

    # Extract first name if full name is provided
    first_name = None
    if user_name:
        first_name = user_name.split()[0] if user_name else None

    prompt = template_weekly_insights_prompt(markdown, first_name)

    try:
        resp = client.chat.completions.create(
            model="gpt-5-nano-2025-08-07",
            messages=[{"role": "user", "content": prompt}],
            max_completion_tokens=5000  # GPT-5 nano seems to need higher token limit
        )
        content = resp.choices[0].message.content
        if not content:
            raise ValueError("GPT returned empty response - try increasing max_completion_tokens")

        content = content.strip()

        # Parse JSON response
        try:
            summary_json = json.loads(content)
        except json.JSONDecodeError:
            # If JSON parsing fails, try to extract JSON from markdown code block
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0].strip()
                summary_json = json.loads(content)
            elif "```" in content:
                content = content.split("```")[1].split("```")[0].strip()
                summary_json = json.loads(content)
            else:
                raise ValueError(f"GPT response is not valid JSON. Response was: {content[:500]}")

        # Validate required keys
        required_keys = ["weekly_insight", "headline", "quick_wins", "push_notification", "hacker_hints"]
        missing = [k for k in required_keys if k not in summary_json]
        if missing:
            raise ValueError(f"GPT response missing required keys: {missing}")

        # Return based on format
        if format == "text":
            # Convert JSON to formatted text
            text_parts = [
                f"**WEEKLY INSIGHT**\n{summary_json['weekly_insight']}\n",
                f"**HEADLINE**\n{summary_json['headline']}\n",
                f"**QUICK WINS**"
            ]
            for i, win in enumerate(summary_json['quick_wins'], 1):
                text_parts.append(f"{i}. {win}")
            text_parts.append(f"\n**PUSH NOTIFICATION**\n{summary_json['push_notification']}\n")
            text_parts.append("**HACKER HINTS**")
            for i, hint in enumerate(summary_json['hacker_hints'], 1):
                text_parts.append(f"{i}. {hint}")
            return "\n".join(text_parts)
        else:
            # Return JSON structure
            return summary_json

    except Exception as e:
        raise Exception(f"OpenAI request failed: {e}")
