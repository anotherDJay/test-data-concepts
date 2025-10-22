# bayou_dashboard.py
import os
import datetime as dt
from typing import List, Dict

import pandas as pd
import requests
import streamlit as st

# -----------------------------------------------------------------------------
# 1. Config – API key & environment
# -----------------------------------------------------------------------------
st.set_page_config(page_title="Bayou Energy – Monthly Dashboard", layout="wide")
st.title("🔋 Bayou Energy Dashboard – Current‑Month View")

default_env = "production"  # change to "staging" for testing
bayou_domain = st.sidebar.selectbox("Bayou environment", ["staging", "production"],
                                    index=0 if default_env == "staging" else 1)
bayou_domain = f"{bayou_domain}.bayou.energy" if bayou_domain == "staging" else "bayou.energy"

api_key = (
    st.secrets.get("BAYOU_API_KEY")        # <‑‑ Streamlit Cloud / .streamlit/secrets.toml
    or os.getenv("BAYOU_API_KEY")          # <‑‑ Local dev – export BAYOU_API_KEY=...
)
if not api_key:
    st.error("Add your Bayou API key to run the dashboard.")
    st.stop()

AUTH = (api_key, "")  # Basic‑auth tuple expected by requests

# -----------------------------------------------------------------------------
# 2. Helper functions (cached)
# -----------------------------------------------------------------------------
@st.cache_data(ttl=900)  # 15 min cache
def fetch_customers() -> List[Dict]:
    url = f"https://{bayou_domain}/api/v2/customers"
    resp = requests.get(url, auth=AUTH, timeout=30)
    resp.raise_for_status()
    return resp.json()              # List[dict]

@st.cache_data(ttl=900)
def fetch_bills(cust_id: str) -> List[Dict]:
    url = f"https://{bayou_domain}/api/v2/customers/{cust_id}/bills"
    resp = requests.get(url, auth=AUTH, timeout=30)
    resp.raise_for_status()
    return resp.json()              # List[dict]

@st.cache_data(ttl=900)
def fetch_intervals(cust_id: str) -> Dict:
    url = f"https://{bayou_domain}/api/v2/customers/{cust_id}/intervals"
    resp = requests.get(url, auth=AUTH, timeout=60)
    resp.raise_for_status()
    return resp.json()              # {"meters":[…]}

# -----------------------------------------------------------------------------
# 3. Pick a customer
# -----------------------------------------------------------------------------
customers = fetch_customers()
if not customers:
    st.warning("No customers found for this key/environment.")
    st.stop()

cust_options = {
    f"{c.get('external_id') or c['id']} – {c.get('email','<no‑email>')}": c["id"]
    for c in customers
}
cust_label = st.sidebar.selectbox("Customer", sorted(cust_options))
cust_id = cust_options[cust_label]
st.subheader(f"Customer **{cust_id}**")

# -----------------------------------------------------------------------------
# 4. Time window helpers
# -----------------------------------------------------------------------------
today = dt.date.today()
month_start = today.replace(day=1)
next_month_start = (month_start + dt.timedelta(days=32)).replace(day=1)

def ts_to_dt(ts: str) -> dt.datetime:
    # Bayou returns ISO 8601 strings
    return dt.datetime.fromisoformat(ts.replace("Z", "+00:00"))

# -----------------------------------------------------------------------------
# 5. Bills – table + metrics
# -----------------------------------------------------------------------------
bills_raw = fetch_bills(cust_id)

def bill_in_month(b: Dict) -> bool:
    # Most bills include `billed_on`, `start_date`, `end_date`
    start = ts_to_dt(b.get("start_date") or b.get("billed_on")).date()
    end = ts_to_dt(b.get("end_date") or b.get("billed_on")).date()
    return not (end < month_start or start >= next_month_start)

bills_month = [b for b in bills_raw if bill_in_month(b)]

st.markdown("### 📄 Bills – Current Month")
if bills_month:
    bills_df = pd.json_normalize(bills_month)
    # Human‑friendly subset of columns
    show_cols = ["id", "start_date", "end_date", "amount_cents", "usage_kwh", "pdf_url"]
    show_cols = [c for c in show_cols if c in bills_df.columns]
    st.dataframe(bills_df[show_cols].rename(columns={
        "amount_cents": "amt ¢", "usage_kwh": "kWh"}))

    col1, col2 = st.columns(2)
    col1.metric("Total cost (USD)", f"${bills_df['amount_cents'].sum()/100:,.2f}")
    col2.metric("Total usage (kWh)", f"{bills_df['usage_kwh'].sum():,.0f}")
else:
    st.info("No bills overlap this month (yet).")

# -----------------------------------------------------------------------------
# 6. Interval data – per meter line chart
# -----------------------------------------------------------------------------
intervals_json = fetch_intervals(cust_id)
st.markdown("### 📈 Interval Usage – Current Month (daily kWh)")

for meter in intervals_json.get("meters", []):
    m_id = meter.get("id")
    ivals = meter.get("intervals", [])
    # Build DF
    df = pd.DataFrame(ivals)
    if df.empty:
        continue
    df["start"] = pd.to_datetime(df["start"])
    month_mask = (df["start"].dt.date >= month_start) & (df["start"].dt.date < next_month_start)
    df = df.loc[month_mask]

    if df.empty:
        continue

    # Aggregate kWh (or whatever unit) by day
    df_day = df.groupby(df["start"].dt.date)["value"].sum().reset_index()
    df_day.rename(columns={"start": "day", "value": "kWh"}, inplace=True)

    st.markdown(f"**Meter {m_id}**")
    st.line_chart(df_day.set_index("day"))        # x‑axis auto‑formatted
    with st.expander("Raw rows"):
        st.dataframe(df.loc[:, ["start", "value", "unit"]])

if not intervals_json.get("meters"):
    st.info("No interval data available for this customer yet.")

# -----------------------------------------------------------------------------
# 7. Developer panel
# -----------------------------------------------------------------------------
with st.expander("ℹ️ API Calls & Raw JSON"):
    st.write(f"Environment: `{bayou_domain}`")
    st.code(f"GET /api/v2/customers/{cust_id}/bills\nGET /api/v2/customers/{cust_id}/intervals")
    st.json({"bills": bills_month, "intervals": intervals_json})
