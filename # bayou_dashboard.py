# bayou_dashboard.py  •  Streamlit 1‑file demo               2025‑07‑18
import os, time, datetime as dt, json, requests, pandas as pd, altair as alt, streamlit as st
from typing import List, Dict
# ── CONFIG ────────────────────────────────────────────────────────────────────
st.set_page_config(page_title="Palmetto × Bayou Dashboard", layout="wide")
ENV = st.sidebar.selectbox("Environment", ["production", "staging"], index=0)
DOMAIN = "bayou.energy" if ENV == "production" else "staging.bayou.energy"
API_KEY = st.secrets.get("BAYOU_API_KEY") or os.getenv("BAYOU_API_KEY")
if not API_KEY: st.error("Set BAYOU_API_KEY"); st.stop()
AUTH = (API_KEY, "")
TODAY = dt.date.today()
MONTH_START = TODAY.replace(day=1)
NEXT_MONTH  = (MONTH_START + dt.timedelta(days=32)).replace(day=1)

# ── SAFE REQUEST HELPERS ─────────────────────────────────────────────────────
def _safe(method, url, **k):
    try:
        r = requests.request(method, url, auth=AUTH, timeout=30, **k)
        if r.status_code == 401:
            st.error("🔐 401 – key/environment mismatch"); st.stop()
        r.raise_for_status(); return r
    except requests.HTTPError as e:
        st.error(f"{method} {url} → {e}"); st.stop()

def jget(path, **q):    return _safe("GET", f"https://{DOMAIN}{path}", params=q).json()
def jpost(path, **j):   return _safe("POST",f"https://{DOMAIN}{path}", json=j).json()
def jdel(path):         return _safe("DELETE",f"https://{DOMAIN}{path}").json()

# ── CACHE LAYERS (invalidate via st.button) ───────────────────────────────────
@st.cache_data(ttl=900)       # 15 min
def customers(): return jget("/api/v2/customers")

@st.cache_data(ttl=900)
def utilities(): return jget("/api/v2/utilities")

@st.cache_data(ttl=900)
def customer(cid): return jget(f"/api/v2/customers/{cid}")

@st.cache_data(ttl=900)
def bills(cid): return jget(f"/api/v2/customers/{cid}/bills")

@st.cache_data(ttl=900)
def intervals(cid): return jget(f"/api/v2/customers/{cid}/intervals")

def clear_caches():
    st.cache_data.clear()   # one‑liner to purge all above

# ── SIDEBAR: CREATE OR PICK CUSTOMER ─────────────────────────────────────────
st.sidebar.markdown("## ➕ Create customer")
with st.sidebar.form("create"):
    u_map = {u["name"]: u["slug"] for u in utilities()}
    util  = st.selectbox("Utility", list(u_map))
    email = st.text_input("E‑mail (optional)")
    x_id  = st.text_input("External ID (optional)")
    if st.form_submit_button("Create"):
        body = {"utility": u_map[util]} | ({ "email": email } if email else {}) | ({ "external_id": x_id } if x_id else {})
        c    = jpost("/api/v2/customers", **body)
        st.sidebar.success("Send this onboarding link:")
        st.sidebar.code(c["onboarding_link"])

st.sidebar.divider()
C_MAP = {f"{c.get('external_id') or c['id']} – {c.get('email','<no email>')}": c["id"] for c in customers()}
C_SEL = st.sidebar.selectbox("Current customer", sorted(C_MAP))
CID   = C_MAP[C_SEL]

# ── DANGER ZONE ───────────────────────────────────────────────────────────────
with st.sidebar.expander("🗑️ Delete customer"):
    if st.button("Delete", help="Irreversible!"):
        jdel(f"/api/v2/customers/{CID}"); st.sidebar.success("Deleted"); clear_caches(); st.experimental_rerun()

# ── REFRESH BUTTONS ───────────────────────────────────────────────────────────
st.sidebar.divider()
if st.sidebar.button("🔄 Force refresh API caches"): clear_caches(); st.experimental_rerun()

# ── DATA FETCH ───────────────────────────────────────────────────────────────
cust       = customer(CID)
raw_bills  = bills(CID)
raw_int    = intervals(CID)

# ── WEBHOOK‑LIKE STATE FLAGS (polling) ────────────────────────────────────────
HAS_BILLS      = bool(raw_bills)
HAS_INTERVALS  = any(m["intervals"] for m in raw_int.get("meters", []))
NEEDS_REAUTH   = cust.get("needs_reauthentication", False)   # fits webhook step 13
# unlocked?  find at least one bill with unlocked==True
UNLOCKED       = any(b.get("is_unlocked") for b in raw_bills)

# ── BADGES ────────────────────────────────────────────────────────────────────
st.subheader(f"Customer **{CID}**")
b1,b2,b3,b4 = st.columns(4)
badge = lambda col, txt, yes: col.metric(txt, "✅" if yes else "—", "", delta_color="off")
badge(b1,"Bills",      HAS_BILLS)
badge(b2,"Intervals",  HAS_INTERVALS)
badge(b3,"Unlocked",   UNLOCKED)
badge(b4,"Re‑auth",    False if not NEEDS_REAUTH else "⚠️")

# ── STEP 9 / 10 : METER PICK + UNLOCK  ───────────────────────────────────────
st.markdown("### 🔓 Unlock data")
if not HAS_BILLS:
    st.info("Waiting for bills_ready …")
else:
    # gather {meter_id → addr}
    meter_addr = {}
    for acc in cust.get("accounts", []):
        for m in acc.get("meters", []):
            meter_addr[m["id"]] = m.get("cleaned_address") or m.get("address") or "Unknown address"
    if not meter_addr:
        st.warning("Customer hierarchy not loaded yet."); st.stop()

    MSEL = st.selectbox("Choose meter", [f"{m}  –  {a}" for m,a in meter_addr.items()], key="meter")
    M_ID = MSEL.split()[0]

    # pick latest bill for that meter
    mbills = [b for b in raw_bills if b["meter_id"]==M_ID]
    latest = max(mbills, key=lambda b: b["billed_on"]) if mbills else None

    if not latest:
        st.warning("No bills for that meter yet.")
    else:
        colL, colR = st.columns([3,1])
        colL.write(f"Latest bill **{latest['id']}** billed on {latest['billed_on'][:10]}")
        if not latest.get("is_unlocked"):
            if colR.button("Unlock latest bill"):
                jpost(f"/api/v2/bills/{latest['id']}/unlock", meter_id=M_ID)
                st.success("Unlock requested"); clear_caches(); st.experimental_rerun()
        else:
            colR.success("✔ Unlocked")

# ── SECTION 6 : BILLS TABLE (kWh converter) ──────────────────────────────────
st.markdown("### 📄 Bills – current month")
for b in raw_bills:
    if "usage_kwh" not in b and "usage_wh" in b:
        b["usage_kwh"] = round(b["usage_wh"]/1_000,3)
BM = [b for b in raw_bills if MONTH_START<=dt.date.fromisoformat(b["billed_on"][:10])<NEXT_MONTH]
if BM:
    dfB = pd.json_normalize(BM)
    st.dataframe(dfB[["id","billed_on","amount_cents","usage_kwh","is_unlocked"]])
    t1,t2 = st.columns(2)
    t1.metric("Cost", f"${dfB['amount_cents'].sum()/100:,.2f}")
    t2.metric("Usage",f"{dfB['usage_kwh'].sum():,.0f} kWh")
else:
    st.info("No bills this month.")

# ── SECTION 7 : INTERVAL VISUALS (schema‑aware) ──────────────────────────────
st.markdown("### 📈 Interval usage")
if not HAS_INTERVALS:
    st.info("No intervals yet.")
else:
    for m in raw_int["meters"]:
        df = pd.DataFrame(m["intervals"])
        if df.empty: continue
        df["start"] = pd.to_datetime(df["start"])
        # map numeric field → value (kWh)
        if "value" not in df.columns:
            for alt in ["net_electricity_consumption","electricity_consumption","generated_electricity"]:
                if alt in df.columns:
                    df["value"] = df[alt]/1_000; break
        if "value" not in df.columns: continue
        dfM = df[(df["start"].dt.date>=MONTH_START)&(df["start"].dt.date<NEXT_MONTH)]
        if dfM.empty: continue

        daily = (dfM.groupby(dfM["start"].dt.date)["value"].sum().reset_index()
                 .rename(columns={"start":"day","value":"kWh"}))
        st.markdown(f"#### Meter {m['id']}")
        st.line_chart(daily.set_index("day"), height=150)
        plot = alt.Chart(dfM.rename(columns={"start":"ts","value":"kWh"})).mark_line().encode(
            x="ts:T", y="kWh:Q", tooltip=["ts:T","kWh:Q"]).interactive()
        st.altair_chart(plot,use_container_width=True)
        with st.expander("Raw intervals"): st.dataframe(dfM)

# ── MANUAL REFRESH (interval ping) ────────────────────────────────────────────
if st.button("🔄 Ping Bayou for fresh intervals"):
    clear_caches(); st.experimental_rerun()

# ── DEBUG JSON ───────────────────────────────────────────────────────────────
with st.expander("Debug payloads"):
    st.json({"cust": cust, "bills": raw_bills[:2], "intervals": raw_int[:1]})