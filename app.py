import json
import numpy as np
import pandas as pd
import streamlit as st
import altair as alt

st.set_page_config(page_title="Climate & Animal Health", page_icon="🐄", layout="wide")

@st.cache_data
def load_data():
    return pd.read_csv("data/animal_health_climate_sample.csv")

@st.cache_data
def load_model():
    with open("model/model_coefficients.json") as f:
        return json.load(f)

df, model = load_data(), load_model()

# ---------- Header ----------
st.title("🐄 Does climate change affect animal disease?")
st.caption("A small learning project inspired by WOAH's work linking WAHIS disease records with climate data.")
st.warning("**Honest note:** the data here is SIMULATED (made up on purpose) so anyone can run this. "
           "It is not real WAHIS data. The goal is to show the method, not to report real findings.")

tab1, tab2, tab3, tab4, tab5 = st.tabs(
    ["1. The idea", "2. Look at the data", "3. Climate vs outbreaks", "4. What-if tool", "5. Limits & next steps"])

# ---------- Tab 1 ----------
with tab1:
    st.header("The idea in plain English")
    st.markdown("""
**The question:** when a country gets warmer or wetter than usual, do more animal disease outbreaks get reported?

**Why it matters:** warmer weather can help insects (like midges and mosquitoes) spread diseases, and dry or wet
spells can change how animals and germs meet. But very few studies have tested this with long, official records.

**What this app does (the same steps a real analysis follows):**
1. **Data** - one row per country, per year, per disease: outbreak count + temperature, rain, number of animals, and how well the country reports.
2. **Model** - a *Poisson regression*, a standard tool for **counts** (like number of outbreaks).
3. **Result** - "For every +1°C warmer than normal, outbreaks change by X%."
4. **Care** - we also include *reporting quality*, because countries often report **more** over time, which can look like a climate effect but is not.
""")
    c1, c2, c3 = st.columns(3)
    c1.metric("Countries", df["country"].nunique())
    c2.metric("Years", f'{df["year"].min()}-{df["year"].max()}')
    c3.metric("Diseases", df["disease"].nunique())

# ---------- Tab 2 ----------
with tab2:
    st.header("Look at the data")
    col1, col2 = st.columns(2)
    disease = col1.selectbox("Disease", sorted(df["disease"].unique()), key="d2")
    regions = col2.multiselect("Regions", sorted(df["region"].unique()), default=sorted(df["region"].unique()))
    d = df[(df["disease"] == disease) & (df["region"].isin(regions))]

    st.subheader("Reported outbreaks per year")
    yearly = d.groupby("year")["outbreaks"].sum().reset_index()
    st.altair_chart(alt.Chart(yearly).mark_line(point=True).encode(
        x=alt.X("year:O", title="Year"), y=alt.Y("outbreaks:Q", title="Total outbreaks")), width="stretch")

    st.subheader("Temperature anomaly per year")
    st.caption("Anomaly = how much warmer than the country's normal. It rises over time in this data.")
    temp = d.groupby("year")["temp_anomaly_c"].mean().reset_index()
    st.altair_chart(alt.Chart(temp).mark_line(point=True, color="firebrick").encode(
        x=alt.X("year:O", title="Year"), y=alt.Y("temp_anomaly_c:Q", title="Average anomaly (°C)")), width="stretch")

    st.subheader("Reporting score over time")
    st.caption("A country that reports better will show more outbreaks even if nothing changed in reality.")
    rep = d.groupby("year")["reporting_score"].mean().reset_index()
    st.altair_chart(alt.Chart(rep).mark_line(point=True, color="seagreen").encode(
        x=alt.X("year:O", title="Year"), y=alt.Y("reporting_score:Q", title="Average reporting score (0-1)")), width="stretch")

    with st.expander("See the raw table"):
        st.dataframe(d, width="stretch")
        st.download_button("Download CSV", d.to_csv(index=False), "data_subset.csv")

# ---------- Tab 3 ----------
with tab3:
    st.header("Climate vs outbreaks")
    rows = []
    for dis, coefs in model.items():
        for var, label, unit in [("temp_anomaly_c", "+1°C warmer", 1), ("rain_anomaly_10mm", "+10 mm rain per month", 1)]:
            c = coefs[var]
            rows.append(dict(disease=dis, driver=label,
                             change_pct=100 * (np.exp(c["coef"]) - 1),
                             low=100 * (np.exp(c["ci_low"]) - 1),
                             high=100 * (np.exp(c["ci_high"]) - 1),
                             p_value=c["p_value"],
                             clear_signal="Yes" if c["p_value"] < 0.05 else "No"))
    res = pd.DataFrame(rows)

    driver = st.radio("Climate driver", res["driver"].unique(), horizontal=True)
    r = res[res["driver"] == driver]
    base = alt.Chart(r).encode(y=alt.Y("disease:N", title=None))
    chart = (base.mark_rule().encode(x=alt.X("low:Q", title="Change in outbreaks (%)"), x2="high:Q") +
             base.mark_circle(size=140).encode(x="change_pct:Q", color=alt.Color("clear_signal:N",
                    scale=alt.Scale(domain=["Yes", "No"], range=["#d62728", "#999999"]), title="Clear signal?")) +
             alt.Chart(pd.DataFrame({"x": [0]})).mark_rule(strokeDash=[4, 4]).encode(x="x:Q"))
    st.altair_chart(chart, width="stretch")
    st.caption("Dot = best estimate. Line = range of plausible values. If the line crosses the dashed zero line, "
               "we cannot say there is a real effect.")

    for _, row in r.iterrows():
        direction = "more" if row["change_pct"] > 0 else "fewer"
        if row["clear_signal"] == "Yes":
            st.success(f"**{row['disease']}**: {driver} goes with about **{abs(row['change_pct']):.0f}% {direction}** outbreaks.")
        else:
            st.info(f"**{row['disease']}**: no clear link found (estimate {row['change_pct']:+.0f}%, range {row['low']:+.0f}% to {row['high']:+.0f}%).")

    st.subheader("Check it yourself")
    dis = st.selectbox("Disease", sorted(df["disease"].unique()), key="d3")
    x = st.selectbox("Climate variable", ["temp_anomaly_c", "rain_anomaly_10mm"], key="x3")
    sub = df[df["disease"] == dis]
    sc = alt.Chart(sub).mark_circle(opacity=0.4).encode(x=x, y="outbreaks:Q", tooltip=["country", "year", "outbreaks"])
    st.altair_chart(sc + sc.transform_regression(x, "outbreaks", method="poly", order=1).mark_line(color="red"),
                    width="stretch")
    with st.expander("Full model numbers"):
        st.json(model[dis])

# ---------- Tab 4 ----------
with tab4:
    st.header("What-if tool")
    st.write("Pick a disease and a situation. The model estimates how many outbreaks a country might report in a year.")
    dis = st.selectbox("Disease", sorted(model.keys()), key="d4")
    a, b = st.columns(2)
    t = a.slider("Temperature vs normal (°C)", -2.0, 4.0, 1.0, 0.1)
    rn = b.slider("Rain vs normal (mm per month)", -30, 30, 0, 5)
    live = a.slider("Livestock index (1 = average)", 0.3, 2.0, 1.0, 0.1)
    rep = b.slider("Reporting quality (0 = poor, 1 = excellent)", 0.1, 1.0, 0.6, 0.05)

    c = model[dis]
    log_mu = (c["const"]["coef"] + c["temp_anomaly_c"]["coef"] * t + c["rain_anomaly_10mm"]["coef"] * rn / 10
              + c["log_reporting"]["coef"] * np.log(rep) + c["log_livestock"]["coef"] * np.log(live))
    base_mu = (c["const"]["coef"] + c["log_reporting"]["coef"] * np.log(rep) + c["log_livestock"]["coef"] * np.log(live))
    m1, m2 = st.columns(2)
    m1.metric("Expected outbreaks (this scenario)", f"{np.exp(log_mu):.1f}",
              delta=f"{np.exp(log_mu) - np.exp(base_mu):+.1f} vs normal weather")
    m2.metric("Expected outbreaks (normal weather)", f"{np.exp(base_mu):.1f}")
    st.caption("Estimates come from the simulated data. Treat them as a demo of how such a tool works.")

# ---------- Tab 5 ----------
with tab5:
    st.header("Limits and next steps")
    st.markdown("""
**What this simple project does NOT do (and a real study must):**
- **Real data:** use actual WAHIS records (2005 onward), Handistatus II (1996-2004), and old WOAH yearbooks (needs PDF/OCR extraction).
- **Real climate:** use ERA5, CHIRPS, CRU TS or WorldClim, matched to each country or outbreak location.
- **Reporting bias:** countries differ a lot in how much they report. Here one score handles it; real work needs more care.
- **Time and place patterns:** nearby countries and consecutive years are similar. Real work uses mixed models, spatio-temporal or Bayesian models (e.g. INLA).
- **Correlation is not proof:** a link in the data does not prove climate caused it. Other factors (trade, animal movement, land use) matter.
- **Literature review:** the evidence base should be mapped first (PRISMA-style) to choose which diseases and variables to study.

**Ideas to extend this repo:**
1. Swap in real climate data for a few countries.
2. Add a country-level random effect (mixed model).
3. Add a monthly version to study seasonality.
4. Use a Quarto report that re-runs on new data.
""")
