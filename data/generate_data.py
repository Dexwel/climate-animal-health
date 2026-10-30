"""
Builds a SMALL, SIMULATED dataset that mimics the shape of the real problem:
  country x year x disease  ->  number of reported outbreaks + climate columns.

IMPORTANT: this is NOT real WAHIS data. Real WAHIS data comes from WOAH.
The climate effects below are put in on purpose, so we can check that
the model can find them (a common way to test an analysis pipeline).
"""
import numpy as np
import pandas as pd

rng = np.random.default_rng(42)

# (name, region, base temperature in C, base rain in mm/month, livestock index)
countries = [
    ("Nigeria", "Africa", 27, 110, 1.0), ("Kenya", "Africa", 22, 70, 0.9),
    ("Ethiopia", "Africa", 20, 60, 1.2), ("South Africa", "Africa", 18, 45, 0.8),
    ("France", "Europe", 12, 65, 1.1), ("Spain", "Europe", 15, 40, 0.9),
    ("Poland", "Europe", 8, 50, 0.8), ("Italy", "Europe", 14, 80, 0.8),
    ("India", "Asia", 25, 100, 1.5), ("Vietnam", "Asia", 25, 130, 0.7),
    ("China", "Asia", 14, 55, 1.6), ("Thailand", "Asia", 28, 125, 0.6),
    ("Brazil", "Americas", 24, 120, 1.7), ("Argentina", "Americas", 16, 70, 1.2),
    ("Mexico", "Americas", 21, 55, 1.0), ("Colombia", "Americas", 24, 200, 0.8),
    ("Australia", "Oceania", 21, 40, 1.0), ("New Zealand", "Oceania", 11, 100, 0.7),
    ("Turkiye", "Middle East", 12, 40, 0.9), ("Iran", "Middle East", 17, 20, 0.8),
    ("Saudi Arabia", "Middle East", 26, 8, 0.5), ("Egypt", "Middle East", 22, 3, 0.7),
    ("Mongolia", "Asia", 1, 20, 0.9), ("Peru", "Americas", 19, 90, 0.6),
]

years = list(range(2005, 2025))
diseases = ["Bluetongue", "Anthrax", "Foot-and-mouth disease"]

# True effects used to simulate: per +1 C temperature anomaly, per +10 mm rain anomaly
TRUE = {
    "Bluetongue":             dict(intercept=-1.2, temp=0.30, rain=0.05),   # vector-borne: likes warmth
    "Anthrax":                dict(intercept=-1.5, temp=0.10, rain=-0.12),  # dry spells raise risk
    "Foot-and-mouth disease": dict(intercept=-1.0, temp=0.00, rain=0.00),   # control: no climate link
}

rows = []
for name, region, t0, r0, live in countries:
    for y in years:
        temp_anom = 0.04 * (y - 2005) + rng.normal(0, 0.5)   # warming trend + noise
        rain_anom = rng.normal(0, 1.2)                        # in units of 10 mm/month
        # reporting improves over time: a big reason counts rise in real data
        reporting = float(np.clip(0.35 + 0.025 * (y - 2005) + rng.normal(0, 0.04), 0.1, 1.0))
        for d in diseases:
            p = TRUE[d]
            log_mu = (p["intercept"] + p["temp"] * temp_anom + p["rain"] * rain_anom
                      + np.log(live) + np.log(reporting) + 2.6)
            rows.append(dict(
                country=name, region=region, year=y, disease=d,
                mean_temp_c=round(t0 + temp_anom, 2),
                temp_anomaly_c=round(temp_anom, 2),
                rain_mm_month=round(max(r0 + rain_anom * 10, 0), 1),
                rain_anomaly_10mm=round(rain_anom, 2),
                livestock_index=live,
                reporting_score=round(reporting, 2),
                outbreaks=int(rng.poisson(np.exp(log_mu)))))

df = pd.DataFrame(rows)
df.to_csv("data/animal_health_climate_sample.csv", index=False)
print("Saved", df.shape, "rows/cols")
