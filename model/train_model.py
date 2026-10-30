"""
Simple model: Poisson regression (a standard way to model COUNTS like outbreaks).

For each disease we ask:
   outbreaks ~ temperature anomaly + rain anomaly + reporting score + livestock
Reporting score and livestock are included so the model does not blame climate
for things that are really "more reporting" or "more animals".

Output: model/model_coefficients.json (small, readable, easy to load in Streamlit)
"""
import json
import numpy as np
import pandas as pd
import statsmodels.api as sm

df = pd.read_csv("data/animal_health_climate_sample.csv")
out = {}

for disease, d in df.groupby("disease"):
    X = pd.DataFrame({
        "temp_anomaly_c": d["temp_anomaly_c"],
        "rain_anomaly_10mm": d["rain_anomaly_10mm"],
        "log_reporting": np.log(d["reporting_score"]),
        "log_livestock": np.log(d["livestock_index"]),
    })
    X = sm.add_constant(X)
    fit = sm.GLM(d["outbreaks"], X, family=sm.families.Poisson()).fit()
    ci = fit.conf_int()
    out[disease] = {
        name: {"coef": float(fit.params[name]), "p_value": float(fit.pvalues[name]),
               "ci_low": float(ci.loc[name, 0]), "ci_high": float(ci.loc[name, 1])}
        for name in fit.params.index}
    t = out[disease]["temp_anomaly_c"]
    print(f"{disease:25s} +1C -> {100*(np.exp(t['coef'])-1):+.1f}% outbreaks (p={t['p_value']:.3f})")

with open("model/model_coefficients.json", "w") as f:
    json.dump(out, f, indent=2)
print("Saved model/model_coefficients.json")
