# 🐄 Climate & Animal Health: a small Streamlit project

**One-line summary:** a simple app that shows whether warmer or wetter-than-normal years go with more reported animal disease outbreaks, built to practise the kind of analysis described in the World Organisation for Animal Health (WOAH) internship on *climate change and animal health using WAHIS data*.

> ⚠️ **The data is simulated, not real WAHIS data.** Effects were built in on purpose so we can test that the model finds them. Nothing here is a real scientific finding.

## What is inside

```
climate-animal-health/
├── app.py                              # the Streamlit app (5 simple tabs)
├── requirements.txt                    # libraries to install
├── data/
│   ├── generate_data.py                # makes the sample data
│   └── animal_health_climate_sample.csv  # 24 countries x 20 years x 3 diseases
└── model/
    ├── train_model.py                  # trains the simple model
    └── model_coefficients.json         # the saved model (what the app reads)
```

## The data (one row = one country, one year, one disease)

| Column | Meaning |
|---|---|
| country, region, year, disease | Where, when, which disease |
| temp_anomaly_c | How much warmer than normal (°C) |
| rain_anomaly_10mm | How much wetter than normal (units of 10 mm/month) |
| livestock_index | Relative number of animals (1 = average) |
| reporting_score | How well the country reports (0 to 1) |
| outbreaks | Number of reported outbreaks (**what we try to explain**) |

Diseases: **Bluetongue** (insect-spread, built to react to heat), **Anthrax** (built to react to dry spells), **Foot-and-mouth disease** (built with *no* climate link, as a control).

## The model

A **Poisson regression** (the standard tool for count data). It includes **reporting score** and **livestock** so the model does not confuse "more reporting" or "more animals" with a climate effect.

Typical result: Bluetongue rises roughly 30% for each +1°C. Foot-and-mouth shows no clear link, as built.

## Run it

```bash
pip install -r requirements.txt
python data/generate_data.py     # optional: rebuilds the data
python model/train_model.py      # optional: retrains the model
streamlit run app.py
```

## Deploy free on Streamlit Community Cloud

1. Push this folder to a GitHub repository.
2. Go to https://share.streamlit.io and sign in with GitHub.
3. Click **New app**, choose your repo, set main file to `app.py`, then **Deploy**.

## Limits and next steps

Real work would use actual WAHIS data, ERA5/CHIRPS climate data, mixed or spatio-temporal models (e.g. INLA), and careful handling of reporting bias. See the "Limits & next steps" tab in the app.

## Link to the WOAH role

This mini-project practises the same pipeline: **data → climate covariates → statistical model → plain-language results → reproducible code on GitHub.**
