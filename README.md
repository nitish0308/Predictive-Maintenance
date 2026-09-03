# Product Defect / Predictive Maintenance Detection

Predicts whether a milling machine will experience a failure (defect) from live process
sensor readings — air/process temperature, rotational speed, torque, and tool wear — trained
on the AI4I 2020 Predictive Maintenance dataset.

## Contents

| File | Purpose |
|---|---|
| `predictive_maintenance.ipynb` | EDA, feature engineering, model training/comparison (Logistic Regression, Random Forest, XGBoost), evaluation, and export |
| `predictive_maintenance_model.joblib` | Saved best pipeline (preprocessing + XGBoost classifier), produced by the notebook |
| `data.csv` | AI4I 2020 dataset, 10,000 rows / 14 columns |
| `dat_details` | Original dataset description and citation |
| `app.py` | Flask app — numeric prediction form plus the LLM-routed Q&A endpoint |
| `models.py` | The underlying prediction models/computations (failure classifier + diagnosis, process-temp forecast, tool-wear estimate, rotational-speed feasibility check) |
| `llm_router.py` | Calls a local Ollama model with all four models as tools; the LLM decides which one answers a free-text question — no keyword routing in code |
| `templates/index.html` | Numeric prediction form |
| `templates/ask.html` | Free-text question form (LLM-routed Q&A) |
| `requirements.txt` | Python dependencies |

## How the model works

- **Target**: `Machine failure` (binary).
- **Dropped columns**: `UDI`, `Product ID` (identifiers), and `TWF`/`HDF`/`PWF`/`OSF`/`RNF`
  (these five flags are OR'd together to produce the label itself — including them would leak
  the answer).
- **Engineered features**, computed from raw sensor readings:
  - `temp_diff_K` = process temperature − air temperature
  - `power_W` = torque × rotational speed × 2π/60
  - `wear_torque` = tool wear × torque
- **Preprocessing**: `StandardScaler` on numeric features, `OneHotEncoder` on `Type` (L/M/H).
- **Best model**: XGBoost (`scale_pos_weight` set for class imbalance — failures are ~3.4% of
  the data), selected by PR-AUC. Test set: PR-AUC 0.886, ROC-AUC 0.983, F1 0.765.

See the notebook for the full comparison against Logistic Regression and Random Forest, plus
confusion matrix, feature importance, and 5-fold cross-validation.

## Setup

```bash
python -m pip install -r requirements.txt
```

The Q&A endpoint (`/ask`) calls a **local** [Ollama](https://ollama.com) model — no API key,
no cloud calls. Install Ollama, pull the model, and make sure the Ollama service is running
before starting the Flask app:

```bash
ollama pull qwen3:4b
ollama serve   # if not already running as a background service
```

`llm_router.py` uses `qwen3:4b` by default (verified locally to support Ollama's tool-calling
format). The numeric form (`/`) and `/api/predict` don't touch Ollama at all — only `/ask` does.
Expect a `/ask` response to take **10-30+ seconds** on CPU, especially the first call after the
model has to load into memory; this is a local 4B-parameter model, not a hosted API.

## Run the web app locally

```bash
python app.py
```

Then open **http://127.0.0.1:5000** in a browser. Fill in:

- `Type` (L / M / H)
- `Air temperature [K]`
- `Process temperature [K]`
- `Rotational speed [rpm]`
- `Torque [Nm]`
- `Tool wear [min]`

Submit to get a prediction — **OK / no failure** or **defect / failure likely** — along with
the failure probability. The three engineered features are computed automatically from your
inputs before scoring.

## API

`POST /api/predict` with a JSON body of the same fields, e.g.:

```bash
curl -X POST http://127.0.0.1:5000/api/predict \
  -H "Content-Type: application/json" \
  -d '{
    "Type": "L",
    "Air temperature [K]": 300.9,
    "Process temperature [K]": 310.7,
    "Rotational speed [rpm]": 1477,
    "Torque [Nm]": 43.2,
    "Tool wear [min]": 143
  }'
```

Response:

```json
{
  "predicted_failure": 0,
  "failure_probability": 0.0034,
  "label": "OK / NO FAILURE PREDICTED"
}
```

## Ask a question (LLM-routed Q&A)

Open **http://127.0.0.1:5000/ask** and type a free-text question, e.g.:

- *"Given air temp 305K, process temp 315K, 1400 rpm, 48 Nm torque and 100 min tool wear on a
  medium-quality part, should we flag this for maintenance?"*
- *"What process temperature results from air temp 300K, 1500 rpm, and 46 Nm torque?"*
- *"What rotational speed do I need to hit a process temperature of 315K, given air temp 299K
  and torque 44 Nm?"*
- *"Expected cumulative tool wear after 10,000 cycles at 1450 rpm and 42 Nm torque?"*

There are four underlying models in `models.py`: the failure classifier, a process-temperature
forecast, a tool-wear-over-cycles estimate, and a rotational-speed feasibility check (the last
two exist because the AI4I data shows tool wear accumulates per-cycle independent of sensor
readings, and process temperature is essentially `air_temp + ~10K` independent of rotational
speed/torque — so "what rpm hits this temperature" is usually unanswerable by design, and the
model says so rather than inventing a number). **Which model answers a given question is decided
by the local LLM at request time**, via tool use in `llm_router.py` — the Flask route and
`models.py` contain no if/elif keyword matching for this; the model reads each tool's
description and the question and picks one. The page shows both the final answer and a trace of
which tool was called with which arguments, so the routing decision is visible, not just the
result.

`POST /api/ask` accepts the same question as JSON (`{"question": "..."}`) and returns
`{"answer": ..., "trace": [...]}`.

## Retraining

Re-run `predictive_maintenance.ipynb` end to end; the last cell overwrites
`predictive_maintenance_model.joblib`, which `app.py` loads on startup. If you retrain with a
different scikit-learn/XGBoost version than what's installed when serving, you may see
`InconsistentVersionWarning` messages — harmless, but re-saving with matching versions clears
them.

## Notes & limitations

- Trained on a **synthetic** 10,000-row dataset; validate against real sensor logs before any
  production use.
- Threshold is the model's default 0.5 — tune it against the PR curve in the notebook if false
  alarms vs. missed failures have different real-world costs.
- `Machine failure` bundles five distinct failure modes (tool wear, heat dissipation, power,
  overstrain, random). A production system would likely benefit from separate per-mode
  classifiers rather than one blended label.
