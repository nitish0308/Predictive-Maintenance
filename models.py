"""Deterministic prediction models for the predictive-maintenance system.

These are the *computations* the system can perform. Which one applies to a given
natural-language question is decided at runtime by an LLM (see llm_router.py) reading
each function's description and choosing a tool to call -- this module intentionally
contains no keyword/regex routing logic.
"""

import numpy as np
import pandas as pd
import joblib

_DF = pd.read_csv("data.csv")
_FAILURE_PIPELINE = joblib.load("predictive_maintenance_model.joblib")

# --- Empirical relationships derived from data.csv (see predictive_maintenance.ipynb) ---

_TEMP_OFFSET = _DF["Process temperature [K]"] - _DF["Air temperature [K]"]
TEMP_OFFSET_MEAN = float(_TEMP_OFFSET.mean())
TEMP_OFFSET_STD = float(_TEMP_OFFSET.std())
TEMP_OFFSET_RPM_CORR = float(_TEMP_OFFSET.corr(_DF["Rotational speed [rpm]"]))
TEMP_OFFSET_TORQUE_CORR = float(_TEMP_OFFSET.corr(_DF["Torque [Nm]"]))

_sorted = _DF.sort_values("UDI").reset_index(drop=True)
_wear_diff = _sorted["Tool wear [min]"].diff()
_inc = _sorted.loc[_wear_diff > 0].copy()
_inc["wear_diff"] = _wear_diff[_wear_diff > 0]
WEAR_RATE_BY_TYPE = _inc.groupby("Type")["wear_diff"].mean().to_dict()
WEAR_RATE_OVERALL = float(_inc["wear_diff"].mean())

TWF_WINDOW = (200, 240)


# --- Model 1: overall machine-failure classifier + specific failure-mode diagnosis ---

def diagnose_failure_modes(air_temp, process_temp, rpm, torque, tool_wear, product_type=None):
    temp_diff = process_temp - air_temp
    power = torque * rpm * (2 * np.pi / 60)
    wear_torque = tool_wear * torque

    modes = {}

    twf_window_start = TWF_WINDOW[0]
    if tool_wear < twf_window_start:
        modes["TWF"] = {
            "triggered": False,
            "detail": f"tool wear {tool_wear:.0f} min is {twf_window_start - tool_wear:.0f} min below the 200-240 min risk window",
            "relative_margin": (twf_window_start - tool_wear) / twf_window_start,
        }
    else:
        modes["TWF"] = {
            "triggered": None,
            "detail": "tool wear is inside the 200-240 min risk window (a coin flip in the source data, not deterministic)",
            "relative_margin": 0.0,
        }

    temp_margin = temp_diff - 8.6
    speed_margin = rpm - 1380
    temp_rel = temp_margin / 8.6
    speed_rel = speed_margin / 1380
    hdf_triggered = temp_margin < 0 and speed_margin < 0
    if temp_rel >= speed_rel:
        bottleneck_name, bottleneck_rel = "temp_diff", temp_rel
    else:
        bottleneck_name, bottleneck_rel = "rotational speed", speed_rel
    modes["HDF"] = {
        "triggered": hdf_triggered,
        "detail": (f"temp_diff={temp_diff:.1f} K (needs <8.6, {temp_rel:.1%} margin), "
                   f"rotational speed={rpm:.0f} rpm (needs <1380, {speed_rel:.1%} margin); "
                   f"both required, {bottleneck_name} is furthest from tripping"),
        "relative_margin": abs(bottleneck_rel),
    }

    lower_margin = power - 3500
    upper_margin = 9000 - power
    pwf_triggered = power < 3500 or power > 9000
    if lower_margin <= upper_margin:
        nearest_margin, nearest_bound = lower_margin, 3500
    else:
        nearest_margin, nearest_bound = upper_margin, 9000
    modes["PWF"] = {
        "triggered": pwf_triggered,
        "detail": f"power={power:.0f} W, safe band is 3500-9000 W; nearest bound is {nearest_bound} W",
        "relative_margin": abs(nearest_margin) / nearest_bound,
    }

    if product_type is not None:
        osf_threshold = {"L": 11000, "M": 12000, "H": 13000}[product_type]
        modes["OSF"] = {
            "triggered": wear_torque > osf_threshold,
            "detail": f"tool_wear*torque={wear_torque:.0f} min*Nm vs {osf_threshold} min*Nm threshold for type {product_type}",
            "relative_margin": abs(osf_threshold - wear_torque) / osf_threshold,
        }

    comparable = {k: v for k, v in modes.items() if k in ("TWF", "HDF", "PWF")}
    closest_mode = min(comparable, key=lambda k: comparable[k]["relative_margin"])

    return {"temp_diff_K": temp_diff, "power_W": power, "wear_torque": wear_torque,
            "modes": modes, "closest_mode": closest_mode}


def predict_machine_failure(air_temp, process_temp, rpm, torque, tool_wear, product_type="M"):
    """Predicts overall Machine failure (binary) + probability, and diagnoses which
    specific failure mode (TWF/HDF/PWF/OSF) the reading sits closest to."""
    row = pd.DataFrame([{
        "Type": product_type,
        "Air temperature [K]": air_temp,
        "Process temperature [K]": process_temp,
        "Rotational speed [rpm]": rpm,
        "Torque [Nm]": torque,
        "Tool wear [min]": tool_wear,
        "temp_diff_K": process_temp - air_temp,
        "power_W": torque * rpm * (2 * np.pi / 60),
        "wear_torque": tool_wear * torque,
    }])
    proba = float(_FAILURE_PIPELINE.predict_proba(row)[:, 1][0])
    pred = int(_FAILURE_PIPELINE.predict(row)[0])
    diagnosis = diagnose_failure_modes(air_temp, process_temp, rpm, torque, tool_wear, product_type)

    return {
        "predicted_failure": pred,
        "failure_probability": round(proba, 4),
        "closest_specific_failure_mode": diagnosis["closest_mode"],
        "failure_modes": {
            name: {"triggered": info["triggered"], "detail": info["detail"],
                   "relative_margin": round(info["relative_margin"], 4)}
            for name, info in diagnosis["modes"].items()
        },
    }


# --- Model 2: process-temperature forecast ---

def predict_process_temperature(air_temp, rpm=None, torque=None):
    """Forecasts process temperature from air temperature. Empirically (see data.csv),
    process_temp = air_temp + ~10K (SD ~1K) with ~0 correlation to rpm/torque, so those
    two inputs are accepted but do not move the prediction -- this is reported explicitly
    rather than silently ignored."""
    predicted = air_temp + TEMP_OFFSET_MEAN
    return {
        "predicted_process_temp_K": round(predicted, 2),
        "expected_range_K": [round(predicted - 2 * TEMP_OFFSET_STD, 2),
                              round(predicted + 2 * TEMP_OFFSET_STD, 2)],
        "offset_mean_K": round(TEMP_OFFSET_MEAN, 3),
        "offset_std_K": round(TEMP_OFFSET_STD, 3),
        "rpm_correlation_with_offset": round(TEMP_OFFSET_RPM_CORR, 4),
        "torque_correlation_with_offset": round(TEMP_OFFSET_TORQUE_CORR, 4),
        "note": ("process_temp is essentially air_temp + a near-constant offset in this dataset; "
                 "rotational speed and torque have negligible correlation with that offset "
                 f"({TEMP_OFFSET_RPM_CORR:.3f} and {TEMP_OFFSET_TORQUE_CORR:.3f}) and were not used "
                 "in the prediction even if supplied."),
    }


# --- Model 3: cumulative tool wear over N cycles ---

def estimate_tool_wear(n_cycles, product_type=None):
    """Estimates cumulative tool wear after N production cycles from the empirical
    per-cycle wear-rate distribution, and reports the TWF (tool wear failure) 200-240 min
    replacement window the tool will hit long before very large cycle counts."""
    rate = WEAR_RATE_BY_TYPE.get(product_type, WEAR_RATE_OVERALL) if product_type else WEAR_RATE_OVERALL
    naive_total = rate * n_cycles
    cycles_to_window_start = TWF_WINDOW[0] / rate
    cycles_to_window_end = TWF_WINDOW[1] / rate

    return {
        "per_cycle_wear_rate_min": round(rate, 3),
        "naive_unbounded_wear_after_n_cycles_min": round(naive_total, 1),
        "cycles_to_reach_twf_window": [round(cycles_to_window_start, 1), round(cycles_to_window_end, 1)],
        "twf_window_min": list(TWF_WINDOW),
        "note": (f"At ~{rate:.2f} min/cycle, the tool reaches the {TWF_WINDOW[0]}-{TWF_WINDOW[1]} min TWF "
                 f"replacement/failure window after roughly {cycles_to_window_start:.0f}-{cycles_to_window_end:.0f} "
                 f"cycles -- far fewer than {n_cycles}. In practice the tool is replaced (or fails) there and wear "
                 f"resets, so cumulative wear does not grow unboundedly to {naive_total:.0f} min; it cycles between "
                 f"0 and ~{TWF_WINDOW[1]} min in steady state. Note: this dataset's own documentation claims a "
                 "clean 2/3/5 min-per-cycle split by product Type (L/M/H), but the empirical per-cycle increments "
                 f"here do not show that split cleanly ({WEAR_RATE_BY_TYPE}); the rate used is measured directly "
                 "from consecutive-row wear increases, not taken from the documentation."),
    }


# --- Model 4: rotational speed needed for a target process temperature ---

def estimate_rotational_speed_for_target_temp(target_process_temp, air_temp, torque=None):
    """Attempts to solve for the rotational speed needed to hit a target process
    temperature. Since process temperature is empirically independent of rotational
    speed in this dataset, this reports infeasibility rather than fabricating an rpm."""
    expected = air_temp + TEMP_OFFSET_MEAN
    diff = target_process_temp - expected
    sds_away = diff / TEMP_OFFSET_STD if TEMP_OFFSET_STD else float("inf")
    feasible = abs(sds_away) <= 2

    return {
        "expected_process_temp_at_this_air_temp_K": round(expected, 2),
        "target_process_temp_K": target_process_temp,
        "standard_deviations_away": round(sds_away, 2),
        "feasible_by_adjusting_rotational_speed": False,
        "within_normal_noise_range": feasible,
        "note": (f"Rotational speed has ~0 empirical correlation ({TEMP_OFFSET_RPM_CORR:.3f}) with process "
                 f"temperature in this dataset -- process_temp is essentially air_temp + {TEMP_OFFSET_MEAN:.1f}K "
                 f"+/- {TEMP_OFFSET_STD:.1f}K, independent of rpm/torque. No rotational-speed setting can reliably "
                 f"move process temperature to {target_process_temp}K here; at air_temp={air_temp}K the expected "
                 f"process temperature is ~{expected:.1f}K regardless of speed, which is {sds_away:.1f} standard "
                 f"deviations from the target. To reach {target_process_temp}K, air temperature would need to "
                 f"change instead (to about {target_process_temp - TEMP_OFFSET_MEAN:.1f}K)."),
    }
