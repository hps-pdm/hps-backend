# app/api_faults.py

from __future__ import annotations
from typing import Dict, Any, List

import numpy as np
import pandas as pd

FAULT_FAMILIES = {
    "unbalance": "mechanical",
    "looseness": "mechanical",
    "misalignment": "mechanical",
    "bearing_rms_high": "bearing",
    "overall_severity": "overall",
    "gear_issue": "gearbox",
    "motor_mech_issue": "motor",
    "process_or_mech_issue": "process",
    "fan_unbalance_or_looseness": "mechanical",
    "pulley_belt_issue": "belt_drive",
}

FAULT_SOURCES = {
    "unbalance": "harmonics_rule",
    "looseness": "harmonics_rule",
    "misalignment": "harmonics_rule",
    "bearing_rms_high": "bearing_rms_rule",
    "overall_severity": "iso_rms",
    "gear_issue": "type_specific_rule",
    "motor_mech_issue": "type_specific_rule",
    "process_or_mech_issue": "type_specific_rule",
    "fan_unbalance_or_looseness": "type_specific_rule",
    "pulley_belt_issue": "type_specific_rule",
}


def _severity_from_iso(cond: str | None) -> str:
    """
    ISO condition: Good / Satisfactory / Unsatisfactory / Unacceptable
    → severity: ok / info / warning / alarm / critical
    """
    if cond is None:
        return "ok"
    cond = str(cond)
    if cond == "Good":
        return "ok"
    if cond == "Satisfactory":
        return "info"
    if cond == "Unsatisfactory":
        return "warning"
    if cond == "Unacceptable":
        return "alarm"
    return "ok"


def _severity_from_code(code: str, levels: Dict[str, str]) -> str:
    """
    Map coded levels (u0/u1/u2/u3, l0.., m0.., brms1-3) to severity strings.
    Example levels mapping: {"u0": "ok", "u1": "info", "u2": "warning", "u3": "alarm"}
    """
    return levels.get(code, "ok")


def build_fault_list(metrics: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Turn VibExtractor.get_metrics(...) result into a list of fault dicts
    with (code, family, severity, source, message, metrics).
    """

    faults: list[dict] = []

    # --- Unpack basic metrics
    rms_vel_max = float(metrics.get("rms_vel_max") or 0.0)
    max_direction = int(metrics.get("max_direction") or 1)
    x1amp = metrics.get("x1amp")
    mygroup = int(metrics.get("group", 1))

    # fault + faults DataFrames
    fault_df = metrics.get("fault")       # summary DataFrame
    faults_df = metrics.get("faults")     # per-direction DataFrame
    mc_fault_rms_res = metrics.get("mc_fault_rms_res")
    mc_fault_peak_res = metrics.get("mc_fault_peak_res")

    # type_specific_faults (from classify_type_specific_faults if you added it)
    type_specific_faults = metrics.get("type_specific_faults", []) or []

    # ----- 1) Overall ISO severity fault -----
    # use mc_fault_rms_res (preferred) or mc_fault_peak_res
    iso_cond = None
    if isinstance(mc_fault_rms_res, pd.DataFrame) and not mc_fault_rms_res.empty:
        iso_cond = str(mc_fault_rms_res["cond"].iloc[0])
        iso_color = str(mc_fault_rms_res["color"].iloc[0])
    elif isinstance(mc_fault_peak_res, pd.DataFrame) and not mc_fault_peak_res.empty:
        iso_cond = str(mc_fault_peak_res["cond"].iloc[0])
        iso_color = str(mc_fault_peak_res["color"].iloc[0])
    else:
        iso_color = "green"

    overall_sev = _severity_from_iso(iso_cond)
    faults.append({
        "code": "overall_severity",
        "family": FAULT_FAMILIES["overall_severity"],
        "severity": overall_sev,
        "source": FAULT_SOURCES["overall_severity"],
        "direction": None,
        "axis": None,
        "message": f"Overall vibration severity is {iso_cond or 'Unknown'} (group {mygroup}).",
        "metrics": {
            "rms_vel_max_in_s": rms_vel_max,
            "iso_cond": iso_cond,
            "iso_color": iso_color,
            "group": mygroup,
        },
    })

    # ----- 2) Harmonic faults: unbalance, looseness, misalignment, bearing RMS -----
    if isinstance(fault_df, pd.DataFrame) and not fault_df.empty:
        row = fault_df.iloc[0]

        # Unbalance
        ucode = str(row.get("unbalance", "u0"))
        u_levels = {"u0": "ok", "u1": "info", "u2": "warning", "u3": "alarm"}
        u_sev = _severity_from_code(ucode, u_levels)
        if u_sev != "ok":
            faults.append({
                "code": "unbalance",
                "family": FAULT_FAMILIES["unbalance"],
                "severity": u_sev,
                "source": FAULT_SOURCES["unbalance"],
                "direction": f"D{max_direction}",
                "axis": None,
                "message": f"Unbalance pattern detected (code {ucode}).",
                "metrics": {
                    "code": ucode,
                    "rms_vel_max_in_s": rms_vel_max,
                    "x1amp": x1amp,
                },
            })

        # Looseness
        lcode = str(row.get("loosenes", "l0"))
        l_levels = {"l0": "ok", "l1": "info", "l2": "warning", "l3": "alarm"}
        l_sev = _severity_from_code(lcode, l_levels)
        if l_sev != "ok":
            faults.append({
                "code": "looseness",
                "family": FAULT_FAMILIES["looseness"],
                "severity": l_sev,
                "source": FAULT_SOURCES["looseness"],
                "direction": f"D{max_direction}",
                "axis": None,
                "message": f"Looseness pattern detected (code {lcode}).",
                "metrics": {
                    "code": lcode,
                    "rms_vel_max_in_s": rms_vel_max,
                },
            })

        # Misalignment
        mcode = str(row.get("misalignment", "m0"))
        m_levels = {"m0": "ok", "m1": "info", "m2": "warning", "m3": "alarm"}
        m_sev = _severity_from_code(mcode, m_levels)
        if m_sev != "ok":
            faults.append({
                "code": "misalignment",
                "family": FAULT_FAMILIES["misalignment"],
                "severity": m_sev,
                "source": FAULT_SOURCES["misalignment"],
                "direction": f"D{max_direction}",
                "axis": None,
                "message": f"Misalignment pattern detected (code {mcode}).",
                "metrics": {
                    "code": mcode,
                    "rms_vel_max_in_s": rms_vel_max,
                },
            })

        # Bearing RMS
        bcode = str(row.get("bearing_rms", "brms1"))
        b_levels = {"brms1": "ok", "brms2": "warning", "brms3": "alarm"}
        b_sev = _severity_from_code(bcode, b_levels)
        if b_sev != "ok":
            faults.append({
                "code": "bearing_rms_high",
                "family": FAULT_FAMILIES["bearing_rms_high"],
                "severity": b_sev,
                "source": FAULT_SOURCES["bearing_rms_high"],
                "direction": f"D{max_direction}",
                "axis": None,
                "message": f"Bearing RMS elevated (code {bcode}).",
                "metrics": {
                    "code": bcode,
                    "rms_acc_max": metrics.get("rms_acc_max"),
                },
            })

    # ----- 3) Type-specific faults (gearbox, motor, pump, fan, pulley, etc.) -----
    for tf in type_specific_faults:
        code = tf.get("code")
        family = FAULT_FAMILIES.get(code, "mechanical")
        source = FAULT_SOURCES.get(code, "type_specific_rule")
        severity = tf.get("severity", "warning")
        reason = tf.get("reason", "")

        faults.append({
            "code": code,
            "family": family,
            "severity": severity,
            "source": source,
            "direction": None,
            "axis": None,
            "message": reason,
            "metrics": {},
        })

    return faults
