"""
fault_schema.py
================

This module defines the UNIFIED, NORMALIZED fault schema used by ALL diagnostics
endpoints in the PdM backend. It receives the raw analytics from VibExtractor
(get_metrics) and transforms them into a stable, UI-friendly format:

Each fault object has:
    {
        "code": "unbalance",
        "family": "mechanical",
        "severity": "warning",         # ok | info | warning | alarm | critical
        "source": "harmonics_rule",    # which algorithm triggered it
        "direction": "D1",             # optional (for harmonic faults)
        "axis": "Vertical",            # optional (future use)
        "message": "Human readable",
        "metrics": {...}               # optional small diagnostics snippet
    }

The frontend renders these directly as fault cards, color-coded by severity.
The schema MUST remain stable even when analytics evolve.
"""

from __future__ import annotations
from typing import Dict, Any, List
import pandas as pd

# ---------------------------------------------------------------------
# FAULT FAMILY MAPPING
# ---------------------------------------------------------------------
# These groups are used by the UI (color coding + icons).
FAULT_FAMILIES: Dict[str, str] = {
    "unbalance": "mechanical",
    "looseness": "mechanical",
    "misalignment": "mechanical",
    "bearing_rms_high": "bearing",
    "overall_severity": "overall",

    # Type-specific faults
    "gear_issue": "gearbox",
    "motor_mech_issue": "motor",
    "process_or_mech_issue": "process",
    "fan_unbalance_or_looseness": "mechanical",
    "pulley_belt_issue": "belt_drive",

    # Advanced analytics
    "belt_slip": "belt_drive",
    "fan_blade_damage": "mechanical",
    "gear_mesh_modulation": "gearbox",
    "low_freq_looseness": "mechanical",

    # Envelope demod placeholders
    "bearing_bpfo": "bearing",
    "bearing_bpfi": "bearing",
    "bearing_bsf": "bearing",
    "bearing_ftf": "bearing",
}

# ---------------------------------------------------------------------
# SOURCE MAPPING (for explainability logs)
# ---------------------------------------------------------------------
FAULT_SOURCES: Dict[str, str] = {
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

    "belt_slip": "type_specific_rule",
    "fan_blade_damage": "type_specific_rule",
    "gear_mesh_modulation": "pattern_score",
    "low_freq_looseness": "pattern_score",

    # Envelope demod (future)
    "bearing_bpfo": "envelope_demod",
    "bearing_bpfi": "envelope_demod",
    "bearing_bsf": "envelope_demod",
    "bearing_ftf": "envelope_demod",
}

# ---------------------------------------------------------------------
# SEVERITY SCORING TABLE
# ---------------------------------------------------------------------
SEVERITY_SCORE = {
    "ok": 1.00,
    "info": 0.85,
    "warning": 0.45,
    "alarm": 0.10,
    "critical": 0.0,
}


def compute_severity_score(faults: List[Dict[str, Any]]) -> float:
    """
    Compute overall HEALTH SCORE 0..1 for an equipment.

    Rules:
    - Score is dominated by the WORST severity
    - "ok" = 1.0
    - "info" = 0.85
    - "warning" = 0.45
    - "alarm" = 0.1
    - "critical" = 0.0

    We can later enhance to weighted models (bearing faults > unbalance, etc.)
    """
    if not faults:
        return 1.0

    worst = min((SEVERITY_SCORE.get(f["severity"], 1.0) for f in faults))
    return float(worst)


# ---------------------------------------------------------------------
# ISO MAPPING HELPERS
# ---------------------------------------------------------------------
def _severity_from_iso(cond: str | None) -> str:
    """
    Convert ISO 10816/20816 condition into normalized severities.
    """
    if cond is None:
        return "ok"
    cond = str(cond).strip()
    return {
        "Good": "ok",
        "Satisfactory": "info",
        "Unsatisfactory": "warning",
        "Unacceptable": "alarm",
    }.get(cond, "ok")


def _severity_from_code(code: str, mapping: Dict[str, str]) -> str:
    """Translate harmonic codes (u0..u3, m0..m3, etc.) into normalized form."""
    return mapping.get(str(code), "ok")


# ---------------------------------------------------------------------
# MAIN NORMALIZER: build_fault_list()
# ---------------------------------------------------------------------
def build_fault_list(metrics: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Convert ALL raw outputs from get_metrics() into normalized fault objects.

    This function is the ONLY place where faults become UI-ready objects.
    """
    faults: List[Dict[str, Any]] = []

    # Extract basics
    rms_vel_max = float(metrics.get("rms_vel_max") or 0.0)
    max_direction = int(metrics.get("max_direction") or 1)
    x1amp = metrics.get("x1amp")
    mygroup = int(metrics.get("group", 1))

    # ISO severity frames
    mc_fault_rms_res = metrics.get("mc_fault_rms_res")
    mc_fault_peak_res = metrics.get("mc_fault_peak_res")
    fault_df = metrics.get("fault")  # summary DF
    type_specific_faults = metrics.get("type_specific_faults", []) or []

    # -----------------------------------------------------------------
    # 1) OVERALL ISO SEVERITY
    # -----------------------------------------------------------------
    iso_cond = None
    iso_color = "green"

    if isinstance(mc_fault_rms_res, pd.DataFrame) and not mc_fault_rms_res.empty:
        iso_cond = str(mc_fault_rms_res["cond"].iloc[0])
        iso_color = str(mc_fault_rms_res["color"].iloc[0])

    overall_sev = _severity_from_iso(iso_cond)

    faults.append(
        {
            "code": "overall_severity",
            "family": FAULT_FAMILIES["overall_severity"],
            "severity": overall_sev,
            "source": FAULT_SOURCES["overall_severity"],
            "direction": None,
            "axis": None,
            "message": f"Overall vibration severity is {iso_cond or 'Unknown'} (ISO group {mygroup}).",
            "metrics": {
                "rms_vel_max_mm_s": rms_vel_max,
                "iso_cond": iso_cond,
                "iso_color": iso_color,
                "group": mygroup,
            },
        }
    )

    # -----------------------------------------------------------------
    # 2) HARMONIC FAULTS (unbalance, looseness, misalignment, bearing RMS)
    # -----------------------------------------------------------------
    if isinstance(fault_df, pd.DataFrame) and not fault_df.empty:
        row = fault_df.iloc[0]

        # Unbalance
        uc = str(row.get("unbalance", "u0"))
        unb_map = {"u0": "ok", "u1": "info", "u2": "warning", "u3": "alarm"}
        unb_sev = _severity_from_code(uc, unb_map)
        if unb_sev != "ok":
            faults.append({
                "code": "unbalance",
                "family": FAULT_FAMILIES["unbalance"],
                "severity": unb_sev,
                "source": FAULT_SOURCES["unbalance"],
                "direction": f"D{max_direction}",
                "axis": None,
                "message": f"Unbalance pattern detected (code {uc}).",
                "metrics": {"code": uc, "x1amp_in_s": x1amp},
            })

        # Looseness
        lc = str(row.get("loosenes", "l0"))
        lo_map = {"l0": "ok", "l1": "info", "l2": "warning", "l3": "alarm"}
        lo_sev = _severity_from_code(lc, lo_map)
        if lo_sev != "ok":
            faults.append({
                "code": "looseness",
                "family": FAULT_FAMILIES["looseness"],
                "severity": lo_sev,
                "source": FAULT_SOURCES["looseness"],
                "direction": f"D{max_direction}",
                "axis": None,
                "message": f"Looseness pattern detected (code {lc}).",
                "metrics": {"code": lc},
            })

        # Misalignment
        mc = str(row.get("misalignment", "m0"))
        mi_map = {"m0": "ok", "m1": "info", "m2": "warning", "m3": "alarm"}
        mi_sev = _severity_from_code(mc, mi_map)
        if mi_sev != "ok":
            faults.append({
                "code": "misalignment",
                "family": FAULT_FAMILIES["misalignment"],
                "severity": mi_sev,
                "source": FAULT_SOURCES["misalignment"],
                "direction": f"D{max_direction}",
                "axis": None,
                "message": f"Misalignment pattern detected (code {mc}).",
                "metrics": {"code": mc},
            })

        # Bearing RMS
        bc = str(row.get("bearing_rms", "brms1"))
        br_map = {"brms1": "ok", "brms2": "warning", "brms3": "alarm"}
        br_sev = _severity_from_code(bc, br_map)
        if br_sev != "ok":
            faults.append({
                "code": "bearing_rms_high",
                "family": FAULT_FAMILIES["bearing_rms_high"],
                "severity": br_sev,
                "source": FAULT_SOURCES["bearing_rms_high"],
                "direction": f"D{max_direction}",
                "axis": None,
                "message": f"Bearing RMS elevated (code {bc}).",
                "metrics": {"code": bc, "rms_acc_max": metrics.get("rms_acc_max")},
            })

    # -----------------------------------------------------------------
    # 3) TYPE-SPECIFIC FAULTS (motor, pulley, fan, gearbox, pump, etc.)
    # -----------------------------------------------------------------
    for tf in type_specific_faults:
        code = tf.get("code")
        faults.append(
            {
                "code": code,
                "family": FAULT_FAMILIES.get(code, "mechanical"),
                "severity": tf.get("severity", "warning"),
                "source": FAULT_SOURCES.get(code, "type_specific_rule"),
                "direction": None,
                "axis": None,
                "message": tf.get("reason", ""),
                "metrics": {},
            }
        )

    return faults
