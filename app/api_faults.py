"""
api_faults.py
============

This module provides normalized diagnostics for single equipment and
fleet-level summaries on top of EnhancedVibExtractor.get_metrics().

Key outputs:

- build_diagnostics_for_sn(sn) → diagnostics dict:
    {
      "serialNumber": 189314686,
      "equip_type": "Pulley",
      "group": 2,
      "rpm_hz": 33.58,
      "rpm_rpm": 2015.0,
      "rpm_confidence": "strong",
      "overall": {... ISO 10816/20816 block ...},
      "severity_score": 0.45,              # 0..1, higher = healthier
      "equipment_severity": "warning",     # "ok" | "warning" | "alarm"
      "misalignment": "ok",
      "unbalance": "ok",
      "looseness": "ok",
      "bearing": "warning",
      "faults": [... detailed fault list ...],
    }

- get_fleet_diagnostics() → list[diagnostics dict]

- build_fleet_summary() → pies for SummaryPage:
    {
      "total_equipment": 4,
      "severity":   { "labels": ["OK","Warning","Alarm"], "values": [...] },
      "fault_type": { "labels": [...], "values": [...] },
      "iso_zone":   { "labels": [...], "values": [...] },
      "equip_type": { "labels": [...], "values": [...] },
    }
"""

from __future__ import annotations

from collections import Counter
from typing import Any, Dict, List, Optional
import json
import os

import pandas as pd

from app.data.EnhancedVibExtractor import (
    get_equipment_id,
    get_latest_vibration_data,
    get_metrics,
    info_dat,
)

from .fault_schema import (
    build_fault_list,
    compute_severity_score,
)

# Config loader
CONFIG_PATH = os.path.join(os.path.dirname(__file__), "config", "config.json")


def _load_app_config() -> dict:
    if not os.path.exists(CONFIG_PATH):
        return {}
    try:
        with open(CONFIG_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


_APP_CFG = _load_app_config()




def rms_group_to_status(group: Optional[int]) -> str:
    """
    Map ISO group (1..4) to unified status: ok / warning / alarm.

    Typical convention:
      - Group 1–2 -> ok
      - Group 3   -> warning
      - Group 4   -> alarm
    """
    if group is None:
        return "ok"

    try:
        g = int(group)
    except Exception:
        return "ok"

    if g <= 2:
        return "ok"
    if g == 3:
        return "warning"
    return "alarm"


# ---------------------------------------------------------------------
# Helpers: RPM and ISO overall
# ---------------------------------------------------------------------


def _derive_rpm_fields(metrics: Dict[str, Any]) -> Dict[str, Any]:
    """
    Derive UI-friendly RPM fields from VibExtractor metrics.

    - rpm_1x_hz: best estimate of running speed in Hz
    - rpm_1x_rpm: same in RPM
    - rpm_confidence: "strong" / "medium" / "weak" / "unknown"
    """
    cal_hz = metrics.get("cal_rpm_hz")
    guess_hz = metrics.get("guess_rpm_hz")

    rpm_1x_hz = (
        float(cal_hz)
        if cal_hz is not None
        else (float(guess_hz) if guess_hz is not None else None)
    )
    rpm_1x_rpm = float(rpm_1x_hz * 60.0) if rpm_1x_hz is not None else None

    # Confidence heuristic:
    if cal_hz is not None and guess_hz is not None:
        diff = abs(float(cal_hz) - float(guess_hz))
        base = max(abs(float(cal_hz)), 1e-6)
        rel = diff / base
        if rel < 0.03:
            rpm_conf = "strong"
        elif rel < 0.10:
            rpm_conf = "medium"
        else:
            rpm_conf = "weak"
    elif rpm_1x_hz is not None:
        rpm_conf = "medium"
    else:
        rpm_conf = "unknown"

    return {
        "rpm_1x_hz": rpm_1x_hz,
        "rpm_1x_rpm": rpm_1x_rpm,
        "rpm_confidence": rpm_conf,
    }


def _build_overall(metrics: Dict[str, Any]) -> Dict[str, Any]:
    """
    Build the 'overall' section (ISO severity, RMS) from metrics.

    Metrics expected:
      - rms_vel_max (mm/s)
      - mc_fault_rms_res DataFrame (with cond/color/group)
    """
    overall: Dict[str, Any] = {}
    mc_fault_rms_res = metrics.get("mc_fault_rms_res")
    rms_vel_max = float(metrics.get("rms_vel_max") or 0.0)
    group = metrics.get("group")

    if isinstance(mc_fault_rms_res, pd.DataFrame) and not mc_fault_rms_res.empty:
        iso_zone = str(mc_fault_rms_res["cond"].iloc[0])
        color = str(mc_fault_rms_res["color"].iloc[0])
        group_val = int(group or mc_fault_rms_res["group"].iloc[0])

        overall = {
            "velocity_rms_mm_s": rms_vel_max,
            "iso_zone": iso_zone,
            "color": color,
            "group": group_val,
        }
    else:
        if group is not None:
            overall = {
                "velocity_rms_mm_s": rms_vel_max,
                "iso_zone": None,
                "color": None,
                "group": int(group),
            }
        else:
            overall = {
                "velocity_rms_mm_s": rms_vel_max,
                "iso_zone": None,
                "color": None,
                "group": None,
            }

    return overall


# ---------------------------------------------------------------------
# Unified severity helpers
# ---------------------------------------------------------------------


Status = str  # "ok" | "warning" | "alarm"


def _normalize_status(val: Any, default: Status = "ok") -> Status:
    if not isinstance(val, str):
        return default
    v = val.lower().strip()
    if v in ("ok", "warning", "alarm"):
        return v
    return default


def _derive_equipment_severity_from_faults(
    misalignment: Status,
    unbalance: Status,
    looseness: Status,
    bearing: Status,
) -> Status:
    """
    Equipment severity = worst of fault-level severities.
    """
    vals = [misalignment, unbalance, looseness, bearing]
    if "alarm" in vals:
        return "alarm"
    if "warning" in vals:
        return "warning"
    return "ok"


def _severity_score_to_equipment_status(score: Optional[float]) -> Status:
    """
    Map severity_score (0..1, 1=best) to "ok"/"warning"/"alarm".
    """
    if score is None:
        return "ok"

    try:
        s = float(score)
    except Exception:
        return "ok"

    if s >= 0.75:
        return "ok"
    if s >= 0.5:
        return "warning"
    return "alarm"


# ---------------------------------------------------------------------
# Single-equipment diagnostics
# ---------------------------------------------------------------------


def build_diagnostics_for_sn(sn: int) -> Dict[str, Any]:
    """
    High-level helper used by /api/diagnostics/<sn> and /api/diagnostics?sn=.

    Steps:
      - Get latest vibration data (cache-first) for the given serialNumber.
      - Run get_metrics(df, sn).
      - Normalize detailed faults with build_fault_list().
      - Compute severity_score (0..1) from faults.
      - Derive:
          * rpm_hz / rpm_rpm / rpm_confidence
          * ISO "overall" block
          * per-fault severities (misalignment/unbalance/looseness/bearing)
          * unified equipment_severity ("ok"/"warning"/"alarm")
      - Return a JSON-friendly diagnostics dict.
    """
    try:
        try:
            df = get_latest_vibration_data(serialNumber=sn, use_cache=True)
        except TypeError:
            df = get_latest_vibration_data(serialNumber=sn)

        if df is None or df.empty:
            return {
                "serialNumber": int(sn),
                "error": "No vibration data available for this serialNumber.",
            }

        df_latest = df.iloc[[0]]

        metrics = get_metrics(df_latest, sn)
        if isinstance(metrics, dict) and "error" in metrics:
            return {"serialNumber": int(sn), "error": metrics["error"]}

        # Detailed fault list from fault_schema
        faults = build_fault_list(metrics)
        severity_score = compute_severity_score(faults)

        # RPM + overall ISO block
        rpm_fields = _derive_rpm_fields(metrics)
        overall = _build_overall(metrics)

        equip_type = metrics.get("equip_type")
        group = metrics.get("group")
        
        velocity_rms_status = rms_group_to_status(overall.get("group"))


        # Extract per-fault severities from fault list, if present
        misalignment = "ok"
        unbalance = "ok"
        looseness = "ok"
        bearing = "ok"

        for f in faults:
            code = str(f.get("code") or f.get("id") or "").lower()
            status = _normalize_status(
                f.get("status")
                or f.get("level")
                or f.get("state")
                or f.get("severity"),
                "ok",
            )

            if code == "misalignment":
                misalignment = status
            elif code == "unbalance":
                unbalance = status
            elif code == "looseness":
                looseness = status
            elif code == "bearing":
                bearing = status


        # Unified equipment severity: worst of faults, with severity_score fallback
        equipment_severity = _derive_equipment_severity_from_faults(
            misalignment, unbalance, looseness, bearing
        )

        # If everything looks "ok" but severity_score is low, fall back to score-based mapping
        if equipment_severity == "ok" and severity_score is not None:
            equipment_severity = _severity_score_to_equipment_status(severity_score)

        asset_name = metrics.get("asset_name")
        if not asset_name and "serialNumber" in info_dat.columns:
            try:
                row_info = info_dat[info_dat["serialNumber"] == sn]
                if not row_info.empty and "AssetName" in row_info.columns:
                    val = row_info["AssetName"].values[0]
                    if isinstance(val, str) and val.strip():
                        asset_name = val.strip()
            except Exception:
                asset_name = None

        # Optional per-asset thresholds (Vel/Acc) from info_dat
        asset_thresholds = {}
        try:
            row = info_dat[info_dat["serialNumber"] == sn]
            if not row.empty:
                r = row.iloc[0]
                vel_ok = r.get("VelRmsOk_mm_s")
                vel_warn = r.get("VelRmsWarning_mm_s")
                vel_alarm = r.get("VelRmsAlarm_mm_s")
                if pd.notna(vel_ok) and pd.notna(vel_warn) and pd.notna(vel_alarm):
                    asset_thresholds["vel_rms_mm_s"] = {
                        "ok": float(vel_ok),
                        "warning": float(vel_warn),
                        "alarm": float(vel_alarm),
                    }
                acc_ok = r.get("AccRmsOk_g")
                acc_warn = r.get("AccRmsWarning_g")
                acc_alarm = r.get("AccRmsAlarm_g")
                if pd.notna(acc_ok) and pd.notna(acc_warn) and pd.notna(acc_alarm):
                    asset_thresholds["acc_rms_g"] = {
                        "ok": float(acc_ok),
                        "warning": float(acc_warn),
                        "alarm": float(acc_alarm),
                    }
        except Exception:
            asset_thresholds = {}

        payload: Dict[str, Any] = {
            "serialNumber": int(sn),
            "asset_name": asset_name,
            "name": asset_name,  # legacy-friendly alias for UI
            "equip_type": equip_type,
            "group": int(group) if group is not None else None,
            "rpm_hz": rpm_fields["rpm_1x_hz"],
            "rpm_rpm": rpm_fields["rpm_1x_rpm"],
            "rpm_confidence": rpm_fields["rpm_confidence"],
            "overall": overall,
            "severity_score": float(severity_score) if severity_score is not None else None,
            "equipment_severity": equipment_severity,
            "misalignment": misalignment,
            "unbalance": unbalance,
            "looseness": looseness,
            "bearing": bearing,
            "velocity_rms_status": velocity_rms_status,  # NEW
            "faults": faults,
        }

        # Include threshold config/matrix for frontend gauges (best-effort)
        thresholds_cfg = _APP_CFG.get("thresholds")
        threshold_matrix = _APP_CFG.get("threshold_matrix")
        if thresholds_cfg or threshold_matrix:
            payload["thresholds_config"] = {
                "thresholds": thresholds_cfg,
                "threshold_matrix": threshold_matrix,
                "asset_thresholds": asset_thresholds if asset_thresholds else None,
            }

        # --- Optional per-asset threshold statuses (Vel/Acc) applied to overall severity ---
        def eval_status(value: Optional[float], ok: Optional[float], warn: Optional[float]) -> Optional[str]:
            if value is None or ok is None or warn is None:
                return None
            try:
                v = float(value)
                o = float(ok)
                w = float(warn)
            except Exception:
                return None
            if v < o:
                return "ok"
            if v < w:
                return "warning"
            return "alarm"

        vel_status_thresh = None
        acc_status_thresh = None

        # Velocity per-asset thresholds
        vel_thr = asset_thresholds.get("vel_rms_mm_s") if asset_thresholds else None
        if vel_thr:
            vel_status_thresh = eval_status(
                metrics.get("rms_vel_max"), vel_thr.get("ok"), vel_thr.get("warning")
            )
        # Acceleration per-asset thresholds
        acc_thr = asset_thresholds.get("acc_rms_g") if asset_thresholds else None
        if acc_thr:
            acc_status_thresh = eval_status(
                metrics.get("rms_acc_max"), acc_thr.get("ok"), acc_thr.get("warning")
            )

        if vel_status_thresh:
            payload["velocity_rms_status_thresholds"] = vel_status_thresh
        if acc_status_thresh:
            payload["acc_rms_status_thresholds"] = acc_status_thresh

        # Escalate equipment severity by the worst of per-asset threshold statuses
        order = {"ok": 0, "warning": 1, "alarm": 2}
        worst = order.get(payload["equipment_severity"], 0)
        if vel_status_thresh:
            worst = max(worst, order.get(vel_status_thresh, 0))
        if acc_status_thresh:
            worst = max(worst, order.get(acc_status_thresh, 0))
        rev = {v: k for k, v in order.items()}
        payload["equipment_severity"] = rev.get(worst, payload["equipment_severity"])
        payload["status"] = payload["equipment_severity"]
        payload["severity_label"] = payload["equipment_severity"]

        # Legacy aliases to avoid breaking existing frontend:


        return payload
    except Exception as exc:
        return {"serialNumber": int(sn), "error": str(exc)}


# ---------------------------------------------------------------------
# Fleet diagnostics and summary
# ---------------------------------------------------------------------


def get_fleet_diagnostics() -> List[Dict[str, Any]]:
    """
    Build per-equipment diagnostics across the fleet using cached equipment IDs.
    """
    try:
        try:
            df_ids = get_equipment_id(use_cache=True)  # type: ignore[arg-type]
        except TypeError:
            df_ids = get_equipment_id()  # type: ignore[call-arg]

        if df_ids is None or df_ids.empty:
            return []

        diags: List[Dict[str, Any]] = []
        for _, row in df_ids.iterrows():
            sn = int(row["serialNumber"])
            diag = build_diagnostics_for_sn(sn)
            if diag.get("error"):
                continue
            # Ensure name field for UI (prefer asset_name)
            if "name" not in diag:
                diag["name"] = diag.get("asset_name") or diag.get("equip_type") or f"Equipment {sn}"
            overall = diag.get("overall")
            if isinstance(overall, dict):
                iso_zone = overall.get("iso_zone")
                diag.setdefault("iso_zone", iso_zone)
                diag.setdefault("iso_zone_label", iso_zone)
            diags.append(diag)

        return diags
    except Exception:
        return []


# Fault-type bucket mapping for pies
FAULT_TYPE_BUCKETS: Dict[str, str] = {
    # Unbalance
    "unbalance": "Unbalance",
    "imbalance": "Unbalance",

    # Misalignment
    "misalignment": "Misalignment",
    "angular_misalignment": "Misalignment",
    "parallel_misalignment": "Misalignment",

    # Looseness
    "looseness": "Looseness",
    "mechanical_looseness": "Looseness",
    "structural_looseness": "Looseness",

    # Bearing
    "bearing": "Bearing",
    "bearing_outer": "Bearing",
    "bearing_inner": "Bearing",
    "bearing_ball": "Bearing",
    "bearing_cage": "Bearing",
    "bpfo": "Bearing",
    "bpfi": "Bearing",
    "bsf": "Bearing",
    "ftf": "Bearing",

    # Belt
    "belt": "Belt Slip",
    "belt_slip": "Belt Slip",

    # Fan / blade
    "blade_pass": "Fan / Blade",
    "fan_blade": "Fan / Blade",

    # Gear
    "gear": "Gear Mesh",
    "gear_mesh": "Gear Mesh",
}


def _bucket_fault_id(raw_id: Optional[str]) -> str:
    if not raw_id:
        return "Other"
    raw_id_norm = str(raw_id).strip().lower()
    return FAULT_TYPE_BUCKETS.get(raw_id_norm, "Other")


def build_fleet_summary() -> Dict[str, Any]:
    """
    Aggregate diagnostics across the fleet into pies:
      - severity     (via equipment_severity)
      - fault_type   (from faults list)
      - iso_zone     (ISO 10816 / 20816)
      - equip_type   (motor, fan, pump, gearbox, compressor, etc.)
    """
    fleet_diags: List[Dict[str, Any]] = get_fleet_diagnostics()

    severity_counts: Counter[str] = Counter()
    fault_counts: Counter[str] = Counter()
    iso_counts: Counter[str] = Counter()
    equip_type_counts: Counter[str] = Counter()

    total_equipment = 0

    for diag in fleet_diags:
        if not isinstance(diag, dict):
            continue

        total_equipment += 1

        # --- Severity (OK / Warning / Alarm) ---
        equip_sev = _normalize_status(diag.get("equipment_severity") or "ok")
        label_map = {"ok": "OK", "warning": "Warning", "alarm": "Alarm"}
        sev_label = label_map.get(equip_sev, "OK")
        severity_counts[sev_label] += 1

        # --- ISO zone ---
        iso_zone = (
            diag.get("iso_zone")
            or diag.get("iso_zone_label")
            or diag.get("iso_zone_code")
        )
        if not iso_zone:
            iso_zone = "Unknown"
        else:
            iso_zone = str(iso_zone).strip() or "Unknown"
        iso_counts[iso_zone] += 1

        # --- Equipment type ---
        equip_type = diag.get("equip_type") or diag.get("type") or "Unknown"
        equip_type = str(equip_type).strip() or "Unknown"
        equip_type_counts[equip_type] += 1

        # --- Fault types ---
        faults = diag.get("faults") or []
        if isinstance(faults, dict):
            faults = list(faults.values())
        if not isinstance(faults, list):
            faults = []

        for f in faults:
            if not isinstance(f, dict):
                continue
            active = f.get("active")
            try:
                f_sev = float(f.get("severity", 0.0))
            except Exception:
                f_sev = 0.0
            if not active and f_sev < 0.1:
                continue

            raw_id = f.get("code") or f.get("id") or f.get("name") or ""
            bucket = _bucket_fault_id(raw_id)
            fault_counts[bucket] += 1

    def _to_pie_block(counter: Counter[str]) -> Dict[str, Any]:
        labels = list(counter.keys())
        values = [counter[label] for label in labels]
        return {"labels": labels, "values": values}

    return {
        "total_equipment": total_equipment,
        "severity": _to_pie_block(severity_counts),
        "fault_type": _to_pie_block(fault_counts),
        "iso_zone": _to_pie_block(iso_counts),
        "equip_type": _to_pie_block(equip_type_counts),
    }
