# app/api.py
from flask import Blueprint, request, jsonify, current_app, Response
import numpy as np
import pandas as pd
import orjson
import json
import os
from typing import Optional, Dict, Any
from app.data.EnhancedVibExtractor import get_rms_forecast
from .api_faults import build_diagnostics_for_sn
from .api_faults import build_fleet_summary


from .data.EnhancedVibExtractor import (
    execute_query,
    get_latest_vibration_data,
    get_equipment_id,
    get_metrics,
    info_dat,
    sensor_naming_map,
    get_cache_stats,
    refresh_all_caches,
    get_historical_data,
)

# ---------------------------------------------------------------------
# Config loader (for defaults like forecast horizon)
# ---------------------------------------------------------------------
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


def _cfg(section: str, key: str, default):
    try:
        return _APP_CFG.get(section, {}).get(key, default)
    except Exception:
        return default

# ---------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------
def orjsonify(obj, status: int = 200) -> Response:
    """Fast JSON responses using orjson."""
    return Response(orjson.dumps(obj), status=status, mimetype="application/json")


def to_f32_list(x):
    """Convert lists/ndarrays to float32 lists to shrink payload size."""
    if isinstance(x, list):
        return np.asarray(x, dtype=np.float32).tolist()
    if isinstance(x, np.ndarray):
        return x.astype(np.float32, copy=False).tolist()
    return x


def downsample_xy(x, y, max_n: int):
    # """
    # Simple decimation-based downsampling for spectra.
    # # """
    # x = np.asarray(x, dtype=float)
    # y = np.asarray(y, dtype=float)
    # n = len(x)
    # if max_n is None or max_n <= 0 or n <= max_n:
    #     return x, y
    # idx = np.linspace(0, n - 1, max_n).astype(int)
    # return x[idx], y[idx]
    return np.asarray(x, dtype=float), np.asarray(y, dtype=float)


def downsample_waveform(sig, max_n: int):
    """
    Simple decimation-based downsampling for time series.
    """
    sig = np.asarray(sig, dtype=float)
    n = len(sig)
    if max_n is None or max_n <= 0 or n <= max_n:
        return sig
    idx = np.linspace(0, n - 1, max_n).astype(int)
    return sig[idx]


def _as_float_list(v):
    """Robustly convert many stored formats into a simple list[float]."""
    import numpy as np
    import json, ast, pickle

    if v is None:
        return []

    # Already a Python sequence / numpy array
    if isinstance(v, (list, tuple, np.ndarray)):
        return [float(x) for x in list(v)]

    # BLOB from SQLite – try pickle first, then treat as text
    if isinstance(v, (bytes, bytearray)):
        # 1) Try pickle (our cache stores direction*, fft_* as pickled blobs)
        try:
            obj = pickle.loads(v)
            if isinstance(obj, (list, tuple, np.ndarray)):
                return [float(x) for x in list(obj)]
        except Exception:
            # 2) Fallback: decode to text and continue as string
            try:
                v = v.decode("utf-8", "ignore")
            except Exception:
                return []

    # String-like → try JSON / literal / "1,2,3" style
    if isinstance(v, str):
        s = v.strip()

        # a) JSON / literal
        for parser in (json.loads, ast.literal_eval):
            try:
                x = parser(s)
                if isinstance(x, (list, tuple)):
                    return [float(y) for y in x]
            except Exception:
                pass

        # b) Loose comma-separated “1,2,3”
        try:
            s2 = s.strip("[]")
            toks = [t for t in s2.replace("\n", " ").split(",") if t.strip()]
            return [float(t) for t in toks]
        except Exception:
            return []

    # Fallback – unknown type
    return []


def _parse_row_arrays(row: dict) -> dict:
    """Convert cached row fields into lists of floats."""
    out = dict(row)

    # waveform directions 1–3
    for d in (1, 2, 3):
        key = f"direction{d}"
        if key in out:
            out[key] = _as_float_list(out[key])

    # FFT arrays
    for k in [
        "frequencies",
        "fft_vel_d1",
        "fft_vel_d2",
        "fft_vel_d3",
        "fft_acc_d1",
        "fft_acc_d2",
        "fft_acc_d3",
    ]:
        if k in out:
            out[k] = _as_float_list(out[k])

    return out


def _load_latest_parsed(sn: int) -> Optional[Dict[str, Any]]:
    """
    Load latest cached row for a given serialNumber, and parse all arrays.
    CACHE-ONLY: uses EnhancedVibExtractor's cache (no live Databricks).
    """
    try:
        df = get_latest_vibration_data(serialNumber=sn, use_cache=True)
    except TypeError:
        # if EnhancedVibExtractor signature differs slightly, fall back
        df = get_latest_vibration_data(serialNumber=sn)

    if df is None or df.empty:
        return None

    row = df.iloc[0].to_dict()
    return _parse_row_arrays(row)


def _load_history_parsed(sn: int, limit: int = 60) -> pd.DataFrame:
    """
    Load daily-averaged historical RMS data (velocity + acceleration)
    for the given serialNumber from EnhancedVibExtractor cache.
    """

    try:
        # Pull cached historical data (rms_vel_d*, rms_acc_d*, timestamps)
        df = get_historical_data(serialNumber=sn, use_cache=True)
    except Exception as e:
        print("History load error:", e)
        return pd.DataFrame()

    if df is None or df.empty:
        return pd.DataFrame()

    # --- Normalize time column name ---
    if "time" in df.columns:
        time_col = "time"
    elif "mytimestamp" in df.columns:
        time_col = "mytimestamp"
    else:
        # Fallback: first column that looks datetime-like
        time_col = df.columns[0]

    df = df.copy()

    # If already datetime64, use it directly; otherwise parse robustly
    if not pd.api.types.is_datetime64_any_dtype(df[time_col]):
        df[time_col] = pd.to_datetime(
            df[time_col],
            utc=True,
            errors="coerce",  # invalid strings -> NaT, not crash
        )

    # Drop rows where time could not be parsed
    df = df.dropna(subset=[time_col])

    # Use calendar date (no time-of-day)
    df["date"] = df[time_col].dt.date

    # --- Keep only the relevant RMS columns ---
    cols_needed = [
        "date",
        "rms_vel_d1", "rms_vel_d2", "rms_vel_d3",
        "rms_acc_d1", "rms_acc_d2", "rms_acc_d3",
    ]
    existing_cols = [c for c in cols_needed if c in df.columns]
    if "date" not in existing_cols:
        # nothing usable
        return pd.DataFrame()
    df = df[existing_cols]

    # --- Aggregate to daily averages ---
    df_daily = (
        df.groupby("date")
        .mean(numeric_only=True)
        .reset_index()
        .sort_values("date", ascending=True)
    )

    # Convert to string for JSON output
    df_daily["date"] = df_daily["date"].astype(str)

    # Limit to last N days
    if limit:
        df_daily = df_daily.tail(int(limit))

    return df_daily


def as_int(v, default=None):
    try:
        return int(v)
    except Exception:
        return default


# ---------------------------------------------------------------------
# Blueprint
# ---------------------------------------------------------------------
api = Blueprint("api", __name__)

# ---------------------------------------------------------------------
# Basic health + cache status
# ---------------------------------------------------------------------


@api.route("/diagnostics/fleet", methods=["GET"])
def diagnostics_fleet():
    """
    Fleet-level per-equipment diagnostics for the SummaryPage.
    """
    try:
        # 1) Get list of serial numbers (cache-first)
        try:
            df_ids = get_equipment_id(use_cache=True)
        except TypeError:
            df_ids = get_equipment_id()

        if df_ids is None or df_ids.empty:
            return orjsonify({"status": "ok", "items": []})

        items = []

        for _, r in df_ids.iterrows():
            sn = int(r["serialNumber"])

            # Load latest cached row and parse arrays
            row = _load_latest_parsed(sn)
            if not row:
                continue

            df = pd.DataFrame([row])

            # IMPORTANT: get_metrics(df, mySN) always gets 2 args
            metrics = get_metrics(df, sn)

            # ---- Extract fault summary (codes) ----
            # Prefer EnhancedVibExtractor's JSON-ready "fault_summary"
            f = metrics.get("fault_summary")

            # Fallback: if only a DataFrame/list is present under "fault"
            if f is None:
                raw_fault = metrics.get("fault")
                if isinstance(raw_fault, list) and raw_fault:
                    f = raw_fault[0]
                elif isinstance(raw_fault, pd.DataFrame) and not raw_fault.empty:
                    f = raw_fault.iloc[0].to_dict()

            if f is None:
                continue

            # Map codes (u0/u1/u2/u3, m0..m3, l0..l3, brms1..3) to ok/warning/alarm
            mis_s = _fault_code_to_status(f.get("misalignment"), "misalignment")
            unb_s = _fault_code_to_status(f.get("unbalance"), "unbalance")
            loos_s = _fault_code_to_status(f.get("loosenes"), "loosenes")
            brms_s = _fault_code_to_status(f.get("bearing_rms"), "bearing_rms")

            # ---------- Unified equipment severity ----------
            fault_statuses = (mis_s, unb_s, loos_s, brms_s)
            if "alarm" in fault_statuses:
                equipment_severity = "alarm"
            elif "warning" in fault_statuses:
                equipment_severity = "warning"
            else:
                equipment_severity = "ok"

            # Legacy overall_status string for backwards compatibility
            LEGACY_STATUS_MAP = {
                "ok": "OK",
                "warning": "Warning",
                "alarm": "Alarm",
            }
            overall_status = LEGACY_STATUS_MAP.get(equipment_severity, "OK")

            # ---------- Severity score for ranking ----------
            sev_raw = metrics.get("fault_severity")
            try:
                severity_score = float(sev_raw)
            except (TypeError, ValueError):
                SCORE_MAP = {"ok": 1.0, "warning": 0.5, "alarm": 0.0}
                severity_score = SCORE_MAP.get(equipment_severity, 0.5)

            # ---------- RPM + naming ----------
            rpm_hz = (
                metrics.get("cal_rpm_hz")
                or metrics.get("guess_rpm_hz")
                or metrics.get("cal_rpm")
                or metrics.get("guess_rpm")
            )
            rpm_rpm = float(rpm_hz * 60.0) if rpm_hz is not None else None

            # Name / label: prefer info_dat.AssetName, then sensor_naming_map, then SN
            try:
                name = f"SN {sn}"
                row_info = info_dat[info_dat["serialNumber"] == sn]
                if not row_info.empty and "AssetName" in row_info.columns:
                    val = row_info["AssetName"].values[0]
                    if isinstance(val, str) and val.strip():
                        name = val.strip()
                if name == f"SN {sn}":
                    name = sensor_naming_map.get(sn, name)
            except Exception:
                name = f"SN {sn}"

            # Optional equipment type if available in df_ids or metrics
            equip_type = None
            try:
                if "equip_type" in r:
                    equip_type = r["equip_type"]
            except Exception:
                pass
            if not equip_type:
                equip_type = metrics.get("equip_type")

            # Per-asset thresholds (if provided in info_dat)
            asset_thresholds = {}
            try:
                row_info = info_dat[info_dat["serialNumber"] == sn]
                if not row_info.empty:
                    ri = row_info.iloc[0]
                    vel_ok = ri.get("VelRmsOk_mm_s")
                    vel_warn = ri.get("VelRmsWarning_mm_s")
                    vel_alarm = ri.get("VelRmsAlarm_mm_s")
                    if pd.notna(vel_ok) and pd.notna(vel_warn) and pd.notna(vel_alarm):
                        asset_thresholds["vel_rms_mm_s"] = {
                            "ok": float(vel_ok),
                            "warning": float(vel_warn),
                            "alarm": float(vel_alarm),
                        }
                    acc_ok = ri.get("AccRmsOk_g")
                    acc_warn = ri.get("AccRmsWarning_g")
                    acc_alarm = ri.get("AccRmsAlarm_g")
                    if pd.notna(acc_ok) and pd.notna(acc_warn) and pd.notna(acc_alarm):
                        asset_thresholds["acc_rms_g"] = {
                            "ok": float(acc_ok),
                            "warning": float(acc_warn),
                            "alarm": float(acc_alarm),
                        }
            except Exception:
                asset_thresholds = {}

            def eval_status(val, ok, warn):
                try:
                    if val is None or ok is None or warn is None:
                        return None
                    v = float(val)
                    o = float(ok)
                    w = float(warn)
                except Exception:
                    return None
                if v < o:
                    return "ok"
                if v < w:
                    return "warning"
                return "alarm"

            vel_thr = asset_thresholds.get("vel_rms_mm_s") if asset_thresholds else None
            acc_thr = asset_thresholds.get("acc_rms_g") if asset_thresholds else None
            vel_status_thresh = (
                eval_status(metrics.get("rms_vel_max"), vel_thr.get("ok"), vel_thr.get("warning"))
                if vel_thr else None
            )
            acc_status_thresh = (
                eval_status(metrics.get("rms_acc_max"), acc_thr.get("ok"), acc_thr.get("warning"))
                if acc_thr else None
            )

            # Bearing: prefer acc threshold status if present
            bearing_final = acc_status_thresh if acc_status_thresh else brms_s
            # Unbalance: keep harmonics-based (leave velocity thresholds for gauge only)
            unbalance_final = unb_s

            # Recompute equipment severity using the updated bearing/unbalance
            order = {"ok": 0, "warning": 1, "alarm": 2}
            fault_statuses = (mis_s, unbalance_final, loos_s, bearing_final)
            worst_fault = max(order.get(s, 0) for s in fault_statuses)
            # Also consider threshold-only statuses if present
            if vel_status_thresh:
                worst_fault = max(worst_fault, order.get(vel_status_thresh, 0))
            if acc_status_thresh:
                worst_fault = max(worst_fault, order.get(acc_status_thresh, 0))
            rev = {v: k for k, v in order.items()}
            equipment_severity = rev.get(worst_fault, "ok")
            overall_status = LEGACY_STATUS_MAP.get(equipment_severity, "OK")

            items.append(
                {
                    "serialNumber": sn,
                    "id": sn,
                    "name": name,
                    "status": overall_status,                  # legacy text
                    "equipment_severity": equipment_severity,  # "ok" | "warning" | "alarm"
                    "severity_score": severity_score,          # 0..1
                    "misalignment": mis_s,
                    "unbalance": unbalance_final,
                    "looseness": loos_s,
                    "bearing": bearing_final,
                    "rpmHz": rpm_hz,
                    "rpmRpm": rpm_rpm,
                    "rpmConfidence": metrics.get("rpm_confidence", "unknown"),
                    "equip_type": equip_type,
                    "velocity_rms_status_thresholds": vel_status_thresh,
                    "acc_rms_status_thresholds": acc_status_thresh,
                }
            )

        return orjsonify({"status": "ok", "items": items})

    except Exception as exc:
        current_app.logger.exception("diagnostics_fleet failed")
        return orjsonify({"status": "error", "error": str(exc)}, 500)


@api.get("/health")
def health():
    try:
        stats = get_cache_stats() or {}
        return orjsonify({"status": "ok", "cache": stats})
    except Exception as exc:
        return orjsonify({"status": "error", "error": str(exc)}, 500)


@api.get("/equipment")
def equipment():
    """
    Equipment list for SummaryPage / EquipmentPage.
    Uses cache-only equipment_id + info_dat / sensor_naming_map.
    """
    debug = as_int(request.args.get("debug", 0), 0)

    try:
        df_ids = get_equipment_id(use_cache=True)
    except TypeError:
        df_ids = get_equipment_id()

    if df_ids is None or df_ids.empty:
        return orjsonify([])

    rows = []
    for _, r in df_ids.iterrows():
        sn = int(r["serialNumber"])
        # Prefer AssetName from info_dat if available, else sensor_naming_map, else SN
        name = f"SN {sn}"
        try:
            row_info = info_dat[info_dat["serialNumber"] == sn]
            if not row_info.empty and "AssetName" in row_info.columns:
                asset_name_val = row_info["AssetName"].values[0]
                if isinstance(asset_name_val, str) and asset_name_val.strip():
                    name = asset_name_val.strip()
        except Exception:
            pass
        if name == f"SN {sn}":
            name = sensor_naming_map.get(sn, name)

        rows.append(
            {
                "id": sn,
                "name": name,
                "status": "OK",  # placeholder; UI derives health via diagnostics
            }
        )

    if debug:
        return orjsonify({"items": rows})

    return orjsonify(rows)


@api.route("/diagnostics", methods=["GET"])
def diagnostics():
    """
    Per-equipment diagnostics for EquipmentDetailPage.

    GET /api/diagnostics?sn=189314686 -> {
      "serialNumber": 189314686,
      "metrics": { ... }
    }
    """
    try:
        sn = request.args.get("sn", type=int)
        if sn is None:
            return orjsonify({"error": "Missing sn"}), 400

        df = get_latest_vibration_data(serialNumber=sn)
        if df is None or df.empty:
            return orjsonify({"error": "No data for this serialNumber"}), 404

        metrics = get_metrics(df, sn)

        m = dict(metrics)

        # Attach thresholds config for frontend gauges (same as api_faults)
        # Include DrivenBySN (if present) so frontend can fetch the driver/motor health
        try:
            info_row = info_dat[info_dat["serialNumber"] == sn]
            if not info_row.empty and "DrivenBySN" in info_row.columns:
                driven_val = info_row.iloc[0].get("DrivenBySN")
                if pd.notna(driven_val):
                    m["DrivenBySN"] = int(driven_val)
        except Exception:
            pass

        try:
            # Per-asset overrides (optional) from info_dat
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

            m["thresholds_config"] = {
                "thresholds": _APP_CFG.get("thresholds"),
                "threshold_matrix": _APP_CFG.get("threshold_matrix"),
                "asset_thresholds": asset_thresholds if asset_thresholds else None,
            }
        except Exception:
            pass
        # Also include per-asset threshold statuses (for UI cards) using latest metrics
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
        vel_thr = asset_thresholds.get("vel_rms_mm_s") if asset_thresholds else None
        if vel_thr:
            vel_status_thresh = eval_status(m.get("rms_vel_max"), vel_thr.get("ok"), vel_thr.get("warning"))
        acc_thr = asset_thresholds.get("acc_rms_g") if asset_thresholds else None
        if acc_thr:
            acc_status_thresh = eval_status(m.get("rms_acc_max"), acc_thr.get("ok"), acc_thr.get("warning"))
        if vel_status_thresh:
            m["velocity_rms_status_thresholds"] = vel_status_thresh
        if acc_status_thresh:
            m["acc_rms_status_thresholds"] = acc_status_thresh

        def df_to_records(val):
            if isinstance(val, pd.DataFrame):
                return val.to_dict(orient="records")
            return val

        for key in [
            "mc_fault_rms_res",
            "mc_fault_peak_res",
            "dat_harmonics",
            "faults",
            "fault",
            "faults_by_direction",
        ]:
            if key in m:
                m[key] = df_to_records(m[key])

        return orjsonify({"serialNumber": sn, "metrics": m})

    except Exception as exc:
        current_app.logger.exception("diagnostics failed")
        return orjsonify({"error": str(exc)}), 500


@api.get("/history")
def history():
    """
    Historical RMS trend for a given SN (daily averages).
    Used by TrendPlot and AccTrendPlot.
    """
    sn = request.args.get("sn")
    limit = as_int(request.args.get("limit", 200), 200)

    if not sn:
        return orjsonify({"error": "sn required"}, 400)

    sn = int(sn)
    df = _load_history_parsed(sn, limit=limit)

    if df is None or df.empty:
        return orjsonify([])

    if "date" in df.columns:
        df = df.sort_values("date")
    elif "time" in df.columns:
        df = df.sort_values("time")

    # Remap channels to canonical V/H/A order if info_dat specifies custom mapping
    try:
        info_row = info_dat[info_dat["serialNumber"] == sn]
        if not info_row.empty:
            ir = info_row.iloc[0]
            vert_ch = int(ir.get("SensorVerticalChannel") or 1)
            horiz_ch = int(ir.get("SensorHorizontalChannel") or 2)
            axial_ch = int(ir.get("SensorAxialChannel") or 3)
            channel_map = [vert_ch, horiz_ch, axial_ch]  # dir1, dir2, dir3

            def remap_row(row):
                out = dict(row)
                # velocity
                src_vel = [
                    row.get("rms_vel_d1"),
                    row.get("rms_vel_d2"),
                    row.get("rms_vel_d3"),
                ]
                # acceleration
                src_acc = [
                    row.get("rms_acc_d1"),
                    row.get("rms_acc_d2"),
                    row.get("rms_acc_d3"),
                ]
                for idx, ch in enumerate(channel_map, start=1):
                    if 1 <= ch <= 3:
                        out[f"rms_vel_d{idx}"] = src_vel[ch - 1]
                        out[f"rms_acc_d{idx}"] = src_acc[ch - 1]
                return out

            df = pd.DataFrame([remap_row(r) for r in df.to_dict(orient="records")])
    except Exception:
        pass

    return orjsonify(df.to_dict(orient="records"))


@api.get("/spectrum")
def spectrum():
    """
    Velocity spectrum endpoint.
    - sn: serialNumber
    - dir: 1/2/3 (0 → auto-select best)
    - max_n: max points for downsampling
    """
    sn = request.args.get("sn")
    direction = as_int(request.args.get("dir", 0), 0)
    max_n_req = as_int(request.args.get("max_n", 2048), 2048)

    if not sn:
        return orjsonify({"error": "sn required"}, 400)

    max_n = 1024 if max_n_req <= 1024 else 2048

    sn = int(sn)
    row = _load_latest_parsed(sn)
    if not row:
        return orjsonify({"error": "no data"}, 404)

    freqs = row.get("frequencies") or []
    freqs = _as_float_list(freqs)

    if not freqs:
        return orjsonify({"error": "no frequencies"}, 404)

    if direction not in (1, 2, 3):
        try:
            peaks = [
                float(row.get("rms_acc_fft1000_d1") or 0),
                float(row.get("rms_acc_fft1000_d2") or 0),
                float(row.get("rms_acc_fft1000_d3") or 0),
            ]
            direction = int(np.argmax(peaks)) + 1
        except Exception:
            direction = 3

    # Apply channel mapping: dir (1=Vertical...) -> actual channel
    try:
        info_row = info_dat[info_dat["serialNumber"] == sn]
        if not info_row.empty:
            ir = info_row.iloc[0]
            channel_map = {
                1: int(ir.get("SensorVerticalChannel") or 1),
                2: int(ir.get("SensorHorizontalChannel") or 2),
                3: int(ir.get("SensorAxialChannel") or 3),
            }
            direction = channel_map.get(direction, direction)
    except Exception:
        pass

    spec = row.get(f"fft_vel_d{direction}") or []
    spec = _as_float_list(spec)

    fx, fy = downsample_xy(freqs, spec, max_n)
    out = {
        "serialNumber": sn,
        "direction": direction,
        "frequencies": to_f32_list(fx),
        "fft_vel": to_f32_list(fy),
    }
    return orjsonify(out)


@api.get("/spectrum_acc")
def spectrum_acc():
    """
    Acceleration spectrum endpoint.
    Same semantics as /spectrum but returns fft_acc.
    """
    sn = request.args.get("sn")
    direction = as_int(request.args.get("dir", 0), 0)
    max_n_req = as_int(request.args.get("max_n", 2048), 2048)

    if not sn:
        return orjsonify({"error": "sn required"}, 400)

    max_n = 1024 if max_n_req <= 1024 else 2048
    sn = int(sn)

    row = _load_latest_parsed(sn)
    if not row:
        return orjsonify({"error": "no data"}, 404)

    freqs = _as_float_list(row.get("frequencies") or [])
    if not freqs:
        return orjsonify({"error": "no frequencies"}, 404)

    if direction not in (1, 2, 3):
        try:
            peaks = [
                float(row.get("rms_acc_fft1000_d1") or 0),
                float(row.get("rms_acc_fft1000_d2") or 0),
                float(row.get("rms_acc_fft1000_d3") or 0),
            ]
            direction = int(np.argmax(peaks)) + 1
        except Exception:
            direction = 3

    # Apply channel mapping for acceleration spectrum
    try:
        info_row = info_dat[info_dat["serialNumber"] == sn]
        if not info_row.empty:
            ir = info_row.iloc[0]
            channel_map = {
                1: int(ir.get("SensorVerticalChannel") or 1),
                2: int(ir.get("SensorHorizontalChannel") or 2),
                3: int(ir.get("SensorAxialChannel") or 3),
            }
            direction = channel_map.get(direction, direction)
    except Exception:
        pass

    spec = _as_float_list(row.get(f"fft_acc_d{direction}") or [])
    fx, fy = downsample_xy(freqs, spec, max_n)

    out = {
        "serialNumber": sn,
        "direction": direction,
        "frequencies": to_f32_list(fx),
        "fft_acc": to_f32_list(fy),
    }
    return orjsonify(out)


@api.get("/waveform")
def waveform():
    """
    Waveform endpoint for a single direction.
    - sn: serialNumber
    - dir: 1/2/3 (0 → auto-select)
    - max_n: max samples
    """
    sn = request.args.get("sn")
    direction = as_int(request.args.get("dir", 0), 0)
    max_n_req = as_int(request.args.get("max_n", 4096), 4096)

    if not sn:
        return orjsonify({"error": "sn required"}, 400)

    max_n = 2048 if max_n_req <= 2048 else 4096
    sn = int(sn)

    row = _load_latest_parsed(sn)
    if not row:
        return orjsonify({"error": "no data"}, 404)

    if direction not in (1, 2, 3):
        try:
            peaks = [
                float(row.get("rms_acc_d1") or 0),
                float(row.get("rms_acc_d2") or 0),
                float(row.get("rms_acc_d3") or 0),
            ]
            direction = int(np.argmax(peaks)) + 1
        except Exception:
            direction = 3

    # Apply channel mapping for waveform
    try:
        info_row = info_dat[info_dat["serialNumber"] == sn]
        if not info_row.empty:
            ir = info_row.iloc[0]
            channel_map = {
                1: int(ir.get("SensorVerticalChannel") or 1),
                2: int(ir.get("SensorHorizontalChannel") or 2),
                3: int(ir.get("SensorAxialChannel") or 3),
            }
            direction = channel_map.get(direction, direction)
    except Exception:
        pass

    sig = row.get(f"direction{direction}") or []
    sig = _as_float_list(sig)
    sr = as_int(row.get("sampleRate"), default=0)

    if len(sig) > max_n:
        sig = downsample_waveform(sig, max_n)

    # For downstream labeling, map back from actual channel to canonical dir
    dir_for_label = direction
    try:
        info_row = info_dat[info_dat["serialNumber"] == sn]
        if not info_row.empty:
            ir = info_row.iloc[0]
            channel_map = {
                1: int(ir.get("SensorVerticalChannel") or 1),
                2: int(ir.get("SensorHorizontalChannel") or 2),
                3: int(ir.get("SensorAxialChannel") or 3),
            }
            inverse_map = {v: k for k, v in channel_map.items()}
            dir_for_label = inverse_map.get(direction, direction)
    except Exception:
        pass

    out = {
        "serialNumber": sn,
        "direction": dir_for_label,
        "sampleRate": sr,
        "signal": to_f32_list(sig),
    }
    return orjsonify(out)


@api.get("/waveform_vel")
def waveform_vel():
    """
    Velocity waveform endpoint for a single direction.
    - sn: serialNumber
    - dir: 1/2/3 (0 → auto-select by RMS velocity)
    - max_n: max samples
    """
    sn = request.args.get("sn")
    direction = as_int(request.args.get("dir", 0), 0)
    max_n_req = as_int(request.args.get("max_n", 4096), 4096)

    if not sn:
        return orjsonify({"error": "sn required"}, 400)

    max_n = 2048 if max_n_req <= 2048 else 4096
    sn = int(sn)

    row = _load_latest_parsed(sn)
    if not row:
        return orjsonify({"error": "no data"}, 404)

    if direction not in (1, 2, 3):
        try:
            peaks = [
                float(row.get("rms_vel_d1") or 0),
                float(row.get("rms_vel_d2") or 0),
                float(row.get("rms_vel_d3") or 0),
            ]
            direction = int(np.argmax(peaks)) + 1
        except Exception:
            direction = 3

    # Apply channel mapping
    try:
        info_row = info_dat[info_dat["serialNumber"] == sn]
        if not info_row.empty:
            ir = info_row.iloc[0]
            channel_map = {
                1: int(ir.get("SensorVerticalChannel") or 1),
                2: int(ir.get("SensorHorizontalChannel") or 2),
                3: int(ir.get("SensorAxialChannel") or 3),
            }
            direction = channel_map.get(direction, direction)
    except Exception:
        pass

    sig_raw = row.get(f"dat_vel_d{direction}")
    sig = _as_float_list(sig_raw) if sig_raw is not None else []

    sr = as_int(row.get("sampleRate"), default=0)

    if len(sig) > max_n:
        sig = downsample_waveform(sig, max_n)

    # Map back for labeling
    dir_for_label = direction
    try:
        info_row = info_dat[info_dat["serialNumber"] == sn]
        if not info_row.empty:
            ir = info_row.iloc[0]
            channel_map = {
                1: int(ir.get("SensorVerticalChannel") or 1),
                2: int(ir.get("SensorHorizontalChannel") or 2),
                3: int(ir.get("SensorAxialChannel") or 3),
            }
            inverse_map = {v: k for k, v in channel_map.items()}
            dir_for_label = inverse_map.get(direction, direction)
    except Exception:
        pass

    out = {
        "serialNumber": sn,
        "direction": dir_for_label,
        "sampleRate": sr,
        "signal": to_f32_list(sig),
    }
    return orjsonify(out)


@api.get("/spectrum_env")
def spectrum_env():
    """
    Envelope spectrum via analytic signal (Hilbert) computed from the
    acceleration waveform in the dominant direction.
    """
    sn = request.args.get("sn")
    direction = as_int(request.args.get("dir", 0), 0)
    max_n_req = as_int(request.args.get("max_n", 2048), 2048)

    if not sn:
        return orjsonify({"error": "sn required"}, 400)

    max_n = 1024 if max_n_req <= 1024 else 2048
    sn = int(sn)

    row = _load_latest_parsed(sn)
    if not row:
        return orjsonify({"error": "no data"}, 404)

    if direction not in (1, 2, 3):
        try:
            peaks = [
                float(row.get("rms_acc_d1") or 0),
                float(row.get("rms_acc_d2") or 0),
                float(row.get("rms_acc_d3") or 0),
            ]
            direction = int(np.argmax(peaks)) + 1
        except Exception:
            direction = 3

    sig = _as_float_list(row.get(f"direction{direction}") or [])
    sr = float(row.get("sampleRate") or 0)
    if not sig or sr <= 0:
        return orjsonify({"error": "no waveform"}, 404)

    sig = np.asarray(sig, dtype=float)
    n = len(sig)

    Xf = np.fft.fft(sig)
    h = np.zeros(n)
    if n % 2 == 0:
        h[0] = 1.0
        h[n // 2] = 1.0
        h[1: n // 2] = 2.0
    else:
        h[0] = 1.0
        h[1: (n + 1) // 2] = 2.0

    analytic = np.fft.ifft(Xf * h)
    env = np.abs(analytic)

    spec = np.abs(np.fft.rfft(env)) * (2.0 / n)
    freqs = np.fft.rfftfreq(n, d=1.0 / sr)

    fx, fy = downsample_xy(freqs, spec, max_n)

    out = {
        "serialNumber": sn,
        "direction": direction,
        "sampleRate": sr,
        "frequencies": to_f32_list(fx),
        "fft_env": to_f32_list(fy),
    }
    return orjsonify(out)


# ---------------------------------------------------------------------
# Cache refresh
# ---------------------------------------------------------------------
@api.post("/cache/refresh")
def cache_refresh():
    """
    Trigger a cache refresh using the EnhancedVibExtractor helper.
    """
    try:
        ok = bool(refresh_all_caches())
        details = get_cache_stats() or {}
        return orjsonify({"success": ok, "details": details})
    except Exception as e:
        current_app.logger.exception("cache_refresh failed")
        return orjsonify({"success": False, "error": str(e)}, 500)


def _fault_code_to_status(code: str, kind: str) -> str:
    """
    Convert VibExtractor fault code (u1/u2/u3, m1/m2/m3, etc.)
    into 'ok' / 'warning' / 'alarm' for the UI.
    """
    if not code:
        return "ok"

    code = str(code).lower().strip()

    if kind in ("unbalance", "loosenes", "misalignment"):
        if code.endswith("3"):
            return "alarm"
        elif code.endswith("2"):
            return "warning"
        else:
            return "ok"

    if kind == "bearing_rms":
        if code.endswith("3"):
            return "alarm"
        elif code.endswith("2"):
            return "warning"
        else:
            return "ok"

    return "ok"


@api.route("/diagnostics/summary", methods=["GET"])
def diagnostics_summary():
    """
    Fleet-wide summary pies:
      - severity
      - fault_type
      - iso_zone
      - equip_type
    """
    try:
        summary = build_fleet_summary()
        return jsonify(summary)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ---------------------------------------------------------------------
# RMS forecast severity helper
# ---------------------------------------------------------------------

def classify_rms_value_mm_s(value_mm_s: float) -> Dict[str, Any]:
    """
    Classify a velocity RMS value (mm/s) into an ISO-like group and
    unified status ("ok" | "warning" | "alarm").

    NOTE: Thresholds here should match your ISO 10816/20816 configuration
    as closely as possible. Adjust if needed.
    """
    v = float(value_mm_s)

    # Example thresholds (tune these to your machinery class):
    #  - < 2.8 mm/s  → group 2 (OK)
    #  - 2.8–4.5     → group 3 (Warning)
    #  - > 4.5       → group 4 (Alarm)
    if v < 2.8:
        group = 2
    elif v < 4.5:
        group = 3
    else:
        group = 4

    if group <= 2:
        status = "ok"
    elif group == 3:
        status = "warning"
    else:
        status = "alarm"

    return {"group": group, "status": status}


@api.route("/equipment/<int:sn>/forecast", methods=["GET"])
def api_equipment_forecast(sn: int):
    """
    Example:
      /api/equipment/189314686/forecast?metric=rms_vel&direction=d1&model=arima&horizon=14

    Returns the original forecast dict from EnhancedVibExtractor.get_rms_forecast,
    plus an extra:

      "forecast_status": {
        "group": <int>,        # 2/3/4 etc
        "status": "ok" | "warning" | "alarm"
      }

    computed from the LAST forecast RMS value using classify_rms_value_mm_s().
    """
    metric = request.args.get("metric", "rms_vel")
    direction = request.args.get("direction", "d1")
    model = request.args.get("model", "lgbm")  # "lgbm" | "linear" | "arima"
    default_horizon = _cfg("forecast", "DEFAULT_HORIZON_DAYS", 14)
    horizon = int(request.args.get("horizon", default_horizon))

    # Map canonical direction (d1=Vertical,d2=Horizontal,d3=Axial) to actual channel per info_dat
    def map_direction(dir_str: str) -> str:
        try:
            dnum = int(str(dir_str).lower().replace("d", "") or "1")
        except Exception:
            dnum = 1
        try:
            row = info_dat[info_dat["serialNumber"] == sn]
            if not row.empty:
                ir = row.iloc[0]
                channel_map = {
                    1: int(ir.get("SensorVerticalChannel") or 1),
                    2: int(ir.get("SensorHorizontalChannel") or 2),
                    3: int(ir.get("SensorAxialChannel") or 3),
                }
                mapped = channel_map.get(dnum, dnum)
                return f"d{mapped}"
        except Exception:
            pass
        return f"d{dnum}"

    mapped_direction = map_direction(direction)

    base = get_rms_forecast(
        serialNumber=sn,
        metric=metric,
        direction=mapped_direction,
        horizon_days=horizon,
        model=model,
    )

    # Try to infer the last forecast RMS value in a robust way
    forecast = base.get("forecast")
    last_val = None

    if isinstance(forecast, list) and forecast:
        last = forecast[-1]
        if isinstance(last, dict):
            # common keys that might hold the value
            for key in ("value", "y", "rms", "rms_vel", "metric_value"):
                if key in last and last[key] is not None:
                    last_val = last[key]
                    break
        elif isinstance(last, (int, float)):
            last_val = last

    forecast_status = None
    if last_val is not None:
        try:
            forecast_status = classify_rms_value_mm_s(float(last_val))
        except Exception:
            forecast_status = None

    base["forecast_status"] = forecast_status

    return jsonify(base)


@api.route("/diagnostics/<int:sn>", methods=["GET"])
def diagnostics_sn(sn: int):
    """
    Detailed diagnostics for a single equipment:
      GET /api/diagnostics/189314686
    """
    payload = build_diagnostics_for_sn(sn)
    status = 200 if "error" not in payload else 400
    return jsonify(payload), status
