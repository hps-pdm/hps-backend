from __future__ import annotations

"""
VibExtractor.py
===============

Core vibration-analysis backend for:
- Pulling FFT & RMS data from Databricks.
- Estimating RPM (1×) robustly from FFT.
- Classifying faults (unbalance, looseness, misalignment, bearing, type-specific).
- Mapping overall RMS to ISO 10816/20816 severity zones.
- Returning a rich metrics dict per equipment for the diagnostics API.

This module is used by:
- /api/diagnostics
- /api/diagnostics/fleet
- /api/diagnostics/summary
"""

# -------------------------------------------------------------------------
# Imports
# -------------------------------------------------------------------------
from databricks import sql
from dotenv import load_dotenv
from scipy.interpolate import interp1d

import os
import json
import numpy as np
import pandas as pd

# -------------------------------------------------------------------------
# Environment setup (load .env from project root)
# -------------------------------------------------------------------------
env_path = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(__file__))), ".env"
)
load_dotenv(dotenv_path=env_path)

# -------------------------------------------------------------------------
# Databricks connection + generic query helper
# -------------------------------------------------------------------------
def create_connection():
    """
    Create a Databricks SQL connection using credentials from .env:

    - DATABRICKS_SERVER
    - DATABRICKS_HTTP_PATH
    - DATABRICKS_TOKEN
    """
    return sql.connect(
        server_hostname=os.getenv("DATABRICKS_SERVER"),
        http_path=os.getenv("DATABRICKS_HTTP_PATH"),
        access_token=os.getenv("DATABRICKS_TOKEN"),
    )


def execute_query(query: str) -> pd.DataFrame:
    """
    Execute a SQL query against Databricks and return a pandas DataFrame.
    """
    connection = create_connection()
    cursor = connection.cursor()
    try:
        cursor.execute(query)
        data = cursor.fetchall()
        columns = [desc[0] for desc in cursor.description]
        return pd.DataFrame(data, columns=columns)
    finally:
        cursor.close()
        connection.close()


DEFAULT_DIM_TABLE = "hive_metastore.vibration.dim_equipment"
LEGACY_DIM_TABLE = "vibration.dim_equipment"
LOCAL_INFO_JSON = os.path.join(os.path.dirname(__file__), "info_dat.json")
BEARING_RMS_JSON = os.path.join(os.path.dirname(__file__), "bearing_fault_rms.json")
APP_CONFIG_JSON = os.path.join(os.path.dirname(__file__), "..", "config", "config.json")


def ensure_dim_equipment_table():
    """
    Create the vibration.dim_equipment table if it does not already exist.
    Uses the DDL stored in backend/app/data/dim_equipment.sql.
    """
    ddl_path = os.path.join(os.path.dirname(__file__), "dim_equipment.sql")
    if not os.path.exists(ddl_path):
        raise FileNotFoundError(f"dim_equipment.sql not found at {ddl_path}")

    with open(ddl_path, "r", encoding="utf-8") as f:
        ddl = f.read()

    conn = create_connection()
    cur = conn.cursor()
    try:
        # Split statements so the driver doesn’t choke on multiple commands
        stmts = [s.strip() for s in ddl.split(";") if s.strip()]
        for stmt in stmts:
            cur.execute(stmt)
    finally:
        cur.close()
        conn.close()

# -------------------------------------------------------------------------
# High-level data fetchers from Databricks
# -------------------------------------------------------------------------
def get_latest_vibration_data(serialNumber=None) -> pd.DataFrame:
    """
    Return the latest vibration record per serialNumber from
    vibration.silver.ffts_waveforms, filtered with rms_acc_d3 > 0.35.

    If serialNumber is provided, return only that asset’s latest record.
    Otherwise return one record per asset.
    """
    if serialNumber:
        query = f"""
            WITH RankedData AS (
                SELECT *,
                       ROW_NUMBER() OVER (PARTITION BY serialNumber ORDER BY EnqueuedTimeUtc DESC) AS rn
                FROM vibration.silver.ffts_waveforms
                WHERE rms_acc_d3 > 0.35 AND serialNumber = {serialNumber}
            )
            SELECT * FROM RankedData WHERE rn = 1;
        """
    else:
        query = """
            WITH RankedData AS (
                SELECT *,
                       ROW_NUMBER() OVER (PARTITION BY serialNumber ORDER BY EnqueuedTimeUtc DESC) AS rn
                FROM vibration.silver.ffts_waveforms
                WHERE rms_acc_d3 > 0.35
            )
            SELECT * FROM RankedData WHERE rn = 1;
        """
    return execute_query(query)


def get_equipment_id() -> pd.DataFrame:
    """
    Return distinct serialNumber values from the latest vibration records.
    Useful for populating the equipment list.
    """
    query = """
        WITH RankedData AS (
            SELECT *,
                   ROW_NUMBER() OVER (PARTITION BY serialNumber ORDER BY EnqueuedTimeUtc DESC) AS rn
            FROM vibration.silver.ffts_waveforms
            WHERE rms_acc_d3 > 0.35
        )
        SELECT DISTINCT serialNumber FROM RankedData WHERE rn = 1;
    """
    return execute_query(query)

# -------------------------------------------------------------------------
def _load_info_fallback() -> pd.DataFrame:
    """Fallback static equipment info if dim_equipment is unavailable."""
    return pd.DataFrame(
        {
            "serialNumber": [
                189315064,  # Latitude Scrap Fan Electric Motor - DE
                189315024,  # Hogger Fan Electric Motor - DE
                189315097,  # Latitude Main Electric Motor - DE
                189314686,  # Latitude Scrap Fan Pulley - DE
            ],
            "AssetName": [
                "Latitude Scrap Fan Electric Motor - DE",
                "Hogger Fan Electric Motor - DE",
                "Latitude Main Electric Motor - DE",
                "Latitude Scrap Fan Pulley - DE",
            ],
            "MClass": [2, 3, 2, 2],
            "RPMHZ": [np.nan, np.nan, np.nan, np.nan],
            "PowerHP": [60, 40, 200, 40],
            "NominalRPM": [1780, 1780, 1780, 1780],
            "EquipType": [
                "ElectricMotor",
                "ElectricMotor",
                "ElectricMotor",
                "Pulley",
            ],
            "DrivenBySN": [None, 189315064, 189315064, 189315064],
            "SpeedRatio": [None, 2.5, 3.0, 3.0],
        }
    )


def _load_info_from_json() -> pd.DataFrame | None:
    """Load equipment metadata from a local JSON list (records)."""
    if not os.path.exists(LOCAL_INFO_JSON):
        return None
    try:
        return pd.read_json(LOCAL_INFO_JSON)
    except Exception:
        return None


def _load_bearing_rms_from_json() -> pd.DataFrame | None:
    """Load bearing RMS threshold curve from a local JSON list (records)."""
    if not os.path.exists(BEARING_RMS_JSON):
        return None
    try:
        with open(BEARING_RMS_JSON, "r", encoding="utf-8") as f:
            data = json.load(f)
        return pd.DataFrame(data)
    except Exception:
        return None


def load_info_dat_from_db() -> pd.DataFrame:
    """
    Load equipment metadata from dim_equipment in Databricks if available.
    """
    def _seed_defaults(cur, table_name: str) -> None:
        """
        Seed dim_equipment with the built-in fallback rows if the table is empty.
        This migrates the historical inline metadata into the DB.
        """
        fallback = _load_info_fallback()
        if fallback.empty:
            return

        for _, row in fallback.iterrows():
            cur.execute(
                """
                INSERT INTO {table} (
                  serialNumber,
                  mclass,
                  rpm_hz_cached,
                  power_hp,
                  nominal_rpm,
                  equip_type,
                  driven_by_sn,
                  speed_ratio
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """.format(
                    table=table_name
                ),
                (
                    int(row["serialNumber"]),
                    int(row["MClass"]),
                    None if pd.isna(row["RPMHZ"]) else float(row["RPMHZ"]),
                    None if pd.isna(row["PowerHP"]) else float(row["PowerHP"]),
                    None if pd.isna(row["NominalRPM"]) else float(row["NominalRPM"]),
                    str(row["EquipType"]),
                    None if pd.isna(row["DrivenBySN"]) else int(row["DrivenBySN"]),
                    None if pd.isna(row["SpeedRatio"]) else float(row["SpeedRatio"]),
                ),
            )
        cur.connection.commit()

    try:
        def _query(table_name: str):
            return f"""
            SELECT
              serialNumber,
              mclass           AS MClass,
              rpm_hz_cached    AS RPMHZ,
              power_hp         AS PowerHP,
              power_kw         AS PowerKW,
              nominal_rpm      AS NominalRPM,
              equip_type       AS EquipType,
              asset_name       AS AssetName,
              location         AS Location,
              driven_by_sn     AS DrivenBySN,
              speed_ratio      AS SpeedRatio,
              num_blades       AS NumBlades,
              num_teeth        AS NumTeeth,
              severity_scale   AS SeverityScale,
              custom_vel_yellow AS CustomVelYellow,
              custom_vel_red    AS CustomVelRed,
              sensor_vertical_channel   AS SensorVerticalChannel,
              sensor_horizontal_channel AS SensorHorizontalChannel,
              sensor_axial_channel      AS SensorAxialChannel,
              criticality       AS Criticality
            FROM {table_name}
        """
        conn = create_connection()
        cur = conn.cursor()
        df = None
        try:
            for table_name in (DEFAULT_DIM_TABLE, LEGACY_DIM_TABLE):
                try:
                    cur.execute(_query(table_name))
                    data = cur.fetchall()
                    cols = [d[0] for d in cur.description]
                    df = pd.DataFrame(data, columns=cols)

                    # If the table exists but is empty, seed with defaults and re-query.
                    if df.empty:
                        _seed_defaults(cur, table_name)
                        cur.execute(_query(table_name))
                        data = cur.fetchall()
                        cols = [d[0] for d in cur.description]
                        df = pd.DataFrame(data, columns=cols)

                    # Successful read
                    if df is not None and not df.empty:
                        break
                except Exception:
                    df = None

        finally:
            cur.close()
            conn.close()

        if df is None or df.empty:
            return _load_info_fallback()

        # Ensure expected columns exist
        for col in [
            "RPMHZ",
            "PowerKW",
            "NumBlades",
            "NumTeeth",
            "SeverityScale",
            "CustomVelYellow",
            "CustomVelRed",
            "DrivenBySN",
            "SpeedRatio",
            "AssetName",
            "Location",
            "SensorVerticalChannel",
            "SensorHorizontalChannel",
            "SensorAxialChannel",
            "Criticality",
        ]:
            if col not in df.columns:
                df[col] = np.nan
        return df
    except Exception:
        return _load_info_fallback()


def load_info_dat() -> pd.DataFrame:
    """
    Unified loader for equipment metadata.
    Priority:
      1) Local JSON (backend/app/data/info_dat.json) if present.
      2) Databricks dim_equipment (default or legacy).
      3) Built-in fallback.
    """
    json_df = _load_info_from_json()
    if json_df is not None and not json_df.empty:
        return json_df

    db_df = load_info_dat_from_db()
    if db_df is not None and not db_df.empty:
        return db_df

    return _load_info_fallback()


# Load equipment metadata (local CSV > DB > fallback)
info_dat = load_info_dat()

# Human-friendly names for each sensor
sensor_naming_map = {
    189315024: "Hogger Fan Electric Motor",
    189315064: "Latitude Scrap Fan Electric Motor",
    189315097: "Latitude Main Electric Motor",
    189314686: "Latitude Scrap Fan Pulley",
}

# -------------------------------------------------------------------------
# ISO 10816/20816 severity tables (RMS velocity & peak velocity)
# -------------------------------------------------------------------------
mechanical_fault_rms = pd.DataFrame(
    {
        "group": [1, 2, 3, 4] * 4,
        "lower_l": [
            0,
            0,
            0,
            0,
            1.12,
            1.80,
            2.80,
            4.50,
            2.8,
            4.5,
            7.1,
            11.2,
            7.1,
            11.2,
            18.0,
            28.0,
        ],
        "upper_l": [
            1.12,
            1.80,
            2.80,
            4.50,
            2.8,
            4.5,
            7.1,
            11.2,
            7.1,
            11.2,
            18.0,
            28.0,
            999999,
            999999,
            999999,
            999999,
        ],
        "cond": (
            ["Good"] * 4
            + ["Satisfactory"] * 4
            + ["Unsatisfactory"] * 4
            + ["Unacceptable"] * 4
        ),
        "color": (["green"] * 8 + ["yellow"] * 4 + ["red"] * 4),
    }
)

# Peak velocity (in/s) derived from ISO mm/s RMS
mechanical_fault_peak = pd.DataFrame(
    {
        "group": [1, 2, 3, 4] * 4,
        "lower_l": [
            0,
            0,
            0,
            0,
            0.06,
            0.10,
            0.16,
            0.25,
            0.16,
            0.25,
            0.40,
            0.62,
            0.40,
            0.62,
            1.00,
            1.56,
        ],
        "upper_l": [
            0.06,
            0.10,
            0.16,
            0.25,
            0.16,
            0.25,
            0.40,
            0.62,
            0.40,
            0.62,
            1.00,
            1.56,
            999999,
            999999,
            999999,
            999999,
        ],
        "cond": (
            ["Good"] * 4
            + ["Satisfactory"] * 4
            + ["Unsatisfactory"] * 4
            + ["Unacceptable"] * 4
        ),
        "color": (["green"] * 8 + ["yellow"] * 4 + ["red"] * 4),
    }
)

# -------------------------------------------------------------------------
# Group inference based on RPM & power
# -------------------------------------------------------------------------
def infer_mechanical_group(
    info_row: pd.Series, rpm_hz: float | None, fallback_group: int = 2
) -> int:
    """
    Auto-select ISO-like machine group based on power and speed.

    Logic:
      - If low speed (< 600 RPM), map to group 1.
      - Otherwise, use power (kW) to choose group 1–4.
      - If power is missing, fall back to MClass.
    """
    base_group = int(info_row.get("MClass", fallback_group))

    rpm_hz = float(rpm_hz) if rpm_hz is not None and np.isfinite(rpm_hz) else np.nan
    rpm = rpm_hz * 60.0 if np.isfinite(rpm_hz) else np.nan

    power_kw = np.nan
    if "PowerKW" in info_row.index and not pd.isna(info_row["PowerKW"]):
        power_kw = float(info_row["PowerKW"])
    elif "PowerHP" in info_row.index and not pd.isna(info_row["PowerHP"]):
        power_kw = float(info_row["PowerHP"]) * 0.746  # HP → kW

    if np.isfinite(rpm) and rpm < 600:
        return 1

    if not np.isfinite(power_kw):
        return base_group

    if power_kw <= 15:
        return 1
    elif power_kw <= 75:
        return 2
    elif power_kw <= 300:
        return 3
    else:
        return 4

# -------------------------------------------------------------------------
# -------------------------------------------------------------------------
# Bearing RMS threshold curve
#   - Interpolates “beta” vs RPM for bearing severity thresholds.
#   - Can be overridden by backend/app/data/bearing_fault_rms.json
# -------------------------------------------------------------------------
_bearing_rms_fallback = pd.DataFrame(
    {
        "velocity": [200, 600, 1200, 1800, 3600],  # RPM
        "beta": [0.2, 0.6, 0.95, 1.3, 2.5],        # baseline RMS acc thresholds
    }
)
_bearing_rms_loaded = _load_bearing_rms_from_json()
bearing_fault_rms = (
    _bearing_rms_loaded
    if _bearing_rms_loaded is not None and not _bearing_rms_loaded.empty
    else _bearing_rms_fallback
)
get_bearing_fault_rms = interp1d(
    bearing_fault_rms["velocity"], bearing_fault_rms["beta"], kind="linear"
)

# -------------------------------------------------------------------------
# Type-specific fault classification (gearbox, motor, pump, fan, pulley)
# -------------------------------------------------------------------------
def classify_type_specific_faults(equip_type: str | None, metrics: dict) -> list[dict]:
    """
    Return a list of type-specific fault tags based on equipment type
    and already-computed metrics (overall RMS, 1× amplitude, ISO severity).
    """

    equip_type = (equip_type or "").lower()
    results: list[dict] = []

    rms_vel_max = metrics.get("rms_vel_max", None)  # mm/s
    x1amp = metrics.get("x1amp", None)             # in/s peak
    mc_fault_rms_res = metrics.get("mc_fault_rms_res")
    mc_fault_peak_res = metrics.get("mc_fault_peak_res")

    def _worst_cond(fault_df):
        """
        Extract worst ISO condition from a DataFrame:
        Good < Satisfactory < Unsatisfactory < Unacceptable
        """
        if fault_df is None or getattr(fault_df, "empty", True):
            return None
        conds = list(fault_df["cond"].astype(str))
        order = ["Good", "Satisfactory", "Unsatisfactory", "Unacceptable"]
        conds_sorted = sorted(
            conds, key=lambda c: order.index(c) if c in order else -1
        )
        return conds_sorted[-1] if conds_sorted else None

    worst_rms_cond = _worst_cond(mc_fault_rms_res)
    worst_peak_cond = _worst_cond(mc_fault_peak_res)

    # ----------------------
    # Gearbox → gear issues
    # ----------------------
    if "gear" in equip_type:
        if worst_rms_cond in ("Unsatisfactory", "Unacceptable") or \
           worst_peak_cond in ("Unsatisfactory", "Unacceptable"):
            results.append(
                {
                    "code": "gear_issue",
                    "severity": "alarm"
                    if worst_peak_cond == "Unacceptable"
                    else "warning",
                    "reason": (
                        f"High vibration severity for {equip_type} "
                        f"(cond_rms={worst_rms_cond}, cond_peak={worst_peak_cond})."
                    ),
                }
            )
        gms = metrics.get("gear_mesh_score")
        if gms is not None:
            if gms > 0.3:
                severity = "warning" if gms < 0.6 else "alarm"
                results.append(
                    {
                        "code": "gear_mesh_modulation",
                        "severity": severity,
                        "reason": (
                            f"High gear mesh energy (score={gms:.2f}) relative to band "
                            "– possible gear wear or tooth damage."
                        ),
                    }
                )

    # ----------------------
    # Electric motor
    # ----------------------
    if "motor" in equip_type:
        if rms_vel_max is not None and x1amp is not None and \
           rms_vel_max > 7.1 and x1amp > 0.4:  # ~ISO zone C
            results.append(
                {
                    "code": "motor_mech_issue",
                    "severity": "alarm",
                    "reason": (
                        "High 1× and overall velocity for motor "
                        "(possible unbalance/misalignment)."
                    ),
                }
            )

    # ----------------------
    # Pump / Compressor
    # ----------------------
    if "pump" in equip_type or "compressor" in equip_type:
        if worst_rms_cond in ("Unsatisfactory", "Unacceptable"):
            results.append(
                {
                    "code": "process_or_mech_issue",
                    "severity": "warning"
                    if worst_rms_cond == "Unsatisfactory"
                    else "alarm",
                    "reason": (
                        f"Elevated overall vibration for {equip_type} "
                        f"(cond_rms={worst_rms_cond})."
                    ),
                }
            )

    # ----------------------
    # Fan / Blower
    # ----------------------
    if "fan" in equip_type or "blower" in equip_type:
        if rms_vel_max is not None and rms_vel_max > 4.5:  # mm/s
            results.append(
                {
                    "code": "fan_unbalance_or_looseness",
                    "severity": "warning" if rms_vel_max < 7.1 else "alarm",
                    "reason": (
                        f"Overall velocity {rms_vel_max:.2f} mm/s suggests "
                        "unbalance or looseness."
                    ),
                }
            )
        fan_bpf_amp = metrics.get("fan_bpf_amp")
        if fan_bpf_amp is not None and rms_vel_max is not None:
            ratio = fan_bpf_amp / max(rms_vel_max, 1e-6)
            if ratio > 2.0 and rms_vel_max > 4.5:
                severity = "warning" if ratio < 4.0 else "alarm"
                results.append(
                    {
                        "code": "fan_blade_damage",
                        "severity": severity,
                        "reason": (
                            f"High blade-pass frequency component "
                            f"(BPF amplitude ≈ {fan_bpf_amp:.2f} mm/s) relative to overall RMS "
                            f"({rms_vel_max:.2f} mm/s) – possible blade damage or build-up."
                        ),
                    }
                )

    # ----------------------
    # Pulley / Belt drive
    # ----------------------
    if "pulley" in equip_type or "sheave" in equip_type or "belt" in equip_type:
        if rms_vel_max is not None and x1amp is not None and rms_vel_max > 4.5:
            severity = "warning" if rms_vel_max < 7.1 else "alarm"
            results.append(
                {
                    "code": "pulley_belt_issue",
                    "severity": severity,
                    "reason": (
                        f"Elevated overall vibration ({rms_vel_max:.2f} mm/s) with strong 1× "
                        f"component ({x1amp:.3f} in/s) for pulley/belt drive – possible unbalance, "
                        "misalignment, belt wear, or looseness."
                    ),
                }
            )
        belt_slip_ratio = metrics.get("belt_slip_ratio")
        if belt_slip_ratio is not None and np.isfinite(belt_slip_ratio):
            slip_pct = abs(belt_slip_ratio) * 100.0
            if slip_pct > 2.0:
                severity = "warning" if slip_pct < 5.0 else "alarm"
                results.append(
                    {
                        "code": "belt_slip",
                        "severity": severity,
                        "reason": (
                            f"Measured pulley speed deviates {slip_pct:.1f}% from expected "
                            f"(possible belt slip or ratio mismatch)."
                        ),
                    }
                )

    # ----------------------
    # Low-frequency looseness (any mechanical)
    # ----------------------
    lf_rms = metrics.get("low_freq_rms")
    if lf_rms is not None and rms_vel_max is not None:
        lf_ratio = lf_rms / max(rms_vel_max, 1e-9)
        if lf_ratio > 0.5 and rms_vel_max > 4.5:
            severity = "warning" if lf_ratio < 1.0 else "alarm"
            results.append(
                {
                    "code": "low_freq_looseness",
                    "severity": severity,
                    "reason": (
                        f"Low-frequency vibration dominates overall RMS "
                        f"(LF ratio={lf_ratio:.2f}, LF RMS={lf_rms:.2f} mm/s) – "
                        "possible structural looseness or soft foot."
                    ),
                }
            )

    # NOTE: Envelope demod (BPFO/BPFI/BSF/FTF) can be added later once BPF values
    # are available in info_dat. For now, we keep the schema ready but do not
    # generate such faults (metrics would be None).
    return results

# -------------------------------------------------------------------------
# Fault logic at harmonic levels (unbalance, looseness, misalignment, bearing)
# -------------------------------------------------------------------------
def check_unbalance(x1, x2, xyellow, xred):
    """
    Unbalance:
      - x2 must be small (< 0.15 × x1)
      - Severity escalates with x1 vs xyellow / xred
    """
    if x2 <= (0.15 * x1) and x1 < xyellow:
        return "u1"
    elif x2 <= (0.15 * x1) and xyellow <= x1 < xred:
        return "u2"
    elif x2 <= (0.15 * x1) and x1 >= xred:
        return "u3"
    else:
        return "u0"


def check_loosenes(x1, x2, x3, x4, xyellow, xred):
    """
    Looseness:
      - x2, x3, x4 all relatively large (> 0.3 × x1)
      - Severity escalates with x1 vs xyellow / xred
    """
    cond = (x2 > 0.3 * x1) and (x3 > 0.3 * x1) and (x4 > 0.3 * x1)
    if cond and x1 < xyellow:
        return "l1"
    elif cond and (xyellow <= x1 < xred):
        return "l2"
    elif cond and x1 >= xred:
        return "l3"
    else:
        return "l0"


def check_misalignment(x1, x2, xyellow, xred):
    """
    Misalignment:
      - x2 relatively large vs x1 (>= 0.3–0.5 × x1)
      - Severity based on x1 vs xyellow / xred
    """
    if x2 >= (0.50 * x1) and x1 < xyellow:
        return "m1"
    elif x2 >= (0.30 * x1) and (xyellow <= x1 < xred):
        return "m2"
    elif x2 >= (0.30 * x1) and x1 >= xred:
        return "m3"
    else:
        return "m0"


def check_bearing_rms(rpm, rms_acc):
    """
    Bearing RMS check:
      - Uses speed-dependent beta(rpm) threshold.
      - brms1 = ok, brms2 = warning, brms3 = alarm.
    """
    beta = float(get_bearing_fault_rms([float(rpm)])[0])
    if rms_acc < beta * 0.5:
        return "brms1"
    elif rms_acc < beta:
        return "brms2"
    else:
        return "brms3"


# -------------------------------------------------------------------------
# RPM / 1× estimation (Quinn2 + SNR + EMA + order locking)
# -------------------------------------------------------------------------
_RPM_EMA: dict[int, dict[str, float]] = {}

def _load_app_config() -> dict:
    if not os.path.exists(APP_CONFIG_JSON):
        return {}
    try:
        with open(APP_CONFIG_JSON, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


_APP_CFG = _load_app_config()


def _cfg(section: str, key: str, default):
    try:
        return _APP_CFG.get(section, {}).get(key, default)
    except Exception:
        return default


EMA_ALPHA = float(_cfg("vibextractor", "EMA_ALPHA", float(os.getenv("HPS_RPM_EMA_ALPHA", "0.30"))))
SNR_WEAK = float(_cfg("vibextractor", "SNR_WEAK", float(os.getenv("HPS_RPM_SNR_WEAK", "3.0"))))
SNR_STRONG = float(_cfg("vibextractor", "SNR_STRONG", float(os.getenv("HPS_RPM_SNR_STRONG", "6.0"))))
DEFAULT_SEARCH = (
    float(_cfg("vibextractor", "DEFAULT_SEARCH_MIN_HZ", float(os.getenv("HPS_RPM_MIN_HZ", "15.0")))),
    float(_cfg("vibextractor", "DEFAULT_SEARCH_MAX_HZ", float(os.getenv("HPS_RPM_MAX_HZ", "120.0")))),
)
ORDER_LOCK_SPAN_HZ = float(_cfg("vibextractor", "ORDER_LOCK_SPAN_HZ", float(os.getenv("HPS_ORDER_LOCK_SPAN_HZ", "0.6"))))
ORDER_LOCK_STEP_HZ = float(_cfg("vibextractor", "ORDER_LOCK_STEP_HZ", float(os.getenv("HPS_ORDER_LOCK_STEP_HZ", "0.02"))))
ORDER_LOCK_MAX_HARM = int(_cfg("vibextractor", "ORDER_LOCK_MAX_HARM", int(os.getenv("HPS_ORDER_LOCK_MAX_HARM", "4"))))

# UI/test scale multiplier to quickly boost/attenuate amplitudes (for UI testing only)
# UI scaling disabled (fixed at 1.0 to avoid distortion)
UI_SCALE = 1.0


def _parabolic_interpolate(y: np.ndarray, i: int):
    if i <= 0 or i >= len(y) - 1:
        return 0.0, float(y[i])
    y0, y1, y2 = float(y[i - 1]), float(y[i]), float(y[i + 1])
    denom = y0 - 2.0 * y1 + y2
    if denom == 0.0:
        return 0.0, y1
    x_offset = 0.5 * (y0 - y2) / denom
    y_peak = y1 - 0.25 * (y0 - y2) * x_offset
    return float(x_offset), float(y_peak)


def _quinn2_offset(y: np.ndarray, k: int) -> float:
    n = len(y)
    if k <= 0 or k >= n - 1:
        return 0.0
    ykm1, yk, ykp1 = float(y[k - 1]), float(y[k]), float(y[k + 1])
    if yk <= 0 or (ykm1 <= 0) or (ykp1 <= 0):
        return 0.0

    am = ykm1 / yk
    ap = ykp1 / yk

    def _tau(u: float) -> float:
        if abs(u) < 1e-6:
            return 1.0 + u / 2.0 + (u * u) / 12.0
        denom = 1.0 - np.exp(-u)
        if denom == 0.0:
            return 1.0
        return u / denom

    um = np.log(am) if am > 0 else -np.inf
    up = np.log(ap) if ap > 0 else -np.inf

    if not np.isfinite(um) or not np.isfinite(up):
        return _parabolic_interpolate(y, k)[0]

    taum = _tau(um)
    taup = _tau(up)

    dm = am / (1.0 - am) if am != 1.0 else 0.0
    dp = ap / (1.0 - ap) if ap != 1.0 else 0.0

    num = (dp * taup) - (dm * taum)
    den = 1.0 + (dp * taup) * (dm * taum)
    if den == 0.0:
        return _parabolic_interpolate(y, k)[0]
    d = num / den
    return float(np.clip(d, -1.0, 1.0))


def _peak_refine_quinn2(spec: np.ndarray, k: int) -> float:
    try:
        d = _quinn2_offset(spec, k)
        if not np.isfinite(d) or abs(d) > 1.0:
            raise ValueError
        return float(d)
    except Exception:
        return _parabolic_interpolate(spec, k)[0]


def _band_snr(spec: np.ndarray) -> float:
    peak = float(np.max(spec))
    med = float(np.median(spec) + 1e-12)
    return peak / med


def _order_lock_refine(
    freqs: np.ndarray,
    spec: np.ndarray,
    rpm0_hz: float,
    max_harm: int = ORDER_LOCK_MAX_HARM,
    span_hz: float = ORDER_LOCK_SPAN_HZ,
    step_hz: float = ORDER_LOCK_STEP_HZ,
) -> float:
    if len(freqs) < 3 or rpm0_hz <= 0:
        return rpm0_hz

    df = (freqs[1] - freqs[0]) if len(freqs) >= 2 else 0.0
    if df <= 0:
        return rpm0_hz

    candidates = np.arange(
        rpm0_hz - span_hz, rpm0_hz + span_hz + 1e-9, step_hz
    )
    best_val, best_rpm = -np.inf, rpm0_hz

    for r in candidates:
        if r <= 0:
            continue
        score = 0.0
        for n in range(1, max_harm + 1):
            f = n * r
            k = int(np.argmin(np.abs(freqs - f)))
            k0 = max(0, k - 1)
            k1 = min(len(spec), k + 2)
            win = spec[k0:k1]
            w = 1.0 / n
            score += w * float(np.max(win) if win.size else 0.0)
        if score > best_val:
            best_val, best_rpm = score, r

    return float(best_rpm)


def _ema_update(sn: int, rpm_hz: float) -> float:
    if not np.isfinite(rpm_hz) or rpm_hz <= 0:
        prev = _RPM_EMA.get(sn, {}).get("rpm_hz", np.nan)
        return float(prev) if np.isfinite(prev) else float("nan")

    prev = _RPM_EMA.get(sn, {}).get("rpm_hz", None)
    if prev is None or not np.isfinite(prev):
        smoothed = float(rpm_hz)
    else:
        smoothed = float(EMA_ALPHA * rpm_hz + (1.0 - EMA_ALPHA) * prev)

    _RPM_EMA[sn] = {"rpm_hz": smoothed}
    return smoothed


def estimate_rpm_hz_from_fft_row(
    mydata_row: pd.Series, search_band=(5.0, 60.0)
) -> dict:
    """
    Estimate running speed (1×) in Hz from a single FFT row.
    """
    freqs = np.asarray(mydata_row["frequencies"], dtype=float)
    d1 = np.asarray(mydata_row["fft_vel_d1"], dtype=float)
    d2 = np.asarray(mydata_row["fft_vel_d2"], dtype=float)
    d3 = np.asarray(mydata_row["fft_vel_d3"], dtype=float)

    comp = np.maximum.reduce([d1, d2, d3])

    fmin, fmax = float(search_band[0]), float(search_band[1])
    if fmin >= fmax:
        fmin, fmax = DEFAULT_SEARCH

    mask = (freqs >= fmin) & (freqs <= fmax)
    if not np.any(mask):
        mask = np.ones_like(freqs, dtype=bool)

    comp_band = comp[mask]
    freqs_band = freqs[mask]

    snr = _band_snr(comp_band)

    local_idx = int(np.argmax(comp_band))
    k = np.flatnonzero(mask)[local_idx]

    frac = _peak_refine_quinn2(comp, k)

    dfreq = (freqs[1] - freqs[0]) if len(freqs) >= 2 else 0.0
    rpm_hz = float(freqs[k] + frac * dfreq)

    amp_at_bin = np.array([d1[k], d2[k], d3[k]])
    max_direction = int(np.argmax(amp_at_bin)) + 1

    return {"rpm_hz": rpm_hz, "max_direction": max_direction, "conf": float(snr)}


def ensure_rpmhz_column(df_row: pd.Series, info_row: pd.Series) -> float:
    """
    Backward compatibility helper:

    - If info_dat has a valid RPMHZ, use it.
    - Otherwise estimate RPM from FFT and write it back into info_dat.RPMHZ.
    - Return RPM in Hz.
    """
    # Only trust cached RPM if it is finite and within our default search band
    cached_rpm_hz = None
    if ("RPMHZ" in info_row.index) and pd.notna(info_row["RPMHZ"]):
        try:
            cached_rpm_hz = float(info_row["RPMHZ"])
        except Exception:
            cached_rpm_hz = None

    if cached_rpm_hz is not None and np.isfinite(cached_rpm_hz):
        if cached_rpm_hz >= DEFAULT_SEARCH[0]:
            return float(cached_rpm_hz)

    est = estimate_rpm_hz_from_fft_row(df_row, search_band=DEFAULT_SEARCH)
    rpm_hz = float(est["rpm_hz"])
    try:
        idx = info_dat.index[info_dat["serialNumber"] == info_row["serialNumber"]][0]
        info_dat.loc[idx, "RPMHZ"] = rpm_hz
    except Exception:
        pass
    return rpm_hz

# -------------------------------------------------------------------------
# Metrics / Fault detection for a single asset
# -------------------------------------------------------------------------
def get_metrics(
    df: pd.DataFrame,
    mySN: int,
    info_dat_in: pd.DataFrame = info_dat,
    mechanical_fault_rms_in: pd.DataFrame = mechanical_fault_rms,
    mechanical_fault_peak_in: pd.DataFrame = mechanical_fault_peak,
) -> dict:
    """
    Core entry point per equipment.

    Steps:
      - Estimate RPM (Hz + RPM).
      - Infer ISO group + thresholds.
      - Compute overall RMS metrics.
      - Extract harmonics (1×..4×) per direction.
      - Apply unbalance/looseness/misalignment/bearing rules.
      - Map overall RMS & 1× to ISO severity.
      - Generate type-specific fault hints (gearbox, motor, pump, fan, pulley).
    """
    myinfo = info_dat_in[info_dat_in.serialNumber == mySN]
    mydata = df[df.serialNumber == mySN]

    if mydata.empty:
        return {"serialNumber": int(mySN), "error": "No data for this serialNumber"}

    mydata_row = mydata.iloc[0]
    base_group = int(myinfo.MClass.values[0]) if not myinfo.empty else 1

    # --- RPM estimation ---
    info_row = (
        myinfo.iloc[0]
        if not myinfo.empty
        else pd.Series({"serialNumber": mySN, "RPMHZ": np.nan})
    )
    # Channel mapping (defaults: d1=Vertical, d2=Horizontal, d3=Axial)
    try:
        vert_ch = int(info_row.get("SensorVerticalChannel") or 1)
    except Exception:
        vert_ch = 1
    try:
        horiz_ch = int(info_row.get("SensorHorizontalChannel") or 2)
    except Exception:
        horiz_ch = 2
    try:
        axial_ch = int(info_row.get("SensorAxialChannel") or 3)
    except Exception:
        axial_ch = 3
    dir_channels = [vert_ch, horiz_ch, axial_ch]  # index 0->dir1(V),1->H,2->Ax
    rpm_guess_hz = ensure_rpmhz_column(mydata_row, info_row)
    if not np.isfinite(rpm_guess_hz) or rpm_guess_hz <= 0:
        rpm_guess_hz = DEFAULT_SEARCH[0]

    # Refine only within the default band; do not dip below DEFAULT_SEARCH[0]
    search_low = max(DEFAULT_SEARCH[0], rpm_guess_hz - 20.0)
    search_high = min(DEFAULT_SEARCH[1], rpm_guess_hz + 20.0)
    if not np.isfinite(search_low) or not np.isfinite(search_high) or search_high <= search_low:
        search_low, search_high = DEFAULT_SEARCH

    est = estimate_rpm_hz_from_fft_row(
        mydata_row,
        search_band=(search_low, search_high),
    )
    rpm_est_hz = float(est["rpm_hz"])
    max_direction_raw = int(est["max_direction"])
    # Map raw channel -> canonical direction index (1=Vertical,2=Horizontal,3=Axial)
    raw_to_dir = {vert_ch: 1, horiz_ch: 2, axial_ch: 3}
    max_direction = raw_to_dir.get(max_direction_raw, max_direction_raw)
    conf = float(est["conf"])

    if conf < SNR_WEAK:
        prev = _RPM_EMA.get(int(mySN), {}).get("rpm_hz", np.nan)
        # Only reuse previous EMA if it is within the configured search band
        if np.isfinite(prev) and prev >= DEFAULT_SEARCH[0]:
            rpm_est_hz = float(prev)
    # Guard against sub-band estimates
    if rpm_est_hz < DEFAULT_SEARCH[0]:
        rpm_est_hz = search_low

    freqs = np.asarray(mydata_row["frequencies"], dtype=float)
    spec_dir = np.asarray(mydata_row[f"fft_vel_d{max_direction_raw}"], dtype=float)

    rpm_locked_hz = _order_lock_refine(freqs, spec_dir, rpm_est_hz)
    if rpm_locked_hz < DEFAULT_SEARCH[0]:
        rpm_locked_hz = rpm_est_hz

    dfreq = (freqs[1] - freqs[0]) if len(freqs) >= 2 else 0.0
    mask1 = (freqs >= (rpm_locked_hz - 1.0)) & (freqs <= (rpm_locked_hz + 1.0))
    if np.any(mask1):
        local = spec_dir[mask1]
        idx_local = int(np.argmax(local))
        k = np.flatnonzero(mask1)[idx_local]
        frac = _peak_refine_quinn2(spec_dir, k)
        myrpm_hz = float(freqs[k] + frac * dfreq)
    else:
        myrpm_hz = float(rpm_locked_hz)

    myrpm_hz = _ema_update(int(mySN), myrpm_hz)
    if (not np.isfinite(myrpm_hz)) or myrpm_hz <= 0 or myrpm_hz < DEFAULT_SEARCH[0]:
        # Fallback to the initial estimate within the default band
        myrpm_hz = float(rpm_guess_hz)

    # --- Belt slip ratio (pulley/belt assets) ---
    belt_slip_ratio = None
    if "EquipType" in info_row.index and "SpeedRatio" in info_row.index:
        etype = str(info_row["EquipType"]).lower()
        if etype in ("pulley", "belt", "belt drive", "belt_drive"):
            design_ratio = float(info_row.get("SpeedRatio") or 0.0)
            if design_ratio > 0 and "DrivenBySN" in info_row.index and not pd.isna(
                info_row["DrivenBySN"]
            ):
                driver_sn = int(info_row["DrivenBySN"])
                prev = _RPM_EMA.get(driver_sn, {})
                driver_rpm_hz = float(prev.get("rpm_hz") or 0.0)
                if driver_rpm_hz > 0:
                    expected_pulley_hz = driver_rpm_hz / design_ratio
                    belt_slip_ratio = (myrpm_hz - expected_pulley_hz) / expected_pulley_hz

    # --- ISO group + thresholds ---
    if not myinfo.empty:
        info_row = myinfo.iloc[0]
    else:
        info_row = pd.Series({"MClass": base_group})

    mygroup = infer_mechanical_group(info_row, myrpm_hz, fallback_group=base_group)

    xyellow = float(
        mechanical_fault_peak_in[
            (mechanical_fault_peak_in.color == "yellow")
            & (mechanical_fault_peak_in.group == mygroup)
        ].lower_l.min()
    )
    xred = float(
        mechanical_fault_peak_in[
            (mechanical_fault_peak_in.color == "red")
            & (mechanical_fault_peak_in.group == mygroup)
        ].lower_l.min()
    )

    # --- RMS metrics (velocity mm/s, acceleration native units) with channel mapping ---
    rms_vels_raw = np.array(
        [
            float(mydata_row["rms_vel_fft1000_d1"]),
            float(mydata_row["rms_vel_fft1000_d2"]),
            float(mydata_row["rms_vel_fft1000_d3"]),
        ],
        dtype=float,
    )
    rms_vels_raw *= UI_SCALE
    rms_vels = np.array(
        [rms_vels_raw[ch - 1] if 1 <= ch <= 3 else np.nan for ch in dir_channels],
        dtype=float,
    )
    rms_vel_max = float(np.nanmax(rms_vels))

    rms_accs_raw = np.array(
        [
            float(mydata_row["rms_acc_fft1000_d1"]),
            float(mydata_row["rms_acc_fft1000_d2"]),
            float(mydata_row["rms_acc_fft1000_d3"]),
        ],
        dtype=float,
    )
    rms_accs_raw *= UI_SCALE
    rms_accs = np.array(
        [rms_accs_raw[ch - 1] if 1 <= ch <= 3 else np.nan for ch in dir_channels],
        dtype=float,
    )
    rms_acc_max = float(np.nanmax(rms_accs))

    rms_accs_full_raw = np.array(
        [
            float(mydata_row["rms_acc_fft_d1"]),
            float(mydata_row["rms_acc_fft_d2"]),
            float(mydata_row["rms_acc_fft_d3"]),
        ],
        dtype=float,
    )
    rms_accs_full_raw *= UI_SCALE
    rms_accs_full = np.array(
        [rms_accs_full_raw[ch - 1] if 1 <= ch <= 3 else np.nan for ch in dir_channels],
        dtype=float,
    )
    rms_acc_full_max = float(np.nanmax(rms_accs_full))

    mc_fault_rms_res = mechanical_fault_rms_in[
        (mechanical_fault_rms_in.group == mygroup)
        & (mechanical_fault_rms_in.lower_l <= rms_vel_max)
        & (mechanical_fault_rms_in.upper_l >= rms_vel_max)
    ]

    # --- Harmonics (1×..4×) for each direction ---
    dat_harmonics = pd.DataFrame(
        columns=["direction", "harmonic", "ix", "amp", "freq"]
    )
    for direction in (1, 2, 3):
        ch = dir_channels[direction - 1]
        if ch < 1 or ch > 3:
            continue
        spdir = np.asarray(mydata_row[f"fft_vel_d{ch}"], dtype=float) * UI_SCALE
        for i in (1, 2, 3, 4):
            target_hz = myrpm_hz * i
            maski = (freqs >= (target_hz - 1.0)) & (freqs <= (target_hz + 1.0))
            if not np.any(maski):
                continue
            local = spdir[maski]
            ixx_local = int(np.argmax(local))
            ixx_global = np.flatnonzero(maski)[ixx_local]
            frac_i = _peak_refine_quinn2(spdir, ixx_global)
            df_ = (freqs[1] - freqs[0]) if len(freqs) >= 2 else 0.0
            freq_i = float(freqs[ixx_global] + frac_i * df_)
            amp_i = float(spdir[ixx_global])

            dat_harmonics = pd.concat(
                [
                    dat_harmonics,
                    pd.DataFrame(
                        {
                            "direction": [int(direction)],
                            "harmonic": [int(i)],
                            "ix": [int(ixx_global)],
                            "amp": [float(amp_i)],
                            "freq": [float(freq_i)],
                        }
                    ),
                ],
                ignore_index=True,
            )

    # --- Low-frequency looseness metric ---
    low_freq_rms = None
    try:
        low_freq_band = (freqs >= 0.5) & (freqs <= 10.0)
        sp_all = np.maximum.reduce(
            [
                np.asarray(mydata_row["fft_vel_d1"], dtype=float),
                np.asarray(mydata_row["fft_vel_d2"], dtype=float),
                np.asarray(mydata_row["fft_vel_d3"], dtype=float),
            ]
        )
        if np.any(low_freq_band):
            low_freq_rms = float(np.sqrt(np.mean(sp_all[low_freq_band] ** 2)))
    except Exception:
        low_freq_rms = None

    # --- Fan blade damage metric (BPF amplitude) [future fan assets] ---
    fan_bpf_amp = None
    if "EquipType" in info_row.index and "NumBlades" in info_row.index:
        if str(info_row["EquipType"]).lower() in ("fan", "blower"):
            try:
                n_blades = int(info_row.get("NumBlades") or 0)
            except Exception:
                n_blades = 0
            if n_blades > 0:
                bpf_hz = myrpm_hz * n_blades
                spdir = np.asarray(mydata_row[f"fft_vel_d{max_direction}"], dtype=float)
                mask = (freqs >= (bpf_hz - 0.5)) & (freqs <= (bpf_hz + 0.5))
                if np.any(mask):
                    fan_bpf_amp = float(np.max(spdir[mask]))

    # --- Gear mesh pattern scoring [future gearbox assets] ---
    gear_mesh_score = None
    if str(info_row.get("EquipType", "")).lower().startswith("gear"):
        try:
            n_teeth = int(info_row.get("NumTeeth") or 0)
        except Exception:
            n_teeth = 0
        if n_teeth > 0 and myrpm_hz > 0:
            gmf_hz = myrpm_hz * n_teeth
            spdir = np.asarray(mydata_row[f"fft_vel_d{max_direction}"], dtype=float)
            band1 = (freqs >= gmf_hz * 0.9) & (freqs <= gmf_hz * 1.1)
            band2 = (freqs >= 2 * gmf_hz * 0.9) & (freqs <= 2 * gmf_hz * 1.1)
            num = np.sum(spdir[band1]) + np.sum(spdir[band2])
            band_all = (freqs >= 0.5 * gmf_hz) & (freqs <= 3.0 * gmf_hz)
            den = np.sum(spdir[band_all]) + 1e-9
            gear_mesh_score = float(num / den)

    # --- Directional faults by harmonic pattern ---
    faults = pd.DataFrame(
        columns=["direction", "unbalance", "loosenes", "misalignment", "bearing_rms"]
    )
    for direction in (1, 2, 3):

        def _h_amp(h):
            vals = dat_harmonics[
                (dat_harmonics.direction == direction)
                & (dat_harmonics.harmonic == h)
            ]["amp"].values
            return float(np.max(vals)) if len(vals) else np.nan

        x1 = _h_amp(1)
        x2 = _h_amp(2)
        x3 = _h_amp(3)
        x4 = _h_amp(4)
        if np.isnan([x1, x2, x3, x4]).any():
            continue

        unbalance = check_unbalance(x1, x2, xyellow, xred)
        loosenes = check_loosenes(x1, x2, x3, x4, xyellow, xred)
        misalignment = check_misalignment(x1, x2, xyellow, xred)
        bearing_rms = check_bearing_rms(myrpm_hz * 60.0, rms_accs[direction - 1])

        faults = pd.concat(
            [
                faults,
                pd.DataFrame(
                    {
                        "direction": [int(direction)],
                        "unbalance": [unbalance],
                        "loosenes": [loosenes],
                        "misalignment": [misalignment],
                        "bearing_rms": [bearing_rms],
                    }
                ),
            ],
            ignore_index=True,
        )

    # Summary fault codes (max across directions)
    if faults.empty:
        fault = pd.DataFrame(
            {
                "unbalance": ["u0"],
                "loosenes": ["l0"],
                "misalignment": ["m0"],
                "bearing_rms": ["brms1"],
            }
        )
    else:
        fault = pd.DataFrame(
            {
                "unbalance": [faults["unbalance"].max()],
                "loosenes": [faults["loosenes"].max()],
                "misalignment": [faults["misalignment"].max()],
                "bearing_rms": [faults["bearing_rms"].max()],
            }
        )

    # 1× amplitude in the direction with max overall vibration
    try:
        x1amp_val = dat_harmonics[
            (dat_harmonics.harmonic == 1.0)
            & (dat_harmonics.direction == max_direction)
        ]["amp"].max()
        x1amp = float(x1amp_val) if pd.notna(x1amp_val) else None
    except Exception:
        x1amp = None

    # ISO-based severity for 1× peak (in/s) in this group
    mc_fault_peak_res = mechanical_fault_peak_in[
        (mechanical_fault_peak_in.group == mygroup)
        & (mechanical_fault_peak_in.lower_l <= (x1amp or 0.0))
        & (mechanical_fault_peak_in.upper_l >= (x1amp or 0.0))
    ]

    # --- Type-specific faults ---
    equip_type = None
    if not myinfo.empty and "EquipType" in myinfo.columns:
        equip_type = str(myinfo["EquipType"].values[0])
    asset_name = None
    if not myinfo.empty and "AssetName" in myinfo.columns:
        try:
            asset_name = str(myinfo["AssetName"].values[0])
        except Exception:
            asset_name = None

    type_specific_faults = classify_type_specific_faults(
        equip_type,
        {
            "rms_vel_max": rms_vel_max,
            "x1amp": x1amp,
            "mc_fault_rms_res": mc_fault_rms_res,
            "mc_fault_peak_res": mc_fault_peak_res,
            "belt_slip_ratio": belt_slip_ratio,
            "fan_bpf_amp": fan_bpf_amp,
            "gear_mesh_score": gear_mesh_score,
            "low_freq_rms": low_freq_rms,
        },
    )

    # --- Return all metrics ---
    return {
        "serialNumber": int(mySN),
        "group": int(mygroup),
        "equip_type": equip_type,
        "asset_name": asset_name,
        # RPM (Hz & RPM)
        "guess_rpm": float(rpm_guess_hz),
        "cal_rpm": float(myrpm_hz),
        "guess_rpm_hz": float(rpm_guess_hz),
        "cal_rpm_hz": float(myrpm_hz),
        "guess_rpm_rpm": float(rpm_guess_hz * 60.0),
        "cal_rpm_rpm": float(myrpm_hz * 60.0),
        # RMS metrics
        "rms_vels": [float(v) for v in rms_vels],
        "rms_accs": [float(a) for a in rms_accs],
        "rms_vel_max": float(rms_vel_max),
        "rms_acc_full_max": float(rms_acc_full_max),
        "rms_acc_max": float(rms_acc_max),
        "max_direction": int(max_direction),
        "x1amp": float(x1amp) if (x1amp is not None) else None,
        # ISO severity lookups
        "mc_fault_rms_res": mc_fault_rms_res,
        "mc_fault_peak_res": mc_fault_peak_res,
        # Harmonics & thresholds
        "dat_harmonics": dat_harmonics,
        "xyellow": float(xyellow),
        "xred": float(xred),
        # Fault classification results
        "faults": faults,
        "fault": fault,
        "type_specific_faults": type_specific_faults,
        # Advanced metrics (safe if None)
        "belt_slip_ratio": float(belt_slip_ratio)
        if belt_slip_ratio is not None
        else None,
        "fan_bpf_amp": float(fan_bpf_amp) if fan_bpf_amp is not None else None,
        "gear_mesh_score": float(gear_mesh_score)
        if gear_mesh_score is not None
        else None,
        "low_freq_rms": float(low_freq_rms) if low_freq_rms is not None else None,
    }

# -------------------------------------------------------------------------
# Batch fault detection over a DataFrame of latest records
# -------------------------------------------------------------------------
def detect_faults(df: pd.DataFrame) -> dict:
    """
    Run get_metrics(...) for every serialNumber in the DataFrame.

    Returns:
      { "189314686": {...metrics...}, ... }
    """
    results = {}
    for serialNumber in df.serialNumber:
        try:
            results[str(int(serialNumber))] = get_metrics(df, int(serialNumber))
        except Exception as e:
            results[str(int(serialNumber))] = {
                "serialNumber": int(serialNumber),
                "error": str(e),
            }
    return results


# -------------------------------------------------------------------------
# Manual test harness (run this file directly)
# -------------------------------------------------------------------------
if __name__ == "__main__":
    df = get_latest_vibration_data()
    print(df.head())
    # results = detect_faults(df)
    # print(results)
