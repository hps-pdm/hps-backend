"""
Helper script to manage the vibration.dim_equipment table without using
the Databricks UI. It can create the table (via VibExtractor DDL helper)
and upsert rows you define locally.
"""

from VibExtractor import ensure_dim_equipment_table, create_connection

# ---------------------------------------------------------------------------
# Edit this list to insert/update rows (keyed by serialNumber).
# Only serialNumber and asset_name/equip_type/mclass are commonly needed.
# ---------------------------------------------------------------------------
ROWS = [
    {
        "serialNumber": 189315064,
        "asset_name": "ABBBB",  # change name here
        "equip_type": "Gear",
        "mclass": 2,
        "power_hp": 60,
        "nominal_rpm": 1780,
        "rpm_hz_cached": None,
        "driven_by_sn": None,
        "speed_ratio": None,
    },
    # Add more dicts as needed...
]


def upsert_rows(rows):
    """
    Insert or update dim_equipment rows based on serialNumber.
    """
    if not rows:
        print("No rows provided; nothing to upsert.")
        return

    conn = create_connection()
    cur = conn.cursor()
    try:
        for r in rows:
            cur.execute(
                """
                MERGE INTO hive_metastore.vibration.dim_equipment AS t
                USING (SELECT ? AS serialNumber) AS s
                ON t.serialNumber = s.serialNumber
                WHEN MATCHED THEN UPDATE SET
                  asset_name    = ?,
                  equip_type    = ?,
                  mclass        = ?,
                  power_hp      = ?,
                  nominal_rpm   = ?,
                  rpm_hz_cached = ?,
                  driven_by_sn  = ?,
                  speed_ratio   = ?
                WHEN NOT MATCHED THEN INSERT (
                  serialNumber, asset_name, equip_type, mclass,
                  power_hp, nominal_rpm, rpm_hz_cached, driven_by_sn, speed_ratio
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    r["serialNumber"],
                    r.get("asset_name"),
                    r.get("equip_type"),
                    r.get("mclass"),
                    r.get("power_hp"),
                    r.get("nominal_rpm"),
                    r.get("rpm_hz_cached"),
                    r.get("driven_by_sn"),
                    r.get("speed_ratio"),
                    r["serialNumber"],
                    r.get("asset_name"),
                    r.get("equip_type"),
                    r.get("mclass"),
                    r.get("power_hp"),
                    r.get("nominal_rpm"),
                    r.get("rpm_hz_cached"),
                    r.get("driven_by_sn"),
                    r.get("speed_ratio"),
                ),
            )
        conn.commit()
    finally:
        cur.close()
        conn.close()


def main():
    ensure_dim_equipment_table()
    upsert_rows(ROWS)
    print(f"Upserted {len(ROWS)} row(s).")


if __name__ == "__main__":
    main()
