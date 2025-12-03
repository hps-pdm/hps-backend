# backend/app/debug_print_latest.py
from app.data.vx_adapter import get_latest_vibration_data

sn = 189315097
df = get_latest_vibration_data(serialNumber=sn)
print(df.columns)
print(df[["serialNumber", "sampleRate", "direction1", "direction2", "direction3"]].head())
