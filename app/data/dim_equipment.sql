-- Adjust catalog/schema here to match your environment.
CREATE SCHEMA IF NOT EXISTS hive_metastore.vibration;


CREATE TABLE IF NOT EXISTS hive_metastore.vibration.dim_equipment (
  -- Primary key / join key
  serialNumber       BIGINT        COMMENT 'Sensor / equipment serial number (matches fact table)',

  -- Naming / location
  asset_name         STRING        COMMENT 'Human-friendly asset name (e.g. Latitude Scrap Fan Pulley - DE)',
  equip_type         STRING        COMMENT 'Equipment type: ElectricMotor, Pulley, Fan, Pump, Gearbox, etc.',
  location           STRING        COMMENT 'Optional plant/area/line description',

  -- ISO / mechanical info
  mclass             INT           COMMENT 'ISO 10816/20816 machine group (1-4); fallback if inference fails',
  power_hp           DOUBLE        COMMENT 'Nameplate power in HP (if known)',
  power_kw           DOUBLE        COMMENT 'Nameplate power in kW (if known)',
  nominal_rpm        DOUBLE        COMMENT 'Nameplate speed in RPM (if known)',

  -- Cached running speed from VibExtractor (RPM in Hz)
  rpm_hz_cached      DOUBLE        COMMENT 'Cached running speed from VibExtractor (Hz)',

  -- Sensor orientation / channel mapping
  sensor_vertical_channel   INT    COMMENT 'Channel index for vertical direction (usually 1)',
  sensor_horizontal_channel INT    COMMENT 'Channel index for horizontal direction (usually 2)',
  sensor_axial_channel      INT    COMMENT 'Channel index for axial direction (usually 3)',

  -- Power train relationships (belts, gears, etc.)
  driven_by_sn       BIGINT        COMMENT 'serialNumber of driving asset (e.g. motor driving pulley)',
  speed_ratio        DOUBLE        COMMENT 'Driver_rpm / this_rpm (for belt/gear ratio)',

  -- Fan / gear geometry (for BPF, GMF, etc.)
  num_blades         INT           COMMENT 'Number of blades for fans/blowers (optional)',
  num_teeth          INT           COMMENT 'Number of teeth for gears (optional)',

  -- Severity tuning / criticality
  severity_scale     DOUBLE        COMMENT 'Scale factor applied to ISO thresholds (1.0=default, <1 stricter, >1 looser)',
  criticality        INT           COMMENT 'Criticality class (1=low, 2=medium, 3=high or similar scheme)',
  custom_vel_yellow  DOUBLE        COMMENT 'Optional custom yellow limit for velocity RMS (mm/s)',
  custom_vel_red     DOUBLE        COMMENT 'Optional custom red limit for velocity RMS (mm/s)',

  -- Timestamps / audit
  created_at         TIMESTAMP     COMMENT 'Record creation timestamp',
  updated_at         TIMESTAMP     COMMENT 'Record last update timestamp'
)
USING DELTA
TBLPROPERTIES (
  'delta.autoOptimize.optimizeWrite' = 'true',
  'delta.autoOptimize.autoCompact'   = 'true'
);
