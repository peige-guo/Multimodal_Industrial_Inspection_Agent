"""Parse optional sensor CSV input and evaluate threshold abnormalities.

CSV is expected to have a header row with sensor names. The last row is taken
as the most recent reading. Values are coerced to float where possible.
"""

from __future__ import annotations

import csv
import io

from backend.app.schemas.inspection import SensorReading

# Critical thresholds aligned with the demo standard (clause 3.3).
CRITICAL_SENSOR_THRESHOLDS: dict[str, float] = {
    "temperature": 80.0,  # Celsius
    "vibration": 10.0,  # mm/s
    "pressure_deviation": 15.0,  # percent
}

_UNITS = {
    "temperature": "C",
    "vibration": "mm/s",
    "pressure": "bar",
    "pressure_deviation": "%",
    "current": "A",
}


def parse_sensor_csv(data: bytes | str) -> list[SensorReading]:
    """Parse sensor CSV bytes/str into the latest reading per column."""
    if isinstance(data, bytes):
        data = data.decode("utf-8", errors="replace")
    text = data.strip()
    if not text:
        return []

    reader = csv.DictReader(io.StringIO(text))
    rows = list(reader)
    if not rows or reader.fieldnames is None:
        return []

    latest = rows[-1]
    readings: list[SensorReading] = []
    for name in reader.fieldnames:
        raw = (latest.get(name) or "").strip()
        if raw == "":
            continue
        try:
            value = float(raw)
        except ValueError:
            continue
        key = name.strip().lower()
        readings.append(
            SensorReading(name=key, value=value, unit=_UNITS.get(key))
        )
    return readings


def evaluate_sensor_abnormality(
    readings: list[SensorReading],
) -> tuple[bool, list[str]]:
    """Return (is_critical, reasons) for the given sensor readings."""
    reasons: list[str] = []
    for reading in readings:
        threshold = CRITICAL_SENSOR_THRESHOLDS.get(reading.name)
        if threshold is not None and reading.value > threshold:
            reasons.append(
                f"{reading.name} {reading.value}"
                f"{(' ' + reading.unit) if reading.unit else ''} "
                f"exceeds critical threshold {threshold}"
            )
    return (len(reasons) > 0, reasons)
