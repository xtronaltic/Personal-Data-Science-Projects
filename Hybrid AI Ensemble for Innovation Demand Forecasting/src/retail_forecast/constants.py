from __future__ import annotations

# Source columns (as provided in Historical_Data.csv) -> snake_case canonical names
SOURCE_TO_SNAKE: dict[str, str] = {
    "Markets": "markets",
    "Periods": "periods",
    "Manufacturer": "manufacturer",
    "Category": "category",
    "Trademark": "trademark",
    "Brand": "brand",
    "$": "dollars",
    "Units": "units",
    "EQ": "eq",
    "Avg Unit Price": "avg_unit_price",
    "%ACV": "acv_pct",
    "TDP": "tdp",
    "$ / $MM ACV": "dollars_per_mm_acv",
    "Units / $MM ACV": "units_per_mm_acv",
    "EQ / $MM ACV": "eq_per_mm_acv",
}

# Full hierarchy columns in order (Markets > Manufacturer > Category > Trademark > Brand)
# This is the canonical hierarchy used throughout the system for:
# - series_id construction
# - similarity index grouping
# - forecast output grouping
HIERARCHY_COLS: list[str] = [
    "markets",
    "manufacturer",
    "category",
    "trademark",
    "brand",
]

# Separator for series_id construction
HIERARCHY_SEPARATOR: str = "||"

# Canonical (renamed) columns required after load/standardization
PANEL_REQUIRED_COLUMNS: list[str] = [
    "markets",
    "periods",
    "manufacturer",
    "category",
    "trademark",
    "brand",
    "dollars",
    "units",
    "eq",
    "avg_unit_price",
    "acv_pct",
    "tdp",
    "dollars_per_mm_acv",
    "units_per_mm_acv",
    "eq_per_mm_acv",
    "date",
]

# New innovation input is allowed to be thinner; still must include the core identifiers + periods/date
NEW_INNOVATION_REQUIRED_COLUMNS: list[str] = [
    "markets",
    "periods",
    "manufacturer",
    "category",
    "trademark",
    "brand",
    "dollars",
    "units",
    "eq",
    "acv_pct",
    "tdp",
    "dollars_per_mm_acv",
    "units_per_mm_acv",
    "eq_per_mm_acv",
    "date",
]

# Planned output CSV schema for forecasts (per Market, new brand under a trademark)
FORECAST_OUTPUT_FIELDS: list[str] = [
    "markets",
    "manufacturer",
    "category",
    "trademark",
    "brand",
    "asof_date",
    "date",
    "horizon_week",
    "dollars_p10",
    "dollars_p50",
    "dollars_p90",
    "units_p10",
    "units_p50",
    "units_p90",
    "eq_p10",
    "eq_p50",
    "eq_p90",
]
