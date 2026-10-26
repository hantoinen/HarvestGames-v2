# ETL/utils/create_compat_views.py
import os
from pathlib import Path
from sqlalchemy import create_engine, text
from dotenv import load_dotenv

load_dotenv(dotenv_path=Path(__file__).resolve().parent.parent.parent / ".env")
engine = create_engine(os.getenv("NEON_DATABASE_URL"))

COMPAT = {
    "dim_constraints":              "dim_constraints_v2",
    "dim_coordinates":              "dim_coordinates_v2",
    "dim_soils":                    "dim_soils_v2",
    "src_climate_past":             "src_climate_past_v2",
    "src_climate_future":           "src_climate_future_v2",
    "src_yields_cordex_2010_2025":  "src_yields_cordex_2010_2025_v2",
    "src_cultures_plots_2024":      "src_cultures_plots_2024_v2",
}

with engine.begin() as conn:
    for vue, table in COMPAT.items():
        conn.execute(text(
            f'CREATE OR REPLACE VIEW public."{vue}" AS SELECT * FROM public."{table}";'
        ))
        print(f"✅ {vue}  ->  {table}")