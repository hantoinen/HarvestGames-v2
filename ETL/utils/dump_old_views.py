# ETL/utils/dump_old_views.py
import os
from pathlib import Path
from sqlalchemy import create_engine, text
from dotenv import load_dotenv

load_dotenv(dotenv_path=Path(__file__).resolve().parent.parent.parent / ".env")

OLD_URL = os.getenv("NEON_DB_URL_OLD")   # <-- ancien Neon, en lecture seule
if not OLD_URL:
    raise ValueError("NEON_DB_URL_OLD manquant dans .env")

engine = create_engine(OLD_URL)

VIEWS = [
    "vw_cultures_plots_2024_clean",
    "training_vigne",
    "sol_dept",
    "maille_vigne",
    "climate_maille_vigne",
]

OUT_DIR = Path("dump_old_views")
OUT_DIR.mkdir(exist_ok=True)

with engine.connect() as conn:
    for v in VIEWS:
        row = conn.execute(
            text("SELECT pg_get_viewdef(CAST(:v AS regclass), true)"),
            {"v": v},
        ).scalar()
        if row is None:
            print(f"⚠️  Vue introuvable : {v}")
            continue
        (OUT_DIR / f"{v}.sql").write_text(row, encoding="utf-8")
        print(f"✅ {v}.sql exporté ({len(row)} caractères)")

print(f"\nFichiers écrits dans {OUT_DIR.resolve()}")