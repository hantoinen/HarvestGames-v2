"""
⚠️ ARTEFACT MVP — NON UTILISÉ DANS LE PIPELINE V2
==================================================

Diagnostic du biais CORDEX par méthode Delta.
Ce script a servi à l'exploration du MVP mais n'a pas été intégré à la V2.

Raison : l'application de la correction (era5_clim + delta) induit un
double comptage du réchauffement, car la période de référence CORDEX
(1976-2005) précède la période ERA5 (2010-2024) au lieu de la couvir.
La V2 a donc fait le choix de ne pas corriger les projections CORDEX
et de documenter le biais (~0,9 °C) comme limite assumée.

Ce script est conservé à titre de documentation méthodologique.
"""

"""
Correction de biais CORDEX par méthode delta (anomalies)
=========================================================

Étapes :
1. Récupération CORDEX historical (période de référence) via CADS API
2. Récupération CORDEX future (par horizon : 2054, 2084, 2100) via CADS API
3. Chargement ERA5 déjà téléchargé (aw_era5_monthly_2m_temperature_20260729.nc)
4. Réechantillonnage spatial CORDEX -> grille ERA5
5. Calcul de la climatologie mensuelle (12 valeurs/pixel) pour chaque période
6. Calcul du biais diagnostique (CORDEX historical - ERA5 historical)
7. Calcul du delta (CORDEX futur - CORDEX historical) par mois et par horizon
8. Application du delta à ERA5 -> projection corrigée

Prérequis : cdsapi, xarray, netCDF4, scipy, numpy configurés + ~/.cdsapirc valide
"""

import cdsapi
import xarray as xr
import numpy as np
from pathlib import Path

# ============================================================
# CONFIGURATION — à adapter à ton run CORDEX exact
# ============================================================

WORKDIR = Path("./cordex_data")
WORKDIR.mkdir(exist_ok=True)

ERA5_FILE = Path("raw_era5_monthly_2m_temperature_20260729.nc")

# Domaine CORDEX Europe
CORDEX_DOMAIN = "europe"  # EUR-11
CORDEX_EXPERIMENT_FUTURE = "rcp_8_5"  # à changer en "rcp_4_5" si besoin
CORDEX_GCM = "cnrm_cerfacs_cnrm_cm5"
CORDEX_RCM = "cnrm_aladin63"
CORDEX_ENSEMBLE = "r1i1p1"
CORDEX_VARIABLE = "2m_air_temperature"
CORDEX_TEMPORAL_RESOLUTION = "monthly_mean"

# Emprise spatiale (resserrée sur la France métropolitaine + marge)
# [Nord, Ouest, Sud, Est] en degrés
AREA_FRANCE = [51.5, -5.5, 41.0, 9.5]

# Période de référence historique (standard CORDEX)
HIST_START, HIST_END = 1976, 2005

# Horizons et fenêtres associées (convention IPCC : 20 ans centrés)
HORIZONS = {
    2054: (2045, 2064),
    2084: (2075, 2094),
    2100: (2091, 2100),  # tronqué : CORDEX s'arrête à 2100
}

MONTHS = list(range(1, 13))


# ============================================================
# 1. RÉCUPÉRATION CORDEX VIA CADS API
# ============================================================

def fetch_cordex(experiment, start_year, end_year, out_path):
    """Récupère un run CORDEX mensuel pour une période donnée.

    Depuis le 9 juillet 2026, l'API CORDEX a changé : les mots-clés
    "start_year"/"end_year" sont obsolètes et remplacés par "year"
    (liste d'années) et "month" (liste de mois). Voir :
    https://forum.ecmwf.int/t/new-features-implemented-for-cordex-regional-climate-model-data-on-single-levels-on-the-cds/15145
    """
    if out_path.exists():
        print(f"[SKIP] {out_path} existe déjà.")
        return out_path

    c = cdsapi.Client()
    years = [str(y) for y in range(start_year, end_year + 1)]
    months = [f"{m:02d}" for m in range(1, 13)]

    c.retrieve(
        "projections-cordex-domains-single-levels",
        {
            "format": "zip",
            "domain": CORDEX_DOMAIN,
            "experiment": experiment,
            "horizontal_resolution": "0_11_degree_x_0_11_degree",
            "temporal_resolution": CORDEX_TEMPORAL_RESOLUTION,
            "variable": CORDEX_VARIABLE,
            "gcm_model": CORDEX_GCM,
            "rcm_model": CORDEX_RCM,
            "ensemble_member": CORDEX_ENSEMBLE,
            "year": years,
            "month": months,
            "area": AREA_FRANCE,  # [Nord, Ouest, Sud, Est]
        },
        str(out_path),
    )
    print(f"[OK] Téléchargé : {out_path}")
    return out_path


def fetch_all_cordex():
    """Récupère le run historical + un run par horizon futur."""
    paths = {}

    # Run historique
    hist_zip = WORKDIR / f"cordex_historical_{HIST_START}_{HIST_END}.zip"
    paths["historical"] = fetch_cordex("historical", HIST_START, HIST_END, hist_zip)

    # Runs futurs (un par horizon, car les fenêtres se chevauchent peu)
    for horizon, (start, end) in HORIZONS.items():
        fut_zip = WORKDIR / f"cordex_{CORDEX_EXPERIMENT_FUTURE}_{start}_{end}.zip"
        paths[horizon] = fetch_cordex(CORDEX_EXPERIMENT_FUTURE, start, end, fut_zip)

    return paths


# ============================================================
# 2. CHARGEMENT ET EXTRACTION DES .nc DEPUIS LES ZIP
# ============================================================

def extract_and_load(zip_path):
    """Dézippe et charge le NetCDF CORDEX en dataset xarray."""
    import zipfile

    extract_dir = zip_path.with_suffix("")
    if not extract_dir.exists():
        with zipfile.ZipFile(zip_path, "r") as z:
            z.extractall(extract_dir)

    nc_files = list(extract_dir.glob("*.nc"))
    if not nc_files:
        raise FileNotFoundError(f"Aucun .nc trouvé dans {extract_dir}")

    ds = xr.open_mfdataset(nc_files, combine="by_coords")
    return ds


# ============================================================
# 3. CLIMATOLOGIE MENSUELLE (12 valeurs par pixel)
# ============================================================

def monthly_climatology(ds, var_name="tas"):
    """Moyenne par mois calendaire sur toute la période du dataset."""
    da = ds[var_name]
    if da.max() > 100:  # detection kelvin -> conversion en celsius
        da = da - 273.15
    clim = da.groupby("time.month").mean(dim="time")
    return clim  # dims: (month, lat, lon)


# ============================================================
# 4. RÉECHANTILLONNAGE SPATIAL CORDEX -> GRILLE ERA5
# ============================================================

def regrid_to_era5(cordex_clim, era5_ds):
    """Interpolation bilinéaire de la climatologie CORDEX sur la grille ERA5."""
    target_lat = era5_ds["latitude"]
    target_lon = era5_ds["longitude"]

    regridded = cordex_clim.interp(
        lat=target_lat, lon=target_lon, method="linear"
    )
    return regridded


# ============================================================
# 5. PIPELINE PRINCIPAL
# ============================================================

def main():
    print("=== 1. Téléchargement CORDEX ===")
    cordex_paths = fetch_all_cordex()

    print("=== 2. Chargement ERA5 (local) ===")
    era5_ds = xr.open_dataset(ERA5_FILE)
    era5_var = list(era5_ds.data_vars)[0]  # ajuster si nom connu, ex "t2m"
    print(f"Variable ERA5 détectée : {era5_var}")

    era5_da = era5_ds[era5_var]
    if era5_da.max() > 100:
        era5_da = era5_da - 273.15

    era5_clim = era5_da.groupby("time.month").mean(dim="time")

    print("=== 3. Climatologie CORDEX historical + régrillage ===")
    ds_hist = extract_and_load(cordex_paths["historical"])
    clim_hist = monthly_climatology(ds_hist)
    clim_hist_regrid = regrid_to_era5(clim_hist, era5_ds)

    print("=== 4. Biais diagnostique (CORDEX hist - ERA5 hist) ===")
    # Attention : le biais n'a de sens que sur la période commune (2010-2025)
    bias = clim_hist_regrid - era5_clim
    bias_path = WORKDIR / "biais_cordex_vs_era5.nc"
    bias.to_netcdf(bias_path)
    print(f"[OK] Biais sauvegardé : {bias_path}")
    print("Biais moyen par mois (°C, moyenné spatialement) :")
    print(bias.mean(dim=["lat", "lon"]).values)

    print("=== 5. Delta et projection corrigée par horizon ===")
    results = {}
    for horizon in HORIZONS:
        ds_fut = extract_and_load(cordex_paths[horizon])
        clim_fut = monthly_climatology(ds_fut)
        clim_fut_regrid = regrid_to_era5(clim_fut, era5_ds)

        delta = clim_fut_regrid - clim_hist_regrid
        projection_corrigee = era5_clim + delta

        delta_path = WORKDIR / f"delta_{horizon}.nc"
        proj_path = WORKDIR / f"projection_corrigee_{horizon}.nc"
        delta.to_netcdf(delta_path)
        projection_corrigee.to_netcdf(proj_path)

        results[horizon] = {
            "delta_moyen_annuel": float(delta.mean().values),
            "delta_path": delta_path,
            "projection_path": proj_path,
        }
        print(f"[OK] Horizon {horizon} — delta moyen annuel : "
              f"{results[horizon]['delta_moyen_annuel']:.2f} °C")

    print("=== Terminé ===")
    return bias, results


if __name__ == "__main__":
    main()