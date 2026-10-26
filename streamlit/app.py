"""Harvest Games v2 — Streamlit unifié.

Structure :
  Onglet 1 — Analyse climatique (dataviz climatique + carte de favorabilité)
  Onglet 2 — Rendement vigne ML (XGBoost via API FastAPI)
  Onglet 3 — À propos du modèle

Lancement (depuis la racine du repo) :
    python streamlit/api.py         # terminal 1
    streamlit run streamlit/app.py  # terminal 2
"""

import json
import os
from pathlib import Path

from dotenv import load_dotenv

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
import streamlit as st

# Charge le .env de la racine du repo
ROOT = Path(__file__).resolve().parent.parent
load_dotenv(dotenv_path=ROOT / ".env")


# =========================================================
# CONFIG
# =========================================================

st.set_page_config(page_title="Harvest Games", page_icon="🌾", layout="wide")

BASE_DIR = Path(__file__).resolve().parent
ASSETS_DIR = BASE_DIR / "assets"
API_URL = os.getenv("API_URL", "http://127.0.0.1:8001")

CULTURE_INFO = {
    "Vigne à raisin de cuve": {"picto": "🍇", "id": 1},
    "Pois chiche": {"picto": "🌱", "id": 2},
    "Pomme de terre": {"picto": "🥔", "id": 3},
}

# Profils agronomiques simplifiés pour la carte de favorabilité (différents
# de dim_constraints, volontairement assouplis pour visualiser la migration).
CULTURE_PROFILES = {
    "Vigne": {
        "picto": "🍇",
        "temp_optimal_min": 14.0,
        "temp_optimal_max": 20.0,
        "temp_lethal_min": 8.0,
        "temp_lethal_max": 28.0,
        "preferred_soils": ["cambisols", "rendzines", "fluvisols", "calcisols"],
    },
    "Pois chiche": {
        "picto": "🌱",
        "temp_optimal_min": 17.0,
        "temp_optimal_max": 24.0,
        "temp_lethal_min": 10.0,
        "temp_lethal_max": 32.0,
        "preferred_soils": ["calcisols", "cambisols", "fluvisols", "leptosols"],
    },
    "Pomme de terre": {
        "picto": "🥔",
        "temp_optimal_min": 13.0,
        "temp_optimal_max": 19.0,
        "temp_lethal_min": 7.0,
        "temp_lethal_max": 26.0,
        "preferred_soils": ["luvisols", "cambisols", "fluvisols", "podzols"],
    },
}

DEPT_NAMES = {
    '01': 'Ain', '02': 'Aisne', '03': 'Allier', '04': 'Alpes-de-Haute-Provence',
    '05': 'Hautes-Alpes', '06': 'Alpes-Maritimes', '07': 'Ardèche', '08': 'Ardennes',
    '09': 'Ariège', '10': 'Aube', '11': 'Aude', '12': 'Aveyron',
    '13': 'Bouches-du-Rhône', '14': 'Calvados', '15': 'Cantal', '16': 'Charente',
    '17': 'Charente-Maritime', '18': 'Cher', '19': 'Corrèze', '21': "Côte-d'Or",
    '22': "Côtes-d'Armor", '23': 'Creuse', '24': 'Dordogne', '25': 'Doubs',
    '26': 'Drôme', '27': 'Eure', '28': 'Eure-et-Loir', '29': 'Finistère',
    '2A': 'Corse-du-Sud', '2B': 'Haute-Corse',
    '30': 'Gard', '31': 'Haute-Garonne', '32': 'Gers', '33': 'Gironde',
    '34': 'Hérault', '35': 'Ille-et-Vilaine', '36': 'Indre', '37': 'Indre-et-Loire',
    '38': 'Isère', '39': 'Jura', '40': 'Landes', '41': 'Loir-et-Cher',
    '42': 'Loire', '43': 'Haute-Loire', '44': 'Loire-Atlantique', '45': 'Loiret',
    '46': 'Lot', '47': 'Lot-et-Garonne', '48': 'Lozère', '49': 'Maine-et-Loire',
    '50': 'Manche', '51': 'Marne', '52': 'Haute-Marne', '53': 'Mayenne',
    '54': 'Meurthe-et-Moselle', '55': 'Meuse', '56': 'Morbihan', '57': 'Moselle',
    '58': 'Nièvre', '59': 'Nord', '60': 'Oise', '61': 'Orne',
    '62': 'Pas-de-Calais', '63': 'Puy-de-Dôme', '64': 'Pyrénées-Atlantiques',
    '65': 'Hautes-Pyrénées', '66': 'Pyrénées-Orientales', '67': 'Bas-Rhin',
    '68': 'Haut-Rhin', '69': 'Rhône', '70': 'Haute-Saône', '71': 'Saône-et-Loire',
    '72': 'Sarthe', '73': 'Savoie', '74': 'Haute-Savoie', '75': 'Paris',
    '76': 'Seine-Maritime', '77': 'Seine-et-Marne', '78': 'Yvelines',
    '79': 'Deux-Sèvres', '80': 'Somme', '81': 'Tarn', '82': 'Tarn-et-Garonne',
    '83': 'Var', '84': 'Vaucluse', '85': 'Vendée', '86': 'Vienne',
    '87': 'Haute-Vienne', '88': 'Vosges', '89': 'Yonne', '90': 'Territoire de Belfort',
    '91': 'Essonne', '92': 'Hauts-de-Seine', '93': 'Seine-Saint-Denis',
    '94': 'Val-de-Marne', '95': "Val-d'Oise",
}


# =========================================================
# CONNEXION NEON
# =========================================================

def get_database_url() -> str:
    database_url = os.getenv("NEON_DATABASE_URL")
    if database_url:
        return database_url
    try:
        database_url = st.secrets["NEON_DATABASE_URL"]
    except (KeyError, FileNotFoundError):
        database_url = None
    if not database_url:
        raise RuntimeError("NEON_DATABASE_URL absente du .env ou des secrets Streamlit.")
    return str(database_url)


conn = st.connection(
    "postgresql",
    type="sql",
    url=get_database_url(),
    pool_recycle=300,
    connect_args={
        "sslmode": "require",
        "keepalives": 1,
        "keepalives_idle": 30,
        "keepalives_interval": 10,
        "keepalives_count": 5,
    },
)


# =========================================================
# HELPERS
# =========================================================

def format_entier(value) -> str:
    if pd.isna(value):
        return "Non disponible"
    return f"{float(value):,.0f}".replace(",", " ")


def format_decimal(value, digits: int = 1) -> str:
    if pd.isna(value):
        return "Non disponible"
    return f"{float(value):,.{digits}f}".replace(",", " ")


# =========================================================
# CHARGEMENTS ONGLET 1 — CARTE THERMIQUE
# =========================================================

@st.cache_data(ttl=600)
def load_constraints() -> pd.DataFrame:
    return conn.query(
        "SELECT * FROM public.dim_constraints ORDER BY culture_id;", ttl=600
    )


@st.cache_data(ttl=600)
def load_national_stats(culture_id: int) -> pd.DataFrame:
    return conn.query(
        """
        SELECT
            COUNT(DISTINCT parcel_id) AS nb_parcelles,
            COUNT(DISTINCT coordinates_id) AS nb_mailles,
            SUM(surf_ha) AS total_ha
        FROM public.vw_cultures_plots_2024_clean
        WHERE culture_id = :culture_id;
        """,
        params={"culture_id": int(culture_id)},
        ttl=600,
    )


@st.cache_data(ttl=600)
def load_current_culture_data(culture_id: int, annee_cible: int = 2054) -> pd.DataFrame:
    return conn.query(
        """
        WITH cultures_par_maille AS (
            SELECT coordinates_id,
                   COUNT(DISTINCT parcel_id) AS nb_parcelles,
                   SUM(surf_ha) AS surface_ha_exacte
            FROM public.vw_cultures_plots_2024_clean
            WHERE culture_id = :culture_id
            GROUP BY coordinates_id
        ),
        climat_2024 AS (
            SELECT coordinates_id, month,
                   AVG(temp_mean_past) AS temp_2024
            FROM public.src_climate_past
            WHERE year = 2024 AND month BETWEEN 4 AND 8
            GROUP BY coordinates_id, month
        ),
        climat_futur AS (
            SELECT coordinates_id, month,
                   AVG(temp_mean_future) AS temp_future
            FROM public.src_climate_future
            WHERE year = :annee_cible AND month BETWEEN 4 AND 8
            GROUP BY coordinates_id, month
        )
        SELECT cultures.coordinates_id, coord.lat, coord.lon,
               futur.month, passe.temp_2024, futur.temp_future,
               cultures.nb_parcelles, cultures.surface_ha_exacte
        FROM cultures_par_maille AS cultures
        JOIN public.dim_coordinates AS coord
          ON coord.coordinates_id = cultures.coordinates_id
        JOIN climat_futur AS futur
          ON futur.coordinates_id = cultures.coordinates_id
        LEFT JOIN climat_2024 AS passe
          ON passe.coordinates_id = cultures.coordinates_id
         AND passe.month = futur.month
        ORDER BY cultures.coordinates_id, futur.month;
        """,
        params={"culture_id": int(culture_id), "annee_cible": int(annee_cible)},
        ttl=600,
    )


# =========================================================
# CHARGEMENTS ONGLET 1 — CARTE DE FAVORABILITÉ
# =========================================================

@st.cache_data(ttl=600)
def load_agro_data() -> pd.DataFrame:
    """Charge climat + sol par département (long format : 1 ligne par horizon)."""
    query = """
    WITH dim_soils_norm AS (
        SELECT
            coordinates_id,
            CASE
                WHEN department_code ~ '^0[0-9]{2}$' AND length(department_code) = 3 THEN right(department_code, 2)
                WHEN department_code ~ '^0[0-9]$' THEN department_code
                WHEN department_code ~ '^[0-9]+$' AND length(department_code) <= 2 THEN lpad(department_code, 2, '0')
                WHEN department_code ~ '^[0-9]+\\.0$' THEN lpad(replace(department_code, '.0', ''), 2, '0')
                WHEN department_code IN ('2A', '02A') THEN '2A'
                WHEN department_code IN ('2B', '02B') THEN '2B'
                ELSE department_code
            END AS department_code,
            department_name,
            soil_family_type,
            soil_type
        FROM public.dim_soils
        WHERE department_code IS NOT NULL
    ),
    sol AS (
        SELECT
            department_code,
            MAX(department_name) AS department_name,
            MODE() WITHIN GROUP (ORDER BY soil_family_type) AS soil_family,
            MODE() WITHIN GROUP (ORDER BY soil_type) AS soil_type
        FROM dim_soils_norm
        WHERE soil_family_type IS NOT NULL
        GROUP BY department_code
    ),
    climat_2024 AS (
        SELECT
            dn.department_code,
            AVG(cp.temp_mean_past) AS temp,
            SUM(cp.precip_past) AS precip
        FROM public.src_climate_past cp
        JOIN dim_soils_norm dn ON dn.coordinates_id = cp.coordinates_id
        WHERE cp.year = 2024 AND cp.month BETWEEN 4 AND 8
        GROUP BY dn.department_code
    ),
    climat_futur AS (
        SELECT
            dn.department_code,
            cf.year,
            AVG(cf.temp_mean_future) AS temp,
            SUM(cf.precip_future) AS precip
        FROM public.src_climate_future cf
        JOIN dim_soils_norm dn ON dn.coordinates_id = cf.coordinates_id
        WHERE cf.year IN (2054, 2084, 2100) AND cf.month BETWEEN 4 AND 8
        GROUP BY dn.department_code, cf.year
    )
    SELECT
        sol.department_code,
        sol.department_name,
        sol.soil_family,
        sol.soil_type,
        c24.temp AS temp_2024,
        c24.precip AS precip_2024,
        cf.year,
        cf.temp AS temp_futur,
        cf.precip AS precip_futur
    FROM sol
    LEFT JOIN climat_2024 c24 ON c24.department_code = sol.department_code
    LEFT JOIN climat_futur cf ON cf.department_code = sol.department_code
    ORDER BY sol.department_code, cf.year;
    """
    return conn.query(query, ttl=600)


@st.cache_data(ttl=3600)
def load_departments_geojson() -> dict:
    path = ASSETS_DIR / "departements.geojson"
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


# =========================================================
# SCORING AGRONOMIQUE (pour la carte de favorabilité)
# =========================================================

def normalize_soil(s: str) -> str:
    if pd.isna(s):
        return ""
    return str(s).lower().split(" (")[0].strip()


def score_temperature(temp: float, profile: dict) -> float:
    """Score 0-100 selon la position de la température vs plage optimale."""
    if pd.isna(temp):
        return np.nan
    tmin = profile["temp_optimal_min"]
    tmax = profile["temp_optimal_max"]
    tleth_min = profile["temp_lethal_min"]
    tleth_max = profile["temp_lethal_max"]
    if tmin <= temp <= tmax:
        return 100.0
    if temp < tmin:
        if temp <= tleth_min:
            return 0.0
        return 100 * (temp - tleth_min) / (tmin - tleth_min)
    if temp >= tleth_max:
        return 0.0
    return 100 * (tleth_max - temp) / (tleth_max - tmax)


def score_soil(soil_family: str, profile: dict) -> float:
    soil = normalize_soil(soil_family)
    if not soil:
        return 50.0
    return 100.0 if soil in profile["preferred_soils"] else 30.0


def categorize(score: float) -> str:
    if pd.isna(score):
        return "Non évalué"
    if score >= 80:
        return "Très favorable"
    if score >= 60:
        return "Favorable"
    if score >= 40:
        return "Modéré"
    if score >= 20:
        return "Marginal"
    return "Défavorable"


# =========================================================
# CHARGEMENTS ONGLET 2 (ML vigne via API)
# =========================================================

@st.cache_data(ttl=300)
def fetch_api(endpoint: str) -> pd.DataFrame:
    response = requests.get(f"{API_URL}/{endpoint}", timeout=10)
    response.raise_for_status()
    return pd.DataFrame(response.json())


# =========================================================
# EN-TÊTE + TITRE
# =========================================================

st.title("🌾 Harvest Games — Observatoire agricole du futur")

st.markdown("---")

# =========================================================
# ONGLETS
# =========================================================

tab1, tab2, tab3 = st.tabs([
    "📊 Analyse climatique",
    "🍇 Rendement vigne (ML)",
    "🧠 À propos",
])


# =========================================================
# ONGLET 1 — Analyse climatique
# =========================================================

with tab1:
    st.header("Analyse climatique des cultures")
    st.markdown(
        "Cartographie de l'exposition thermique des cultures historiques "
        "en France, à partir des données ERA5 (passé) et CORDEX (futur)."
    )

    try:
        df_constraints = load_constraints()
    except Exception as e:
        st.error("La connexion à Neon a échoué.")
        st.exception(e)
        st.stop()

    # --- Stats parcelles/surface ---
    col_culture, col_parcelles, col_surface = st.columns([2, 2, 2])

    with col_culture:
        culture_sel_header = st.selectbox(
            "Culture de référence :",
            list(CULTURE_INFO.keys()),
            key="culture_header",
            format_func=lambda v: f"{CULTURE_INFO[v]['picto']} {v}",
        )

    culture_id_header = CULTURE_INFO[culture_sel_header]["id"]
    stats_nat = load_national_stats(culture_id_header)

    if stats_nat.empty:
        nb_parcelles, total_ha_france = 0, 0
    else:
        nb_parcelles = stats_nat.iloc[0]["nb_parcelles"]
        total_ha_france = stats_nat.iloc[0]["total_ha"]

    with col_parcelles:
        st.metric("Parcelles agricoles 2024", format_entier(nb_parcelles))
    with col_surface:
        st.metric("Surface totale en France en 2024", f"{format_entier(total_ha_france)} ha")

    st.markdown("---")

    # ---------------------------------------------------------
    # SECTION 1 — Carte thermique (maille, ERA5/CORDEX)
    # ---------------------------------------------------------

    st.subheader("🌡️ Zones cultivées menacées par la hausse des températures")

    col_cmd, col_carte = st.columns([1, 3], gap="medium")

    with col_cmd:
        st.markdown("### 🎛️ Paramètres")

        culture_sel = st.selectbox(
            "Culture :",
            list(CULTURE_INFO.keys()),
            key="culture_carte1",
            format_func=lambda v: f"{CULTURE_INFO[v]['picto']} {v}",
        )

        mode_analyse = st.radio(
            "Mode d'analyse",
            ["Situation observée (2024)", "Projection CORDEX 2054", "Simulation de hausse"],
            key="mode_carte1",
        )

        delta_temp = 0.0
        if mode_analyse == "Simulation de hausse":
            delta_temp = st.slider(
                "Hausse supplémentaire de température (°C)",
                min_value=0.0, max_value=20.0, value=0.0, step=0.5,
                key="delta_carte1",
            )

    culture_id = CULTURE_INFO[culture_sel]["id"]
    contrainte = df_constraints.loc[df_constraints["culture_id"] == culture_id]

    if not contrainte.empty and pd.notna(contrainte.iloc[0]["temp_opt_max"]):
        seuil_alerte = float(contrainte.iloc[0]["temp_opt_max"])
    else:
        seuil_alerte = 30.0

    with col_carte:
        st.caption(
            "Pour chaque maille cultivée, on conserve le mois le plus "
            "défavorable entre avril et août."
        )

        try:
            df_current = load_current_culture_data(culture_id=culture_id, annee_cible=2054).copy()
        except Exception as e:
            st.error("Les données de la carte thermique n'ont pas pu être chargées.")
            st.exception(e)
            df_current = pd.DataFrame()

        if df_current.empty:
            st.warning("Aucune donnée thermique disponible pour cette culture.")
        else:
            marge_orange = 2.0

            if mode_analyse == "Situation observée (2024)":
                df_current["temperature_analysee"] = df_current["temp_2024"]
                titre = f"Situation observée en 2024 — {culture_sel}"
            elif mode_analyse == "Projection CORDEX 2054":
                df_current["temperature_analysee"] = df_current["temp_future"]
                titre = f"Projection CORDEX 2054 — {culture_sel}"
            else:
                df_current["temperature_analysee"] = df_current["temp_future"] + delta_temp
                titre = f"Projection 2054 + {delta_temp:.1f} °C — {culture_sel}"

            def statut_thermique(t):
                if pd.isna(t):
                    return "Données manquantes"
                if t <= seuil_alerte:
                    return "Viable"
                if t <= seuil_alerte + marge_orange:
                    return "Critique"
                return "Fatale"

            df_current["statut_mensuel"] = df_current["temperature_analysee"].apply(statut_thermique)
            ordre = {"Données manquantes": 0, "Viable": 1, "Critique": 2, "Fatale": 3}
            df_current["ordre"] = df_current["statut_mensuel"].map(ordre)

            df_carte = (
                df_current
                .sort_values(["coordinates_id", "ordre", "temperature_analysee"],
                             ascending=[True, False, False])
                .drop_duplicates("coordinates_id", keep="first")
            )

            total_surf = df_carte["surface_ha_exacte"].sum()
            surfaces = df_carte.groupby("statut_mensuel")["surface_ha_exacte"].sum().to_dict()

            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Surface analysée", f"{format_entier(total_surf)} ha")
            m2.metric("Zones viables 🟢", f"{surfaces.get('Viable', 0) / total_surf * 100:.2f} %" if total_surf else "0 %")
            m3.metric("Zones critiques 🟠", f"{surfaces.get('Critique', 0) / total_surf * 100:.2f} %" if total_surf else "0 %")
            m4.metric("Zones fatales 🔴", f"{surfaces.get('Fatale', 0) / total_surf * 100:.2f} %" if total_surf else "0 %")

            fig = px.scatter_mapbox(
                df_carte,
                lat="lat", lon="lon",
                color="statut_mensuel",
                size="surface_ha_exacte",
                size_max=12,
                color_discrete_map={
                    "Viable": "#77dd77", "Critique": "#f39c12",
                    "Fatale": "#e74c3c", "Données manquantes": "#7f8c8d",
                },
                category_orders={"statut_mensuel": ["Viable", "Critique", "Fatale", "Données manquantes"]},
                mapbox_style="https://basemaps.cartocdn.com/gl/positron-nolabels-gl-style/style.json",
                zoom=4.3, center={"lat": 46.2, "lon": 2.2},
                title=titre, opacity=0.8,
                hover_data={
                    "statut_mensuel": True,
                    "temperature_analysee": ":.1f",
                    "surface_ha_exacte": ":,.2f",
                    "nb_parcelles": True,
                    "lat": False, "lon": False, "coordinates_id": False,
                },
            )
            fig.update_layout(
                height=600, margin={"r": 0, "t": 45, "l": 0, "b": 70},
                hoverlabel={"font_size": 16},
                legend={"orientation": "h", "y": -0.08, "x": 0.5, "xanchor": "center"},
            )
            st.plotly_chart(fig, use_container_width=True)
            st.markdown(
                f"**Légende :** 🟢 ≤ {seuil_alerte:.1f} °C | "
                f"🟠 {seuil_alerte:.1f}–{seuil_alerte + marge_orange:.1f} °C | "
                f"🔴 > {seuil_alerte + marge_orange:.1f} °C"
            )

    st.markdown("---")

    # ---------------------------------------------------------
    # SECTION 2 — Carte de favorabilité agronomique (département)
    # ---------------------------------------------------------

    st.subheader("🗺️ Favorabilité climatique par département (score agronomique simplifié)")

    try:
        df_agro = load_agro_data()
        geojson = load_departments_geojson()
    except Exception as e:
        st.error("Impossible de charger les données de favorabilité.")
        st.exception(e)
        df_agro = pd.DataFrame()
        geojson = None

    if df_agro.empty or geojson is None:
        st.warning("Carte de favorabilité indisponible.")
    else:
        col_cmd_fav, col_carte_fav = st.columns([1, 3], gap="medium")

        with col_cmd_fav:
            st.markdown("### 🎛️ Paramètres")

            culture_fav = st.selectbox(
                "Culture :",
                list(CULTURE_PROFILES.keys()),
                format_func=lambda x: f"{CULTURE_PROFILES[x]['picto']} {x}",
                key="culture_fav",
            )

            horizon_fav = st.radio(
                "Horizon",
                ["2024 (référence)", "2054", "2084", "2100"],
                key="horizon_fav",
            )

            delta_fav = st.slider(
                "Hausse supplémentaire (°C)",
                min_value=0.0,
                max_value=15.0,
                value=0.0,
                step=0.5,
                key="delta_fav",
            )

        # --- Préparation des données (avant l'affichage) ---
        profile = CULTURE_PROFILES[culture_fav]

        if horizon_fav == "2024 (référence)":
            df_fav = df_agro.drop_duplicates(subset=["department_code"]).copy()
            df_fav["temp_effective"] = df_fav["temp_2024"] + delta_fav
            label_fav = "2024"
        else:
            year_fav = int(horizon_fav)
            df_fav = df_agro[df_agro["year"] == year_fav].copy()
            df_fav["temp_effective"] = df_fav["temp_futur"] + delta_fav
            label_fav = str(year_fav)

        df_fav["score_temp"] = df_fav["temp_effective"].apply(
            lambda t: score_temperature(t, profile)
        )
        df_fav["score_soil"] = df_fav["soil_family"].apply(
            lambda s: score_soil(s, profile)
        )
        df_fav["score_global"] = 0.75 * df_fav["score_temp"] + 0.25 * df_fav["score_soil"]
        df_fav["categorie"] = df_fav["score_global"].apply(categorize)

        with col_carte_fav:
            st.caption(
                "Score 0-100 : température (75 %) + sol dominant (25 %). "
                "Le slider permet de simuler une hausse supplémentaire au-delà "
                "du scénario CORDEX."
            )

            nb_eval = df_fav["score_global"].notna().sum()
            counts = df_fav["categorie"].value_counts().to_dict()
            fav = counts.get("Très favorable", 0) + counts.get("Favorable", 0)
            defav = counts.get("Défavorable", 0) + counts.get("Marginal", 0)
            temp_moy = df_fav["temp_effective"].mean()

            mf1, mf2, mf3, mf4 = st.columns(4)
            mf1.metric("Départements évalués", nb_eval)
            mf2.metric("🟢 Favorables", fav, delta=f"{fav/nb_eval*100:.0f} %" if nb_eval else None)
            mf3.metric("🔴 Défavorables / marginaux", defav)
            mf4.metric("Temp. moy. nationale", f"{temp_moy:.1f} °C" if pd.notna(temp_moy) else "n/a")

            color_map = {
                "Très favorable": "#2ecc71",
                "Favorable": "#a8e6a1",
                "Modéré": "#f1c40f",
                "Marginal": "#e67e22",
                "Défavorable": "#e74c3c",
                "Non évalué": "#7f8c8d",
            }

            fig_fav = px.choropleth(
                df_fav,
                geojson=geojson,
                locations="department_code",
                featureidkey="properties.code",
                color="categorie",
                color_discrete_map=color_map,
                category_orders={
                    "categorie": [
                        "Très favorable", "Favorable", "Modéré",
                        "Marginal", "Défavorable", "Non évalué",
                    ]
                },
                hover_name="department_name",
                hover_data={
                    "department_code": True,
                    "score_global": ":.0f",
                    "score_temp": ":.0f",
                    "temp_effective": ":.1f",
                    "soil_family": True,
                },
                labels={
                    "score_global": "Score global",
                    "score_temp": "Score température",
                    "temp_effective": "Température effective (°C)",
                    "soil_family": "Sol dominant",
                },
            )
            fig_fav.update_geos(fitbounds="locations", visible=False)
            fig_fav.update_layout(
                height=600,
                margin={"r": 0, "t": 45, "l": 0, "b": 70},
                hoverlabel={"font_size": 16},
                title=f"{culture_fav} — {label_fav} (+{delta_fav:.1f} °C)",
                showlegend=False,
            )
            st.plotly_chart(fig_fav, use_container_width=True)

            st.markdown(
                "**Légende :** "
                "🟢 Très favorable (≥ 80) | "
                "🟢 Favorable (60-80) | "
                "🟡 Modéré (40-60) | "
                "🟠 Marginal (20-40) | "
                "🔴 Défavorable (< 20)"
            )
            st.caption(
                "⚠️ Cette carte utilise un modèle agronomique simplifié, différent "
                "du ML XGBoost de l'onglet « Rendement vigne ». Elle ne prédit pas "
                "un rendement : elle évalue la compatibilité climat + sol."
            )

        with st.expander("📋 Détail par département"):
            df_detail = (
                df_fav[[
                    "department_code", "department_name", "soil_family",
                    "temp_effective", "score_temp", "score_soil",
                    "score_global", "categorie",
                ]]
                .sort_values("score_global", ascending=False)
                .round(1)
            )
            df_detail.columns = [
                "Code", "Département", "Sol dominant",
                "Temp. (°C)", "Score temp.", "Score sol",
                "Score global", "Catégorie",
            ]
            st.dataframe(df_detail, use_container_width=True, hide_index=True)


# =========================================================
# ONGLET 2 — Rendement vigne ML (via API)
# =========================================================

with tab2:
    st.header("Prédiction du rendement viticole")
    st.markdown(
        "Modèle XGBoost entraîné sur les rendements 2010-2024 (données SAA), "
        "les sols (dim_soils) et le climat ERA5/CORDEX."
    )

    try:
        df_hist = fetch_api("historique")
        df_proj = fetch_api("projections")
    except requests.RequestException as e:
        st.error(
            f"Impossible de joindre l'API à {API_URL}. "
            "Lance `python streamlit/api.py` dans un autre terminal."
        )
        st.exception(e)
        st.stop()

    st.markdown("#### Filtre des courbes")
    depts_dispo = sorted(df_hist["code_departement"].unique())
    options_filtre = ["🇫🇷 Moyenne nationale"] + [
        f"{c} - {DEPT_NAMES.get(c, 'Inconnu')}" for c in depts_dispo
    ]
    filtre = st.selectbox("Vue des courbes", options_filtre, key="filtre_onglet2")

    if filtre == "🇫🇷 Moyenne nationale":
        df_hist_filtre = df_hist.copy()
        df_proj_filtre = df_proj.copy() if not df_proj.empty else pd.DataFrame()
        label_filtre = "moyenne nationale"
    else:
        dept_filtre = filtre.split(" - ")[0]
        df_hist_filtre = df_hist[df_hist["code_departement"] == dept_filtre].copy()
        df_proj_filtre = (
            df_proj[df_proj["code_departement"] == dept_filtre].copy()
            if not df_proj.empty else pd.DataFrame()
        )
        label_filtre = f"département {dept_filtre}"

    st.subheader(f"Rendement historique 2010-2024 — {label_filtre}")

    if df_hist_filtre.empty:
        st.warning("Pas de données historiques pour ce filtre.")
    else:
        rendement_moyen = (
            df_hist_filtre.groupby("year", as_index=False)["rendement_q_ha"]
            .mean()
            .rename(columns={"rendement_q_ha": "Rendement moyen (q/ha)"})
        )
        fig_hist = px.line(
            rendement_moyen, x="year", y="Rendement moyen (q/ha)",
            markers=True, title=f"Évolution — {label_filtre}",
        )
        fig_hist.update_layout(height=380, hovermode="x unified")
        st.plotly_chart(fig_hist, use_container_width=True)

    st.subheader(f"Projections CORDEX 2054 / 2084 / 2100 — {label_filtre}")

    if df_proj_filtre.empty:
        st.warning("Pas de projections disponibles pour ce filtre.")
    else:
        synthese = (
            df_proj_filtre.groupby("year", as_index=False)[
                ["rendement_bas", "rendement_moyen", "rendement_haut"]
            ].mean()
        )

        point_2024 = (
            df_hist_filtre[df_hist_filtre["year"] == 2024]
            .groupby("year")["rendement_q_ha"].mean().reset_index()
        )
        if not point_2024.empty:
            p24 = pd.DataFrame([{
                "year": 2024,
                "rendement_bas": point_2024["rendement_q_ha"].values[0],
                "rendement_moyen": point_2024["rendement_q_ha"].values[0],
                "rendement_haut": point_2024["rendement_q_ha"].values[0],
            }])
            synthese = pd.concat([p24, synthese], ignore_index=True).sort_values("year")

        fig_proj = px.scatter(
            synthese, x="year", y="rendement_moyen",
            error_y=synthese["rendement_haut"] - synthese["rendement_moyen"],
            error_y_minus=synthese["rendement_moyen"] - synthese["rendement_bas"],
            title=f"Rendement projeté — {label_filtre}",
            labels={"rendement_moyen": "Rendement moyen (q/ha)", "year": "Horizon"},
        )
        fig_proj.update_xaxes(tickvals=[2024, 2054, 2084, 2100])
        fig_proj.update_traces(mode="lines+markers", marker={"size": 12})
        fig_proj.update_layout(height=380)
        st.plotly_chart(fig_proj, use_container_width=True)

    st.subheader("Simulateur de prédiction")
    st.markdown(
        "Choisis un département et un horizon. L'application précharge les "
        "données CORDEX correspondantes, que tu peux ajuster avant de prédire."
    )

    codes_dispo = sorted(set(df_hist["code_departement"]).union(set(df_proj["code_departement"]))) if not df_proj.empty else sorted(set(df_hist["code_departement"]))
    options_dept = [f"{c} - {DEPT_NAMES.get(c, 'Inconnu')}" for c in codes_dispo]

    col1, col2 = st.columns(2)

    with col1:
        dept_str = st.selectbox("Département", options_dept, key="sim_dept")
        dept = dept_str.split(" - ")[0]
        year_sim = st.selectbox("Année de projection", [2054, 2084, 2100], key="sim_year")

        if st.button("📥 Charger les données CORDEX", key="sim_load"):
            row = df_proj[(df_proj["code_departement"] == dept) & (df_proj["year"] == year_sim)]
            if not row.empty:
                r = row.iloc[0]
                st.session_state["sim_temp_slider"] = float(r["temp_saison"])
                st.session_state["sim_precip_slider"] = float(r["precip_cumul"])
                st.session_state["sim_solar_slider"] = float(r["solar_moy"])
                st.session_state["sim_family_slider"] = str(r["soil_family_dominant"])
                st.session_state["sim_type_slider"] = str(r["soil_type_dominant"])
                st.success(f"Données CORDEX chargées pour {dept} en {year_sim}.")
                st.rerun()
            else:
                st.warning("Pas de données CORDEX pour ce département/année. Ajuste manuellement.")

    with col2:
        st.session_state.setdefault("sim_temp_slider", 16.0)
        st.session_state.setdefault("sim_precip_slider", 500.0)
        st.session_state.setdefault("sim_solar_slider", 220.0)
        st.session_state.setdefault("sim_family_slider", "luvisols")
        st.session_state.setdefault("sim_type_slider", "limoneuse")

        temp = st.slider("Température saisonnière (°C)", 10.0, 26.0, step=0.1, key="sim_temp_slider")
        precip = st.slider("Précipitations cumulées (mm)", 50.0, 1300.0, step=10.0, key="sim_precip_slider")
        solar = st.slider("Rayonnement solaire moyen", 150.0, 280.0, step=1.0, key="sim_solar_slider")

        soil_families = ["calcaire", "argileux", "sableux", "limoneux", "luvisols", "rendzines"]
        soil_family = st.selectbox(
            "Famille de sol",
            soil_families,
            key="sim_family_slider",
        )

        soil_types = ["limoneuse", "argileuse", "sableuse", "rendzines", "luvisols"]
        soil_type = st.selectbox(
            "Type de sol",
            soil_types,
            key="sim_type_slider",
        )

    if st.button("🚀 Prédire le rendement", type="primary", key="sim_predict"):
        payload = {
            "temp_saison": float(temp),
            "precip_cumul": float(precip),
            "solar_moy": float(solar),
            "soil_family_dominant": str(soil_family),
            "soil_type_dominant": str(soil_type),
            "year": int(year_sim),
            "code_departement": str(dept),
        }
        with st.spinner("Calcul en cours..."):
            try:
                r = requests.post(f"{API_URL}/predict", json=payload, timeout=10)
                r.raise_for_status()
                result = r.json()
                pred = result["rendement_predit_q_ha"]

                # Bornes d'entraînement observées (source : training_vigne)
                TEMP_MIN, TEMP_MAX = 10.84, 21.51
                PRECIP_MIN, PRECIP_MAX = 92.1, 1119.7
                SOLAR_MIN, SOLAR_MAX = 170.9, 260.1

                hors_plage = []
                if not (TEMP_MIN <= temp <= TEMP_MAX):
                    hors_plage.append(f"température ({temp:.1f} °C hors plage {TEMP_MIN:.1f}–{TEMP_MAX:.1f} °C)")
                if not (PRECIP_MIN <= precip <= PRECIP_MAX):
                    hors_plage.append(f"précipitations ({precip:.0f} mm hors plage {PRECIP_MIN:.0f}–{PRECIP_MAX:.0f} mm)")
                if not (SOLAR_MIN <= solar <= SOLAR_MAX):
                    hors_plage.append(f"rayonnement ({solar:.0f} hors plage {SOLAR_MIN:.0f}–{SOLAR_MAX:.0f})")

                st.success(f"Rendement prédit pour **{dept}** en **{year_sim}** : **{pred} q/ha**")

                if hors_plage:
                    st.warning(
                        "⚠️ **Valeurs hors plage d'entraînement** : "
                        + " ; ".join(hors_plage)
                        + ". Le modèle XGBoost ne peut pas extrapoler au-delà des valeurs "
                        + "historiquement observées — sa prédiction reste bornée par ce qu'il "
                        + "a appris. À interpréter avec prudence."
                    )

                moy_hist_dept = df_hist[df_hist["code_departement"] == dept]["rendement_q_ha"].mean()
                if pd.notna(moy_hist_dept):
                    st.info(f"💡 Moyenne historique (2010-2024) pour {dept} : **{moy_hist_dept:.1f} q/ha**.")
                    if pred < moy_hist_dept * 0.9:
                        st.warning("🔍 Baisse par rapport à l'historique.")
                    elif pred > moy_hist_dept * 1.1:
                        st.warning("🔍 Hausse par rapport à l'historique.")
                    else:
                        st.write("🔍 Rendement similaire à l'historique.")
                else:
                    st.info("💡 Pas d'historique viticole pour ce département.")
            except requests.RequestException as e:
                st.error("Erreur lors de l'appel à l'API.")
                st.exception(e)


# =========================================================
# ONGLET 3 — À propos
# =========================================================

with tab3:
    st.header("À propos du modèle")
    st.markdown("""
    Ce projet utilise un modèle **XGBoost** entraîné sur les données de rendement
    viticole départemental de 2010 à 2024 (source SAA), croisées avec :

    - les **sols dominants** par département (`dim_soils`),
    - le **climat saisonnier** (avril-octobre) ERA5 pour l'historique,
    - les **projections CORDEX** pour 2054, 2084, 2100.

    **Comment utiliser le simulateur ?**

    1. Choisis un département et une année (2054, 2084, 2100).
    2. Clique sur **"Charger les données CORDEX"**. Les curseurs se placent sur
       les valeurs climatiques prévues par les scientifiques.
    3. Ajuste les curseurs si tu veux simuler un biais (ex. +2 °C).
    4. Clique sur **"Prédire"**.

    **Limites du modèle**

    - Le modèle repose majoritairement sur l'effet *terroir* (département).
      Les variables climatiques ont peu de poids dans ses prédictions.
    - XGBoost étant un modèle à base d'arbres, il **n'extrapole pas** : si les
      températures futures sortent de la plage observée historiquement, il
      maintient les rendements à un niveau moyen plutôt que de prédire une
      chute drastique.
    - Les projections 2054/2084/2100 doivent donc être lues comme des
      **tendances relatives** entre départements, pas comme des prédictions
      absolues de rendement.

    **La carte de favorabilité agronomique (onglet 1)**

    Complémentaire au modèle XGBoost, elle utilise un **score agronomique
    simplifié** (température + sol) et un slider de hausse de température pour
    visualiser la migration potentielle des zones favorables. Ce score ne
    prédit pas un rendement : il évalue une compatibilité climat/sol.

    **Sources de données**

    - Rendements : Agreste / SAA 2010-2025 (provisoire)
    - Sols : carte des sols agricoles de France
    - Climat passé : ERA5 (Copernicus)
    - Climat futur : CORDEX RCP 8.5, modèle CNRM-ALADIN63
    """)