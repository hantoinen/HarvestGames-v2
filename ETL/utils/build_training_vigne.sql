-- =============================================================================
-- build_training_vigne.sql
-- -----------------------------------------------------------------------------
-- Construction de la table d'entraînement du modèle ML vigne (2010-2024).
--
-- Source   : tables _v2 dans Neon v2 (projet harvestgames-v2)
-- Cible    : table training_vigne — 87 départements viticoles × années 2010-2024
-- Résultat : 1134 lignes, 0 valeur NULL sur rendement_q_ha
--
-- Prérequis : les 7 vues de compatibilité doivent exister avant d'exécuter
--             ce script (voir build_compat_views.sql) :
--               dim_constraints, dim_coordinates, dim_soils,
--               src_climate_past, src_climate_future,
--               src_yields_cordex_2010_2025, src_cultures_plots_2024
--
-- Ordre d'exécution : climate_maille_vigne → sol_dept → maille_vigne
--                     → training_vigne (table)
-- =============================================================================


-- -----------------------------------------------------------------------------
-- 1. Climat saisonnier par maille CORDEX (avril-octobre, moyennes 2010-2024)
-- -----------------------------------------------------------------------------
CREATE OR REPLACE VIEW climate_maille_vigne AS
SELECT coordinates_id, year,
       AVG(temp_mean_past) AS temp_saison,
       SUM(precip_past)    AS precip_cumul,
       AVG(solar_rad_past) AS solar_moy
FROM src_climate_past
WHERE month >= 4 AND month <= 10
GROUP BY coordinates_id, year;


-- -----------------------------------------------------------------------------
-- 2. Sol dominant par département (mode statistique sur les mailles)
-- -----------------------------------------------------------------------------
CREATE OR REPLACE VIEW sol_dept AS
SELECT
    CASE
        WHEN department_code ~ '^0[0-9]{2}$' AND length(department_code) = 3 THEN right(department_code, 2)
        WHEN department_code ~ '^0[0-9]$' THEN department_code
        WHEN department_code ~ '^[0-9]+$' AND length(department_code) <= 2 THEN lpad(department_code, 2, '0')
        WHEN department_code ~ '^[0-9]+\.0$' THEN lpad(replace(department_code, '.0', ''), 2, '0')
        WHEN department_code IN ('2A', '02A') THEN '2A'
        WHEN department_code IN ('2B', '02B') THEN '2B'
        ELSE department_code
    END AS department_code,
    MODE() WITHIN GROUP (ORDER BY soil_family_type) AS soil_family_dominant,
    MODE() WITHIN GROUP (ORDER BY soil_type)        AS soil_type_dominant
FROM dim_soils
WHERE soil_family_type IS NOT NULL
GROUP BY department_code;


-- -----------------------------------------------------------------------------
-- 3. Rendements × climat × sol, par maille CORDEX (vigne uniquement)
--
-- Note : dans src_yields_cordex_2010_2025_v2, la colonne annee est en
--        double precision (héritage d'un import pandas), d'où le cast ::int
--        pour matcher climate_maille_vigne.year (int).
--
-- Note : la colonne surface_ha correspond à la surface de vigne par maille.
--        Elle est aliasée en surface_ha_12_5km pour rester compatible avec
--        le nom utilisé dans la table training_vigne en aval.
-- -----------------------------------------------------------------------------
CREATE OR REPLACE VIEW maille_vigne AS
SELECT
    CASE
        WHEN y.code_departement ~ '^0[0-9]{2}$' AND length(y.code_departement) = 3 THEN right(y.code_departement, 2)
        WHEN y.code_departement ~ '^0[0-9]$' THEN y.code_departement
        WHEN y.code_departement ~ '^[0-9]+$' AND length(y.code_departement) <= 2 THEN lpad(y.code_departement, 2, '0')
        WHEN y.code_departement ~ '^[0-9]+\.0$' THEN lpad(replace(y.code_departement, '.0', ''), 2, '0')
        WHEN y.code_departement IN ('2A', '02A') THEN '2A'
        WHEN y.code_departement IN ('2B', '02B') THEN '2B'
        ELSE y.code_departement
    END AS code_departement,
    y.annee::int AS year,
    y.rendement_q_ha,
    y.surface_ha AS surface_ha_12_5km,
    c.temp_saison,
    c.precip_cumul,
    c.solar_moy,
    s.soil_family_type,
    s.soil_type
FROM src_yields_cordex_2010_2025 y
JOIN climate_maille_vigne c
    ON y.coordinates_id = c.coordinates_id
   AND y.annee::int     = c.year
LEFT JOIN dim_soils s
    ON y.coordinates_id = s.coordinates_id
WHERE y.culture_id = 1
  AND y.surface_ha > 0;


-- -----------------------------------------------------------------------------
-- 4. Table d'entraînement : agrégation départementale pondérée par la surface
--    Chaque ligne = (département, année) avec rendement moyen pondéré et
--    variables climatiques moyennées sur la surface viticole du département.
-- -----------------------------------------------------------------------------
DROP TABLE IF EXISTS training_vigne;

CREATE TABLE training_vigne AS
SELECT
    m.code_departement,
    m.year,
    SUM(m.rendement_q_ha * m.surface_ha_12_5km) / NULLIF(SUM(m.surface_ha_12_5km), 0) AS rendement_q_ha,
    SUM(m.temp_saison    * m.surface_ha_12_5km) / NULLIF(SUM(m.surface_ha_12_5km), 0) AS temp_saison,
    SUM(m.precip_cumul   * m.surface_ha_12_5km) / NULLIF(SUM(m.surface_ha_12_5km), 0) AS precip_cumul,
    SUM(m.solar_moy      * m.surface_ha_12_5km) / NULLIF(SUM(m.surface_ha_12_5km), 0) AS solar_moy,
    sd.soil_family_dominant,
    sd.soil_type_dominant
FROM maille_vigne m
JOIN sol_dept sd ON m.code_departement = sd.department_code
GROUP BY m.code_departement, m.year, sd.soil_family_dominant, sd.soil_type_dominant;


-- -----------------------------------------------------------------------------
-- 5. Contrôle post-exécution (à lancer manuellement pour vérifier)
--    Attendu : 87 départements, 1134 lignes, 2010-2024
-- -----------------------------------------------------------------------------
-- SELECT COUNT(DISTINCT code_departement) AS nb_departements,
--        COUNT(*)                         AS nb_lignes_total,
--        MIN(year)                        AS ymin,
--        MAX(year)                        AS ymax
-- FROM training_vigne;