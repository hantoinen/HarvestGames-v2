 SELECT
        CASE
            WHEN y.code_departement ~ '^0[0-9]{2}$'::text AND length(y.code_departement) = 3 THEN "right"(y.code_departement, 2)
            WHEN y.code_departement ~ '^0[0-9]$'::text THEN y.code_departement
            WHEN y.code_departement ~ '^[0-9]+$'::text AND length(y.code_departement) <= 2 THEN lpad(y.code_departement, 2, '0'::text)
            WHEN y.code_departement ~ '^[0-9]+\.0$'::text THEN lpad(replace(y.code_departement, '.0'::text, ''::text), 2, '0'::text)
            WHEN y.code_departement = ANY (ARRAY['2A'::text, '02A'::text]) THEN '2A'::text
            WHEN y.code_departement = ANY (ARRAY['2B'::text, '02B'::text]) THEN '2B'::text
            ELSE y.code_departement
        END AS code_departement,
    y.year,
    y.rendement_q_ha,
    y.surface_ha_12_5km,
    c.temp_saison,
    c.precip_cumul,
    c.solar_moy,
    s.soil_family_type,
    s.soil_type
   FROM src_yields_cordex_2010_2025 y
     JOIN climate_maille_vigne c ON y.coordinates_id = c.coordinates_id AND y.year = c.year
     LEFT JOIN dim_soils s ON y.coordinates_id = s.coordinates_id
  WHERE y.culture_id = 1 AND y.surface_ha_12_5km > 0::double precision;