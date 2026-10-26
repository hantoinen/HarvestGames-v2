 SELECT
        CASE
            WHEN department_code ~ '^0[0-9]{2}$'::text AND length(department_code) = 3 THEN "right"(department_code, 2)
            WHEN department_code ~ '^0[0-9]$'::text THEN department_code
            WHEN department_code ~ '^[0-9]+$'::text AND length(department_code) <= 2 THEN lpad(department_code, 2, '0'::text)
            WHEN department_code ~ '^[0-9]+\.0$'::text THEN lpad(replace(department_code, '.0'::text, ''::text), 2, '0'::text)
            WHEN department_code = ANY (ARRAY['2A'::text, '02A'::text]) THEN '2A'::text
            WHEN department_code = ANY (ARRAY['2B'::text, '02B'::text]) THEN '2B'::text
            ELSE department_code
        END AS department_code,
    mode() WITHIN GROUP (ORDER BY soil_family_type) AS soil_family_dominant,
    mode() WITHIN GROUP (ORDER BY soil_type) AS soil_type_dominant
   FROM dim_soils
  WHERE soil_family_type IS NOT NULL
  GROUP BY department_code;