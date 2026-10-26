 SELECT coordinates_id,
    year,
    avg(temp_mean_past) AS temp_saison,
    sum(precip_past) AS precip_cumul,
    avg(solar_rad_past) AS solar_moy
   FROM src_climate_past
  WHERE month >= 4 AND month <= 10
  GROUP BY coordinates_id, year;