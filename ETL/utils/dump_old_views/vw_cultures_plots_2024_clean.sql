 WITH parcelles_nettoyees AS (
         SELECT src_cultures_plots_2024.parcel_id,
            src_cultures_plots_2024.culture_id,
            src_cultures_plots_2024.coordinates_id,
            max(src_cultures_plots_2024.surf_ha) AS surf_ha
           FROM src_cultures_plots_2024
          WHERE src_cultures_plots_2024.parcel_id IS NOT NULL AND src_cultures_plots_2024.culture_id IS NOT NULL AND src_cultures_plots_2024.coordinates_id IS NOT NULL AND src_cultures_plots_2024.surf_ha > 0::double precision
          GROUP BY src_cultures_plots_2024.parcel_id, src_cultures_plots_2024.culture_id, src_cultures_plots_2024.coordinates_id
        )
 SELECT parcelles.parcel_id,
    parcelles.culture_id,
    contraintes.culture_name,
    parcelles.surf_ha,
    parcelles.coordinates_id
   FROM parcelles_nettoyees parcelles
     JOIN dim_constraints contraintes ON contraintes.culture_id = parcelles.culture_id;