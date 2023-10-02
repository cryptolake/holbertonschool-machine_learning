-- Old bans
SELECT band_name, IFNULL(split, 2020)-formed lifespan FROM metal_bands WHERE style LIKE '%Glam rock%' ORDER BY lifespan DESC;
