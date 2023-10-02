-- Best Band
SELECT origin, SUM(fans) nb_fans FROM metal_bands GROUP BY origin ORDER BY SUM(fans) DESC;
