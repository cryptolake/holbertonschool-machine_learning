-- Avg temp by city
SELECT city, AVG(value) 'avg_temp' FROM temperatures GROUP BY city ORDER BY AVG(value) DESC;
