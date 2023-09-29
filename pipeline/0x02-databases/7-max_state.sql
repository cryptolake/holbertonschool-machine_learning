-- Max temp in state
SELECT state, MAX(value) 'max_temp' FROM temperatures GROUP BY state;
