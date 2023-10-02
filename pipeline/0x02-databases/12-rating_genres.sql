-- Best genre
SELECT tg.name 'name', SUM(tsr.rate) 'rating' FROM tv_genres tg INNER JOIN tv_show_genres tsg on tg.id = tsg.genre_id INNER JOIN tv_show_ratings tsr on tsr.show_id = tsg.show_id  GROUP BY tg.name  ORDER BY SUM(tsr.rate) DESC;
