-- Number of shows by genre
SELECT g.name 'genre', COUNT(s.show_id) 'number_of_shows' FROM tv_genres g INNER JOIN tv_show_genres s on g.id = s.genre_id GROUP BY g.name ORDER BY COUNT(s.show_id) DESC;
