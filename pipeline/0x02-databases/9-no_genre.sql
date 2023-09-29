-- shows without genre
SELECT s.title 'title', g.genre_id 'genre_id' FROM tv_shows s LEFT JOIN tv_show_genres g on s.id = g.show_id WHERE g.show_id IS NULL ORDER BY s.title, g.genre_id ASC;
