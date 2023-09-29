-- Genre id by show
SELECT s.title 'title', g.genre_id 'genre_id' FROM tv_shows s INNER JOIN tv_show_genres g on g.show_id = s.id ORDER BY s.title, g.genre_id ASC;
