-- Shows by sum of ratings
SELECT s.title 'title', SUM(rate) 'rating' FROM tv_shows s INNER JOIN tv_show_ratings r on s.id = r.show_id GROUP BY s.title ORDER BY SUM(rate) DESC;
