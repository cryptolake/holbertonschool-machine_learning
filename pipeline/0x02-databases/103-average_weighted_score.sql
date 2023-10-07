-- Average weighted score 
DELIMITER //
CREATE PROCEDURE ComputeAverageWeightedScoreForUser (IN user_id INT)
BEGIN
  SELECT SUM(corrections.score * projects.weight) / SUM(projects.weight)
  FROM users
  INNER JOIN corrections ON users.id = corrections.user_id
  INNER JOIN projects ON projects.id = corrections.project_id
  WHERE
  users.id = user_id
  GROUP BY users.id INTO @user_avg;
  UPDATE users SET average_score = @user_avg WHERE id = user_id;
END //
DELIMITER ;
