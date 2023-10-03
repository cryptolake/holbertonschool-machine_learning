-- Create Procedure
DELIMITER //
CREATE PROCEDURE ComputeAverageScoreForUser (IN user_id INT)
BEGIN
  SELECT AVG(corrections.score) FROM users INNER JOIN corrections ON users.id = corrections.user_id WHERE users.id = user_id GROUP BY users.id INTO @user_avg;
  UPDATE users SET average_score = @user_avg WHERE id = user_id;
END //
DELIMITER ;
