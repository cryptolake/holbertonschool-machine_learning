-- SQL functions
DELIMITER //
CREATE FUNCTION SafeDiv (a int, b int) RETURNS float DETERMINISTIC
BEGIN
  IF b = 0 THEN
     RETURN 0;
  ELSE
    RETURN (a/b);
  END IF;
END //
DELIMITER ;
