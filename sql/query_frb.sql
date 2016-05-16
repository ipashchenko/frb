SELECT t1.t, t2.t, t1.dm, t2.dm, t1.antenna, t2.antenna

FROM

(SELECT *  FROM candidates
INNER JOIN searched_data on searched_data.id == candidates.searched_data_id) t1

JOIN
(SELECT *  FROM candidates
INNER JOIN searched_data on searched_data.id == candidates.searched_data_id) t2

ON t1.exp_code==t2.exp_code

WHERE
ABS(t1.dm - t2.dm) < 100 AND
t1.t < t2.t AND
t1.antenna != t2.antenna AND
ABS((julianday(t1.t)-julianday(t2.t))) * 24 * 60 * 60 < 0.1

