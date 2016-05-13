SELECT candidates.t, candidates2.t, ABS((julianday(candidates.t)-julianday(candidates2.t))) * 24 * 60 * 60, candidates.dm, candidates2.dm FROM candidates
JOIN searched_data on candidates.searched_data_id = searched_data.id
JOIN candidates as candidates2 on candidates2.searched_data_id == searched_data.id
JOIN searched_data as searched_data2 on candidates2.searched_data_id = searched_data2.id
WHERE searched_data.exp_code=='raks00' AND
searched_data2.exp_code=='raks00' AND
ABS(candidates.dm - candidates2.dm) < 50 AND
candidates.t < candidates2.t AND
ABS((julianday(candidates.t)-julianday(candidates2.t))) * 24 * 60 * 60 < 0.05