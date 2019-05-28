A = LOAD '/cloud/assignment1/*' using org.apache.pig.piggybank.storage.CSVExcelStorage(',','YES_MULTILINE','NOCHANGE','SKIP_INPUT_HEADER') as (id, score, viewcount, body, title, owneruserid,  displayname);
B = FOREACH A GENERATE id, score, viewcount,REPLACE (body, '\\n|\\r|\\t|<br>',' ') as body_mod,  REPLACE(title,'\\n|\\r|\\t|<br>',' ') as title_mod, owneruserid, displayname;
C = FOREACH B GENERATE id, score, viewcount,REPLACE (body_mod, '<[^>]*>','') as body_mod,  REPLACE(title_mod,'<[^>]*>','') as title_mod, owneruserid, displayname;
D = FOREACH C GENERATE id, score, viewcount,REPLACE (body_mod, '([^a-zA-Z\\s]+)',' ') as body,  REPLACE(title_mod,'([^a-zA-Z\\s]+)',' ') as title, owneruserid, displayname;

Store D into '/cloud/assignment1/resultset' USING PigStorage(',');