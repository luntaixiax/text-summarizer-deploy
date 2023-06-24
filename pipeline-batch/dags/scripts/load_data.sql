LOAD DATA LOCAL INFILE '{csv_file_path}' 
INTO TABLE summarizer.batch_summarization
COLUMNS TERMINATED BY ','
OPTIONALLY ENCLOSED BY '"'
ESCAPED BY '"'
LINES TERMINATED BY '\n'
IGNORE 1 LINES
(title, article, `url`, summary)