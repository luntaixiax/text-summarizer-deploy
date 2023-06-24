CREATE TABLE IF NOT EXISTS summarizer.batch_summarization (
	article_id INT auto_increment NOT NULL PRIMARY KEY,
	title varchar(1000) NOT NULL,
	article varchar(10000) NOT NULL,
	url varchar(1000) NULL,
	summary varchar(1000) NOT NULL
)
ENGINE=InnoDB
DEFAULT CHARSET=utf8mb4
COLLATE=utf8mb4_unicode_ci