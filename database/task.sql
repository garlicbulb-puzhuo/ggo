USE ggo;

DROP TABLE IF EXISTS task;

CREATE TABLE task
  (
     id           INT(10) UNSIGNED NOT NULL auto_increment,
     working_dir  VARCHAR(100) NOT NULL, # The working directory on s3
     state        enum('starting', 'complete','pending','processing', 'error') NOT NULL DEFAULT 'starting',
     worker       VARCHAR(200) NOT NULL, # The ec2 worker name
     created_time TIMESTAMP NOT NULL,
     last_updated TIMESTAMP NOT NULL,
     PRIMARY KEY (id)
  ) ENGINE=InnoDB AUTO_INCREMENT=8 DEFAULT CHARSET=latin1;
