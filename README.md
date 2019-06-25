# DCU Cloud project

### 1. Large dataset Analysis
> “Stack Exchange is a network of question and answer websites on diverse topics in many different fields, each site covering a specific topic, where questions, answers, and users are subject to a reputation award process. The sites are modeled after Stack Overflow, a forum for computer programming questions that was the original site in this network.”

Stack Exchange Data Explorer (SEDE) https://data.stackexchange.com/stackoverflow/query/new

[Task 1] Data Acquisition: 
 - We are required to acquire the top 200,000 posts by viewcount from the Stack Exchange site. Problem is that we can only download 50.000 records at a time. 

[Task 2] Data Cleaning with PIG
 - Extract, transform and load the data as applicable

[Task 3] Querying with HIVE
 - 1. The top 10 posts by score ?
 - 2. The top 10 users by score ?
 - 3. The number of distinct users, who used the word ‘hadoop’ in one of their posts ? 

[Task 4]  Calculate the per-user TF-IDF with HIVE 
 - Find Top 10 terms used for each of the top 10 users by post score

### 2. Data Modeling with Postgres
> __Introduction:__ A startup called Sparkify wants to analyze the data they've been collecting on songs and user activity on their new music streaming app. The analytics team is particularly interested in understanding **what songs users are listening to**. 
 - > Currently, they don't have an easy way to query their data, which resides in a directory of JSON logs on user activity on the app, as well as a directory with JSON metadata on the songs in their app.
 - __Task:__ Create a database schema and ETL pipeline for this analysis. You'll be able to test your database and ETL pipeline by running queries given to you by the analytics team from Sparkify and compare your results with their expected results. you need to define fact and dimension tables for a star schema for a particular analytic focus, and write an ETL pipeline that transfers data from files in two local directories into these tables in Postgres. 
 - __Dataset:__
   - 1. Song Dataset(http://millionsongdataset.com/): The first dataset is a subset of real data from the Million Song Dataset. Each file is in JSON format and contains metadata about a song and the artist of that song. The files are partitioned by the first three letters of each song's track ID.
   - 2. Log Dataset(https://github.com/Interana/eventsim): The second dataset consists of log files in JSON format generated by this event simulator based on the songs in the dataset above. These simulate activity logs from a music streaming app based on specified configurations. The log files in the dataset you'll be working with are partitioned by year and month. 
 - __files:__
   - `test.ipynb` displays the first few rows of each table to let you check your database.
   - `create_tables.py` drops and creates your tables. You run this file to reset your tables before each time you run your ETL scripts.
   - `etl.ipynb` reads and processes a single file from song_data and log_data and loads the data into your tables. This notebook contains detailed instructions on the ETL process for each of the tables.
   - `etl.py` reads and processes files from song_data and log_data and loads them into your tables. You can fill this out based on your work in the ETL notebook.
   - `sql_queries.py` contains all your sql queries, and is imported into the last three files above.
 - __Schema:__
   - Using the song and log datasets, you'll need to create a star schema optimized for queries on song play analysis. This includes the following tables.
     - Fact Table
       - songplays - records in log data associated with song plays i.e. records with page `NextSong`
         - songplay_id, start_time, user_id, level, song_id, artist_id, session_id, location, user_agent
     - Dimension Tables
       - users - users in the app
         - user_id, first_name, last_name, gender, level
       - songs - songs in music database
         - song_id, title, artist_id, year, duration
       - artists - artists in music database
         - artist_id, name, location, latitude, longitude
       - time - timestamps of records in songplays broken down into specific units
         - start_time, hour, day, week, month, year, weekday
 - __Steps to follow:__
   - 1. Create Tables
     - Write `CREATE` statements in `sql_queries.py` to create each table.
     - Write `DROP` statements in `sql_queries.py` to drop each table if it exists.
     - Run `create_tables.py` to create your database and tables.
     - Run `test.ipynb` to confirm the creation of your tables with the correct columns. Make sure to click "Restart kernel" to close the connection to the database after running this notebook.
   - 2. Build ETL Processes
     - Follow instructions in the `etl.ipynb` notebook to develop ETL processes for each table. At the end of each table section, or at the end of the notebook, run `test.ipynb` to confirm that records were successfully inserted into each table. Remember to rerun `create_tables.py` to reset your tables before each time you run this notebook.
   - 3. Build ETL Pipeline
     - Use what you've completed in `etl.ipynb` to complete `etl.py`, where you'll process the entire datasets. Remember to run `create_tables.py` before running `etl.py` to reset your tables. Run `test.ipynb` to confirm your records were successfully inserted into each table.

### 3. Data Modeling with Cassandra
> __Introduction:__ A startup called Sparkify wants to analyze the data they've been collecting on songs and user activity on their new music streaming app. The analytics team is particularly interested in understanding **what songs users are listening to**. 
 - > Currently, there is no easy way to query the data to generate the results, since the data reside in a directory of CSV files on user activity on the app. 
 - __Dataset:__
   - For this project, you'll be working with one dataset: `event_data`. The directory of CSV files partitioned by date. 
 - __Steps to follow:__
   - 1. Modeling your NoSQL database
     - Design tables to answer the queries outlined in the project template
     - Write Apache Cassandra `CREATE KEYSPACE and `SET KEYSPACE` statements
     - Develop your `CREATE` statement for each of the tables to address each question
     - Load the data with `INSERT` statement for each of the tables
     - Include `IF NOT EXISTS` clauses in your **CREATE statements** to create tables only if the tables do not already exist. We recommend you also include **DROP TABLE statement** for each table, this way you can run drop and create tables whenever you want to reset your database and test your ETL pipeline
     - Test by running the proper select statements with the correct `WHERE clause`
   - 2. Build ETL Pipeline
     - Implement the logic in section Part I of the notebook template to iterate through each event file in `event_data` to process and create a new CSV file in Python
     - Make necessary edits to Part II of the notebook template to include Apache Cassandra `CREATE` and `INSERT` statements to load processed records into relevant tables in your data model
     - Test by running `SELECT` statements after running the queries on your database






































































