# Lab 6 Assignment

## Word Count
1 Save only the words that have count greater or equal to 3.

## Spark-sql
1, Add one more cell in ./spark-sql/pyspark-sql.ipynb that select rows with 'age' between 30 and 50 (inclusive) and transforms the selected pyspark dataframe into pandas dataframe and print out the summary statistics using 'describe()'.

2, Wrap all functions in the ./spark-sql/pyspark-sql.ipynb notebook into a .py file and write a run-py-spark.sh file (similar to the one in the word-counts folder). \
When you run 'bash run-py-spark.sh', the .py file should be executed with Spark (i.e including the steps of data reading data, data cleaning, data Transformation, but without the step in task 1 above), and the final dataframe should be stored as .csv file in the './data' folder.
