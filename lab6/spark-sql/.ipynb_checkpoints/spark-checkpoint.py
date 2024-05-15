from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StructField, IntegerType, StringType, FloatType
from pyspark.sql.dataframe import DataFrame
from pyspark.sql.functions import col, when, isnan, isnull, count, avg, trim
import pandas as pd
import os

DATA_FOLDER = "data"
# source https://www.statista.com/statistics/242030/marital-status-of-the-us-population-by-sex/
# the first value is male and the second is for female
MARITAL_STATUS_BY_GENDER = [
    ["Never-married", 47.35, 41.81],
    ["Married-AF-spouse", 67.54, 68.33],
    ["Widowed", 3.58, 11.61],
    ["Divorced", 10.82, 15.09]
]
MARITAL_STATUS_BY_GENDER_COLUMNS = ["marital_status_statistics", "male", "female"]


def read_data(spark: SparkSession) -> DataFrame:
    """
    read data based on the given schema; this is much faster than spark determining the schema
    """
    
    # Define the schema for the dataset
    schema = StructType([
        StructField("age", IntegerType(), True),
        StructField("workclass", StringType(), True),
        StructField("fnlwgt", FloatType(), True),
        StructField("education", StringType(), True),
        StructField("education_num", FloatType(), True),
        StructField("marital_status", StringType(), True),
        StructField("occupation", StringType(), True),
        StructField("relationship", StringType(), True),
        StructField("race", StringType(), True),
        StructField("sex", StringType(), True),
        StructField("capital_gain", FloatType(), True),
        StructField("capital_loss", FloatType(), True),
        StructField("hours_per_week", FloatType(), True),
        StructField("native_country", StringType(), True),
        StructField("income", StringType(), True)
    ])

    # Read the dataset
    data = spark.read \
        .schema(schema) \
        .option("header", "false") \
        .option("inferSchema", "false") \
        .csv(os.path.join(DATA_FOLDER,"*.csv")) 

    data = data.repartition(8)

    float_columns = [f.name for f in data.schema.fields if isinstance(f.dataType, FloatType)]
    for v in float_columns:
        data = data.withColumn(v, data[v].cast(IntegerType()))

    # Get the names of all StringType columns
    string_columns = [f.name for f in data.schema.fields if isinstance(f.dataType, StringType)]

    # Remove leading and trailing spaces in all string columns
    for column in string_columns:
        data = data.withColumn(column, trim(data[column]))

    # Show the first 5 rows of the dataset
    data.show(5)

    return data

def missing_values(data: DataFrame) -> DataFrame:
    """
    count the number of samples with missing values for each row
    remove such samples
    """

    missing_values = data.select([count(when(isnan(c) | isnull(c), c)).alias(c) for c in data.columns])

    # Show the missing values count per column
    missing_values.show()

    # Get the number of samples in the DataFrame
    num_samples = data.count()

    # Print the number of samples
    print("Number of samples:", num_samples)  

    data = data.dropna()      
    
    return data

def feature_engineering(data: DataFrame) -> DataFrame:
    """
    calculate the product of each pair of integer features
    """

    # Create columns consisting of all products of columns of type IntegerType
    integer_columns = [f.name for f in data.schema.fields if isinstance(f.dataType, IntegerType)]
    for i, col1 in enumerate(integer_columns):
        for col2 in integer_columns[i:]:
            product_col_name = f"{col1}_x_{col2}"
            data = data.withColumn(product_col_name, col(col1) * col(col2))

    data.show(5)

    return data

def bias_marital_status(data: DataFrame):
    """
    is there bias in capital gain by marital status
    """

    # Calculate the average capital_gain by marital_status
    average_capital_gain = data.groupBy("marital_status").agg(avg("capital_gain").alias("average_capital_gain"))

    # Show the average capital_gain by marital_status
    average_capital_gain.show()

    # Filter data based on marital_status = Divorced
    divorced_data = data.filter(data.marital_status == "Divorced")

    # Show the first 5 rows of the filtered DataFrame
    divorced_data.show(5)

def join_with_US_gender(spark: SparkSession, data: DataFrame):
    """
    join with respect to the marital_status
    """

    # create a data frame from new data
    columns = ["dept_name","dept_id"]
    us_df = spark.createDataFrame(MARITAL_STATUS_BY_GENDER, MARITAL_STATUS_BY_GENDER_COLUMNS)

    return data.join(us_df, data.marital_status == us_df.marital_status_statistics, 'outer')



def bias_marital_status(data: DataFrame):
    """
    is there bias in capital gain by marital status
    """

    # Calculate the average capital_gain by marital_status
    average_capital_gain = data.groupBy("marital_status").agg(avg("capital_gain").alias("average_capital_gain"))

    # Show the average capital_gain by marital_status
    average_capital_gain.show()

    # Filter data based on marital_status = Divorced
    divorced_data = data.filter(data.marital_status == "Divorced")

    # Show the first 5 rows of the filtered DataFrame
    divorced_data.show(5)

def join_with_US_gender(spark: SparkSession, data: DataFrame):
    """
    join with respect to the marital_status
    """

    # create a data frame from new data
    columns = ["dept_name","dept_id"]
    us_df = spark.createDataFrame(MARITAL_STATUS_BY_GENDER, MARITAL_STATUS_BY_GENDER_COLUMNS)

    return data.join(us_df, data.marital_status == us_df.marital_status_statistics, 'outer')


def main():
    # Create a Spark session
    spark = SparkSession.builder \
        .appName("Read Adult Dataset") \
        .getOrCreate() 

    data = read_data(spark)
    # perform basic EDA - count missing values
    data = missing_values(data)
    data = feature_engineering(data)
    bias_marital_status(data)
    data = join_with_US_gender(spark, data)
    data = data.toPandas()
    data.to_csv("data/transformed_data.csv", header = True, index=False)
    

    spark.stop()
main()