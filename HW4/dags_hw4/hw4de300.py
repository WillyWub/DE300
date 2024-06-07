from airflow import DAG
from airflow.operators.python_operator import PythonOperator
from datetime import datetime, timedelta
import pandas as pd
import boto3
from sklearn.linear_model import LogisticRegression as SklearnLogisticRegression
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pyspark
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, when, mean, pow
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.classification import LogisticRegression as SparkLogisticRegression, LinearSVC
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml import Pipeline
from pyspark.sql.functions import udf
from pyspark.sql.types import DoubleType
import logging

# Configuration for the DAG
default_arguments = {
    'owner': 'Wilson',
    'depends_on_past': False,
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=1),
}

# Initialize the DAG
dag = DAG(
    'WilsonHW4',
    default_args=default_arguments,
    description='WilsonHW4',
    schedule_interval=timedelta(days=1),
    start_date=datetime(2023, 1, 1),
    catchup=False,
)

# Load data from S3 function
def download_data_from_s3(**kwargs):
    s3_client = boto3.client('s3')
    bucket_name = 'lab7wilson'
    key = 'data/heart_disease.csv'
    local_file_path = '/tmp/heart_disease.csv'
    s3_client.download_file(bucket_name, key, local_file_path)
    logging.info(f"Downloaded {key} from {bucket_name} to {local_file_path}")
    return local_file_path

# Preprocessing and EDA function
def perform_eda_alternative(**kwargs):
    data = pd.read_csv("/tmp/heart_disease.csv").head(899)
    selected_columns = ['age', 'sex', 'painloc', 'painexer', 'cp', 'trestbps', 
                        'smoke', 'fbs', 'prop', 'nitr', 'pro', 'diuretic', 
                        'thaldur', 'thalach', 'exang', 'oldpeak', 'slope', 'target']
    df = data[selected_columns]

    # Impute missing values
    fill_with_mode = ['painloc', 'painexer', 'fbs', 'prop', 'nitr', 'pro', 'diuretic', 'exang', 'slope']
    for col in fill_with_mode:
        df[col].fillna(df[col].mode()[0], inplace=True)

    df['trestbps'] = df['trestbps'].apply(lambda x: max(x, 100))
    df['oldpeak'] = df['oldpeak'].apply(lambda x: min(max(x, 0), 4))
    df['thaldur'].fillna(df['thaldur'].mean(), inplace=True)
    df['thalach'].fillna(df['thalach'].mean(), inplace=True)

    # Further cleaning specific columns
    for col in ['fbs', 'prop', 'nitr', 'pro', 'diuretic', 'exang', 'slope']:
        mode_val = df[col].mode()[0]
        df[col] = df[col].apply(lambda x: x if x <= 1 else mode_val)

    # Handle continuous columns
    continuous_cols = ['trestbps', 'oldpeak', 'thaldur', 'thalach']
    skewness = df[continuous_cols].skew()
    for col in continuous_cols:
        fill_value = df[col].mean() if abs(skewness[col]) < 0.5 else df[col].median()
        df[col].fillna(fill_value, inplace=True)

    # Impute 'smoke' column using custom functions
    def smoking_percentage_1(age):
        age_brackets = [(15, 17, 0.016), (18, 24, 0.073), (25, 34, 0.109), 
                        (35, 44, 0.109), (45, 54, 0.138), (55, 64, 0.149), 
                        (65, 74, 0.087), (75, 150, 0.029)]
        for start, end, pct in age_brackets:
            if start <= age <= end:
                return pct
        return None

    def smoking_percentage_2(age, sex):
        female_brackets = [(18, 24, 0.053), (25, 44, 0.126), (45, 64, 0.149), (65, 150, 0.083)]
        male_factor = 0.131 / 0.101
        male_brackets = [(18, 24, 0.053 * male_factor), (25, 44, 0.126 * male_factor), 
                         (45, 64, 0.149 * male_factor), (65, 150, 0.083 * male_factor)]
        if sex == 0:
            brackets = female_brackets
        else:
            brackets = male_brackets
        for start, end, pct in brackets:
            if start <= age <= end:
                return pct
        return None

    df['smoking_src1'] = df.apply(lambda row: row['smoke'] if pd.notnull(row['smoke']) and row['smoke'] in [0, 1] else smoking_percentage_1(row['age']), axis=1)
    df['smoke_src2'] = df.apply(lambda row: row['smoke'] if pd.notnull(row['smoke']) and row['smoke'] in [0, 1] else smoking_percentage_2(row['age'], row['sex']), axis=1)
    
    df.drop(columns=['smoke'], inplace=True)
    
    df.to_csv("/tmp/heart_disease_subset.csv", index=False)
    logging.info("Saved cleaned data to /tmp/heart_disease_subset.csv")
    return "/tmp/heart_disease_subset.csv"


# Create a Spark session
def initiate_spark_session():
    return SparkSession.builder.appName("Heart Disease Analysis").getOrCreate()

# Spark preprocessing function
def spark_eda_alternative(input_path, output_path, **kwargs):
    spark = initiate_spark_session()
    data = spark.read.csv(input_path, header=True, inferSchema=True).limit(899)

    columns_to_keep = ['age', 'sex', 'painloc', 'painexer', 'cp', 'trestbps', 
                       'smoke', 'fbs', 'prop', 'nitr', 'pro', 'diuretic', 
                       'thaldur', 'thalach', 'exang', 'oldpeak', 'slope', 'target']
    df_cleaned = data.select(columns_to_keep)

    # Fill missing values with mode
    for col_name in ['painloc', 'painexer', 'fbs', 'prop', 'nitr', 'pro', 'diuretic', 'exang', 'slope']:
        mode_val = df_cleaned.groupBy(col_name).count().orderBy(col('count').desc()).first()[0]
        df_cleaned = df_cleaned.fillna({col_name: mode_val})

    # Handle specific column adjustments
    df_cleaned = df_cleaned.withColumn('trestbps', when(col('trestbps') < 100, 100).otherwise(col('trestbps')))
    df_cleaned = df_cleaned.withColumn('oldpeak', when(col('oldpeak') < 0, 0).when(col('oldpeak') > 4, 4).otherwise(col('oldpeak')))
    mean_thaldur = df_cleaned.agg(mean('thaldur')).first()[0]
    mean_thalach = df_cleaned.agg(mean('thalach')).first()[0]
    df_cleaned = df_cleaned.fillna({'thaldur': mean_thaldur, 'thalach': mean_thalach})

    for col_name in ['fbs', 'prop', 'nitr', 'pro', 'diuretic']:
        mode_val = df_cleaned.groupBy(col_name).count().orderBy(col('count').desc()).first()[0]
        df_cleaned = df_cleaned.withColumn(col_name, when(col(col_name) > 1, mode_val).otherwise(col(col_name)))

    # Impute 'smoke' column using custom functions
    def smoking_percentage_1(age):
        age_ranges = [(15, 17, 0.016), (18, 24, 0.073), (25, 34, 0.109), 
                      (35, 44, 0.109), (45, 54, 0.138), (55, 64, 0.149), 
                      (65, 74, 0.087), (75, 150, 0.029)]
        for start, end, pct in age_ranges:
            if start <= age <= end:
                return pct
        return None

    def smoking_percentage_2(age, sex):
        if sex == 0:  # Female
            female_ranges = [(18, 24, 0.053), (25, 44, 0.126), (45, 64, 0.149), (65, 150, 0.083)]
            for start, end, pct in female_ranges:
                if start <= age <= end:
                    return pct
        elif sex == 1:  # Male
            factor = 0.131 / 0.101
            male_ranges = [(18, 24, 0.053 * factor), (25, 44, 0.126 * factor), 
                           (45, 64, 0.149 * factor), (65, 150, 0.083 * factor)]
            for start, end, pct in male_ranges:
                if start <= age <= end:
                    return pct
        return None

    smoking_src1_udf = udf(smoking_percentage_1, DoubleType())
    smoking_src2_udf = udf(smoking_percentage_2, DoubleType())

    df_cleaned = df_cleaned.withColumn('smoking_src1', when(col('smoke').isin([0, 1]), col('smoke')).otherwise(smoking_src1_udf(col('age'))))
    df_cleaned = df_cleaned.withColumn('smoke_src2', when(col('smoke').isin([0, 1]), col('smoke')).otherwise(smoking_src2_udf(col('age'), col('sex'))))

    df_cleaned = df_cleaned.drop('smoke')
    df_cleaned.write.csv(output_path, header=True, mode='overwrite')
    logging.info(f"Saved cleaned Spark data to {output_path}")
    return output_path


# Spark feature engineering function 1
def spark_feature_engineering_1(input_path, output_path, **kwargs):
    spark = initiate_spark_session()
    data = spark.read.csv(input_path, header=True, inferSchema=True)
    
    # Create a new column 'age_squared'
    data_with_features = data.withColumn('age_squared', pow(col('age'), 2))
    
    # Generate the output path with '_fe1' suffix
    fe_output_path = output_path.replace('.csv', '_fe1.csv')
    data_with_features.write.csv(fe_output_path, header=True, mode='overwrite')
    
    logging.info(f"Saved feature engineered Spark data to {fe_output_path}")
    return fe_output_path

# Spark feature engineering function 2
def spark_feature_engineering_2(input_path, output_path, **kwargs):
    spark = initiate_spark_session()
    data = spark.read.csv(input_path, header=True, inferSchema=True)
    
    # Create a new column 'trestbps_sqrt'
    data_with_features = data.withColumn('trestbps_sqrt', pow(col('trestbps'), 0.5))
    
    # Generate the output path with '_fe2' suffix
    fe_output_path = output_path.replace('.csv', '_fe2.csv')
    data_with_features.write.csv(fe_output_path, header=True, mode='overwrite')
    
    logging.info(f"Saved feature engineered Spark data to {fe_output_path}")
    return fe_output_path

# Train SVM model using Spark
def spark_train_svm_model():
    spark = initiate_spark_session()
    data_path = '/tmp/heart_disease_subset_fe1.csv'
    data = spark.read.csv(data_path, header=True, inferSchema=True)

    # Assemble features into a single vector
    feature_columns = [column for column in data.columns if column != 'target']
    feature_assembler = VectorAssembler(inputCols=feature_columns, outputCol='features')

    # Split the data into training and test sets
    training_data, testing_data = data.randomSplit([0.9, 0.1], seed=42)

    # Define the SVM model
    svm_classifier = LinearSVC(labelCol='target', featuresCol='features', maxIter=100)

    # Create a pipeline
    svm_pipeline = Pipeline(stages=[feature_assembler, svm_classifier])

    # Train the model
    svm_model = svm_pipeline.fit(training_data)

    # Make predictions
    svm_predictions = svm_model.transform(testing_data)

    # Evaluate the model
    evaluator = MulticlassClassificationEvaluator(labelCol='target', metricName='accuracy')
    svm_accuracy = evaluator.evaluate(svm_predictions)
    logging.info(f"SVM Model Accuracy: {svm_accuracy:.4f}")
    print(f"SVM Model Accuracy: {svm_accuracy:.4f}")

# Train Logistic Regression model using Spark
def spark_train_logistic_model():
    spark = initiate_spark_session()
    data_path = '/tmp/heart_disease_subset_fe2.csv'
    data = spark.read.csv(data_path, header=True, inferSchema=True)

    # Assemble features into a single vector
    feature_columns = [column for column in data.columns if column != 'target']
    feature_assembler = VectorAssembler(inputCols=feature_columns, outputCol='features')

    # Split the data into training and test sets
    training_data, testing_data = data.randomSplit([0.9, 0.1], seed=42)

    # Define the Logistic Regression model
    logistic_classifier = SparkLogisticRegression(labelCol='target', featuresCol='features', maxIter=1000)

    # Create a pipeline
    logistic_pipeline = Pipeline(stages=[feature_assembler, logistic_classifier])

    # Train the model
    logistic_model = logistic_pipeline.fit(training_data)

    # Make predictions
    logistic_predictions = logistic_model.transform(testing_data)

    # Evaluate the model
    evaluator = MulticlassClassificationEvaluator(labelCol='target', metricName='accuracy')
    logistic_accuracy = evaluator.evaluate(logistic_predictions)
    logging.info(f"Logistic Regression Model Accuracy: {logistic_accuracy:.4f}")
    print(f"Logistic Regression Model Accuracy: {logistic_accuracy:.4f}")

# Feature engineering strategy 1
def feature_eng_1(file_path, **kwargs):
    logging.info(f"Loading dataset from {file_path}")
    df = pd.read_csv(file_path)

    # Feature engineering: Create squared age feature
    df['age_squared'] = df['age'].apply(lambda x: x ** 2)

    # Save the modified dataframe
    fe1_path = file_path.replace('.csv', '_fe1.csv')
    df.to_csv(fe1_path, index=False)
    logging.info(f"Feature-engineered data saved to {fe1_path}")
    return fe1_path

# Feature engineering strategy 2
def feature_eng_2(file_path, **kwargs):
    logging.info(f"Loading dataset from {file_path}")
    df = pd.read_csv(file_path)

    # Feature engineering: Create square root of trestbps feature
    df['trestbps_sqrt'] = df['trestbps'].apply(lambda x: x ** 0.5)

    # Save the modified dataframe
    fe2_path = file_path.replace('.csv', '_fe2.csv')
    df.to_csv(fe2_path, index=False)
    logging.info(f"Feature-engineered data saved to {fe2_path}")
    return fe2_path

# Function to train an SVM model
def train_svm(file_path, **kwargs):
    logging.info(f"Loading dataset from {file_path}")
    df = pd.read_csv(file_path)

    # Preparing features and target
    X = df.drop(columns='target')
    y = df['target']

    # Splitting the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

    # Initializing and training the model
    svm_model = SVC(random_state=42)
    svm_model.fit(X_train, y_train)

    # Making predictions and evaluating the model
    y_pred = svm_model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    logging.info(f"SVM Model Accuracy: {accuracy:.4f}")
    print(f"SVM Model Accuracy: {accuracy:.4f}")

# Function to train a Logistic Regression model
def train_logistic(file_path, **kwargs):
    logging.info(f"Loading dataset from {file_path}")
    df = pd.read_csv(file_path)

    # Preparing features and target
    X = df.drop(columns='target')
    y = df['target']

    # Splitting the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

    # Initializing and training the model
    logistic_model = SklearnLogisticRegression(max_iter=1000, random_state=42)
    logistic_model.fit(X_train, y_train)

    # Making predictions and evaluating the model
    y_pred = logistic_model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    logging.info(f"Logistic Regression Model Accuracy: {accuracy:.4f}")
    print(f"Logistic Regression Model Accuracy: {accuracy:.4f}")

# Define Airflow tasks
load_data_task = PythonOperator(
    task_id='fetch_data_from_s3',
    python_callable=download_data_from_s3,
    dag=dag,
)

eda_task = PythonOperator(
    task_id='perform_eda',
    python_callable=perform_eda_alternative,
    provide_context=True,
    dag=dag,
)

spark_eda_task = PythonOperator(
    task_id='spark_eda',
    python_callable=spark_eda_alternative,
    op_kwargs={'input_path': '/tmp/heart_disease.csv', 'output_path': '/tmp/heart_disease_subset_eda.csv'},
    dag=dag,
)

fe1_task = PythonOperator(
    task_id='feature_eng_1',
    python_callable=feature_eng_1,
    provide_context=True,
    op_kwargs={'file_path': '/tmp/heart_disease_subset.csv'},
    dag=dag,
)

fe2_task = PythonOperator(
    task_id='feature_eng_2',
    python_callable=feature_eng_2,
    provide_context=True,
    op_kwargs={'file_path': '/tmp/heart_disease_subset.csv'},
    dag=dag,
)

spark_fe1_task = PythonOperator(
    task_id='spark_feature_eng_1',
    python_callable=spark_feature_engineering_1,
    op_args=['/tmp/heart_disease_subset_eda.csv', '/tmp/heart_disease_subset_fe1.csv'],
    dag=dag,
)

spark_fe2_task = PythonOperator(
    task_id='spark_feature_eng_2',
    python_callable=spark_feature_engineering_2,
    op_args=['/tmp/heart_disease_subset_eda.csv', '/tmp/heart_disease_subset_fe2.csv'],
    dag=dag,
)

train_svm_task = PythonOperator(
    task_id='train_svm',
    python_callable=train_svm,
    provide_context=True,
    op_kwargs={'file_path': '/tmp/heart_disease_subset_fe1.csv'},
    dag=dag,
)

train_logistic_task = PythonOperator(
    task_id='train_logistic',
    python_callable=train_logistic,
    provide_context=True,
    op_kwargs={'file_path': '/tmp/heart_disease_subset_fe2.csv'},
    dag=dag,
)

spark_train_svm_task = PythonOperator(
    task_id='spark_train_svm_model',
    python_callable=spark_train_svm_model,
    dag=dag,
)

spark_train_logistic_task = PythonOperator(
    task_id='spark_train_logistic_model',
    python_callable=spark_train_logistic_model,
    dag=dag,
)

# Define task dependencies
load_data_task >> [eda_task, spark_eda_task]

eda_task >> [fe1_task, fe2_task]
fe1_task >> train_svm_task
fe2_task >> train_logistic_task

spark_eda_task >> [spark_fe1_task, spark_fe2_task]
spark_fe1_task >> spark_train_svm_task
spark_fe2_task >> spark_train_logistic_task
