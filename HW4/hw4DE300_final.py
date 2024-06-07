from airflow import DAG
from airflow.operators.python_operator import PythonOperator, BranchPythonOperator
from airflow.operators.dummy_operator import DummyOperator
from airflow.utils.dates import days_ago
from airflow.hooks.base_hook import BaseHook
import boto3
import pandas as pd
from io import StringIO
import tomli
import pathlib
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import FunctionTransformer
from sklearn.preprocessing import StandardScaler
from scipy import stats



# Define the default args dictionary for DAG
default_args = {
    'owner': 'Wilson',
    'depends_on_past': False,
    'start_date': days_ago(1),
    'retries': 1,
}

def reads3() -> dict:
    s3_client = boto3.client('s3')
    obj = s3_client.get_object(Bucket="lab7wilson", Key="HW4/hw4config.toml")
    config_data = obj['Body'].read().decode('utf-8')
    params = tomli.loads(config_data)
    return params


PARAMS = reads3()


def load_data():

    # Connect to S3
    s3 = boto3.client('s3')

    # Specify the bucket and file key
    bucket_name = "lab7wilson"
    file_key = "HW4/heart_disease(in).csv"

    # Download the file from S3
    response = s3.get_object(Bucket=bucket_name, Key=file_key)
    csv_content = response['Body'].read().decode('utf-8')
    
    # Read the CSV content into a DataFrame
    df = pd.read_csv(StringIO(csv_content))

    return df


def clean_impute_sklearn(**kwargs):

    # Retrieve the DataFrame from the context
    ti = kwargs['ti']
    df = ti.xcom_pull(task_ids='load_data')

    # Trim the DataFrame to remove unnecessary rows
    df = df.iloc[:899, :]

    # Rename the incorrectly named column
    df.rename(columns={"ekgday(day": "ekgday"}, inplace=True)

    # Fill missing values in categorical columns with the mode
    categorical_cols = ["painloc", "painexer", "relrest", "cp", "sex", "htn", "smoke",
                        "fbs", "dm", "famhist", "restecg", "dig", "prop", "nitr",
                        "pro", "diuretic", "proto", "exang", "xhypo", "slope", "ca",
                        "exerckm", "restwm", "exerwm", "thal", "thalsev", "thalpul", "earlobe"]
    for col in categorical_cols:
        df[col].fillna(df[col].mode()[0], inplace=True)

    # Update 'pncaden' column
    df['pncaden'] = df[['painloc', 'painexer', 'relrest']].sum(axis=1)

    # Replace missing values in 'restckm' with zero
    df['restckm'].fillna(0, inplace=True)

    # Impute missing numeric values with the mean
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    imputer = SimpleImputer(strategy='mean')
    df[numeric_cols] = imputer.fit_transform(df[numeric_cols])

    # Remove outliers using the Z-score method
    z_scores = stats.zscore(df[numeric_cols], nan_policy='omit')
    threshold = 6
    outliers_mask = (np.abs(z_scores) > threshold).any(axis=1)
    df_cleaned = df.loc[~outliers_mask]

    return df_cleaned

def feature_eng_1(**kwargs):

    # Retrieve the DataFrame from the context
    ti = kwargs['ti']
    df_sklearn_clean = ti.xcom_pull(task_ids='clean_impute_sklearn')

    # Apply log transformation to specified columns
    log_cols = ["cigs", "years", "rldv5e"]
    df_sklearn_clean[log_cols] = FunctionTransformer(np.log1p).transform(df_sklearn_clean[log_cols])

    # Round all numeric columns to 2 decimal places
    numeric_cols = df_sklearn_clean.select_dtypes(include=[np.number]).columns
    df_sklearn_clean[numeric_cols] = df_sklearn_clean[numeric_cols].round(2)

    return df_sklearn_clean


def clean_impute_spark(**kwargs):
    from pyspark.sql import SparkSession
    from pyspark.sql.functions import col, when, lit, avg
    from pyspark.sql.types import StructType, StructField, StringType

    # Retrieve the DataFrame from the context
    ti = kwargs['ti']
    df_pandas = ti.xcom_pull(task_ids='load_data')

    # Initialize Spark session
    spark = SparkSession.builder.appName("HeartDiseaseEDA").getOrCreate()

    # Convert Pandas DataFrame to Spark DataFrame
    schema = StructType([StructField(col_name, StringType(), True) for col_name in df_pandas.columns])
    df = spark.createDataFrame(df_pandas, schema=schema)

    # Limit DataFrame to first 899 rows and rename the column
    df = df.limit(899).withColumnRenamed("ekgday(day", "ekgday")

    # Fill missing values with mode for specified columns
    def fill_with_mode(dataframe, column):
        mode = dataframe.filter(col(column).isNotNull()) \
                        .groupBy(column).count() \
                        .orderBy('count', ascending=False).first()[0]
        return dataframe.withColumn(column, when(col(column).isNull(), lit(mode)).otherwise(col(column)))

    mode_columns = ['painloc', 'painexer', 'exang', 'slope']
    for col_name in mode_columns:
        df = fill_with_mode(df, col_name)

    # Replace outliers and missing values with average for specified columns
    def replace_outliers_with_avg(dataframe, column, lower_bound, upper_bound=None):
        mean_val = dataframe.filter(col(column).isNotNull()).select(avg(col(column))).first()[0]
        condition = (col(column) < lower_bound) if upper_bound is None else (col(column) < lower_bound) | (col(column) > upper_bound)
        return dataframe.withColumn(column, when(condition | col(column).isNull(), lit(mean_val)).otherwise(col(column)))

    df = replace_outliers_with_avg(df, 'trestbps', 100)
    df = replace_outliers_with_avg(df, 'oldpeak', 0, 4)

    avg_columns = ['thaldur', 'thalach']
    for col_name in avg_columns:
        mean_val = df.filter(col(col_name).isNotNull()).select(avg(col(col_name))).first()[0]
        df = df.withColumn(col_name, when(col(col_name).isNull(), lit(mean_val)).otherwise(col(col_name)))

    # Fill missing values and replace outliers with mode for specific columns
    binary_columns = ['fbs', 'prop', 'nitr', 'pro', 'diuretic']
    for col_name in binary_columns:
        mode_val = df.filter((col(col_name).isNotNull()) & (col(col_name) <= 1)).groupBy(col_name).count().orderBy('count', ascending=False).first()[0]
        df = df.withColumn(col_name, when((col(col_name).isNull()) | (col(col_name) > 1), lit(mode_val)).otherwise(col(col_name)))

    # Drop 'smoke' column and select specified columns
    selected_columns = ['age', 'sex', 'painloc', 'painexer', 'cp', 'trestbps', 'fbs', 'prop', 'nitr', 'pro', 'diuretic', 
                        'thaldur', 'thalach', 'exang', 'oldpeak', 'slope', 'target']
    df_cleaned = df.drop('smoke').select(*selected_columns)

    # Convert the cleaned Spark DataFrame to Pandas DataFrame and push to XCom
    df_cleaned_pandas = df_cleaned.toPandas()
    ti.xcom_push(key='df_clean', value=df_cleaned_pandas)

    # Stop Spark session
    spark.stop()

    return df_cleaned_pandas



def feature_eng_2(**kwargs):
    from pyspark.sql import SparkSession
    from pyspark.sql.functions import col
    from pyspark.ml.feature import VectorAssembler, StandardScaler
    from pyspark.sql.types import StructType, StructField, StringType, DoubleType
    import pandas as pd

    # Retrieve the DataFrame from the context
    ti = kwargs['ti']
    df_dict = ti.xcom_pull(task_ids='clean_impute_spark', key='df_clean')

    # Initialize Spark session
    spark = SparkSession.builder.appName("FeatureEngineering").getOrCreate()

    # Convert the Pandas DataFrame to Spark DataFrame
    schema = StructType([StructField(col_name, StringType(), True) for col_name in df_dict.columns])
    df_spark_clean = spark.createDataFrame(df_dict, schema=schema)

    # Cast string columns to DoubleType
    df_spark_clean = df_spark_clean.select(
        *[col(col_name).cast(DoubleType()).alias(col_name) for col_name in df_spark_clean.columns]
    )

    # Define feature columns
    feature_columns = ['age', 'sex', 'painloc', 'painexer', 'cp', 'trestbps', 'fbs',
                       'prop', 'nitr', 'pro', 'diuretic', 'thaldur', 'thalach', 'exang',
                       'oldpeak', 'slope']

    # Select and filter the DataFrame
    df_spark_features = df_spark_clean.select(*feature_columns, 'target').na.drop()

    # Assemble features into a feature vector
    assembler = VectorAssembler(inputCols=feature_columns, outputCol="features")
    df_assembled = assembler.transform(df_spark_features)

    # Standardize features
    scaler = StandardScaler(inputCol="features", outputCol="scaled_features", withStd=True, withMean=True)
    df_fe2 = scaler.fit(df_assembled).transform(df_assembled)

    # Convert the PySpark DataFrame to a Pandas DataFrame
    df_fe2_pandas = df_fe2.toPandas()

    # Push the cleaned DataFrame to XCom
    ti.xcom_push(key='df_fe2', value=df_fe2_pandas)

    # Stop the Spark session
    spark.stop()

    return df_fe2_pandas


def web_scrape():
    import re
    import urllib.request

    # Function to extract content from a URL
    def fetch_html(url):
        response = urllib.request.urlopen(url)
        return response.read().decode('utf-8')

    # Extract content from source 1 (Australia data)
    url1 = 'https://www.abs.gov.au/statistics/health/health-conditions-and-risks/smoking-and-vaping/latest-release'
    html1 = fetch_html(url1)

    # Use regex to find the second table in the HTML
    table_pattern = re.compile(r'<table class="responsive-enabled".*?>(.*?)</table>', re.DOTALL)
    tables = table_pattern.findall(html1)
    table1_html = tables[1] if len(tables) >= 2 else None

    if not table1_html:
        raise ValueError("Unable to find the required table in the HTML content.")

    # Initialize list to store smoking rates
    smoking_rates = []

    # Regex patterns for extracting data
    row_pattern = re.compile(r'<tr.*?>(.*?)</tr>', re.DOTALL)
    col_pattern = re.compile(r'<td.*?>(.*?)</td>', re.DOTALL)
    age_pattern = re.compile(r'<th scope="row" class="row-header">(.*?)</th>', re.DOTALL)

    # Extract data from each row
    for i in range(1, 8):
        row_html = row_pattern.findall(table1_html)[i]
        cols = col_pattern.findall(row_html)
        if len(cols) >= 10:
            smoke_rate = float(cols[9])
        else:
            print(f"Error: Insufficient columns in row {i}")
            continue

        age_range = age_pattern.findall(row_html)[0]
        age_start, age_end = map(int, age_range.replace('–', '-').split('-'))
        age_to_smoke_rate = {age: smoke_rate for age in range(age_start, age_end + 1)}
        smoking_rates.append(age_to_smoke_rate)

    # Add senior age group data
    senior_row_html = row_pattern.findall(table1_html)[8]
    senior_age_match = re.search(r'<th scope="row" class="row-header">(\d+) years and over</th>', senior_row_html)
    senior_age = senior_age_match.group(1) if senior_age_match else None

    senior_smoke_rate_match = re.findall(r'<td class="data-value">([\d.]+)</td>', senior_row_html)
    senior_smoke_rate = senior_smoke_rate_match[9] if senior_smoke_rate_match else None

    # Extract content from source 2 (US data)
    url2 = 'https://www.cdc.gov/tobacco/data_statistics/fact_sheets/adult_data/cig_smoking/index.htm'
    html2 = fetch_html(url2)

    # Use regex to find the block list in the HTML
    block_list_pattern = re.compile(r'<ul class="block-list">(.*?)</ul>', re.DOTALL)
    block_lists = block_list_pattern.findall(html2)
    desired_lines = re.findall(r'<li>(.*?)</li>', block_lists[1], re.DOTALL)[:4]

    def extract_numbers(text):
        numbers = []
        for sentence in text:
            words = sentence.split()
            for word in words:
                if word.isdigit():
                    numbers.append(word)
                elif '&ndash;' in word:
                    ages = word.split('&ndash;')
                    numbers.extend(ages)
                elif '%' in word:
                    numbers.append(word.strip('()%'))
        return numbers

    pattern = r'\b(?:About|Nearly)\s*(\d+)\s*of every\s*(\d+)\s*adults aged (\d+)&ndash;(\d+) years\s*\((\d+\.\d+)%\)'
    
    def split_desired_line(line):
        return [section for section in re.split(pattern, line) if section]

    group1_data = split_desired_line(desired_lines[0])
    group2_data = split_desired_line(desired_lines[1])
    group3_data = split_desired_line(desired_lines[2])
    group4_data = extract_numbers(split_desired_line(desired_lines[3]))

    return smoking_rates, senior_age, senior_smoke_rate, group1_data, group2_data, group3_data, group4_data, html2

def merge(**kwargs):

    from pyspark.sql.functions import when, col, lit
    from pyspark.sql import SparkSession
    from pyspark.sql.types import StructType, StructField, StringType, DoubleType
    import pandas as pd
    import re
    import urllib.request

    ti = kwargs['ti']
    
    df_fe2 = ti.xcom_pull(task_ids='feature_eng_2')
    smoking_rates, senior_age, senior_smoke_rate, group1_data, group2_data, group3_data, group4_data, html2 = web_scrape()

    # Convert to Spark DataFrame
    spark = SparkSession.builder.appName("Merge").getOrCreate()

    # Convert the Pandas DataFrame to a Spark DataFrame
    schema = StructType([StructField(col_name, StringType(), True) for col_name in df_fe2.columns])
    df_fe2 = spark.createDataFrame(df_fe2, schema=schema)

    # Update DataFrame with smoking rates from source 1
    df_fe2 = df_fe2.withColumn('smokesource1', lit(None).cast('double'))
    df_fe2 = df_fe2.withColumn('smokesource2', lit(None).cast('double'))

    for age_to_smoke_rate in smoking_rates:
        for age, smoking_rate in age_to_smoke_rate.items():
            df_fe2 = df_fe2.withColumn('smokesource1', when(col('age') == age, lit(smoking_rate)).otherwise(col('smokesource1')))
    if 'senior_age' in locals() and 'senior_smoke_rate' in locals():
        df_fe2 = df_fe2.withColumn('smokesource1', when(col('age') >= int(senior_age), lit(float(senior_smoke_rate))).otherwise(col('smokesource1')))

    # Function to update DataFrame with smoking rates based on age groups
    def update_smoking_rates(df, age_start, age_end, smoking_rate):
        return df.withColumn('smokesource2',
                             when((col('sex') == 0) & (col('age') >= age_start) & (col('age') <= age_end), smoking_rate)
                             .otherwise(col('smokesource2')))

    # Update DataFrame with smoking rates from source 2
    def process_age_group(group_data, df):
        age_start, age_end = int(group_data[2]), int(group_data[3])
        smoking_rate = float(group_data[4])
        return update_smoking_rates(df, age_start, age_end, smoking_rate)

    df_fe2 = process_age_group(group1_data, df_fe2)
    df_fe2 = process_age_group(group2_data, df_fe2)
    df_fe2 = process_age_group(group3_data, df_fe2)

    # Age group 4 is slightly different
    age_start4 = int(group4_data[2])
    smoking_rate4 = float(group4_data[3])
    df_fe2 = df_fe2.withColumn('smokesource2',
                               when((col('sex') == 0) & (col('age') >= age_start4), smoking_rate4)
                               .otherwise(col('smokesource2')))

    # Extract smoke rates for men and women from source 2
    block_list_gender_text = re.findall(r'<ul class="block-list">(.*?)</ul>', html2, re.DOTALL)[0]
    men_smoke_rate = float(re.findall(r'<li class="main">.*men \((.*?)%\)</li>', block_list_gender_text)[0])
    women_smoke_rate = float(re.findall(r'<li class="main">.*women \((.*?)%\)</li>', block_list_gender_text)[0])
    gender_ratio = men_smoke_rate / women_smoke_rate

    men_smoke_rates = [float(group1_data[4]) * gender_ratio,
                       float(group2_data[4]) * gender_ratio,
                       float(group3_data[4]) * gender_ratio,
                       float(group4_data[3]) * gender_ratio]

    # Function to update DataFrame with smoking rates for males based on age groups
    def update_smoking_rates_males(df, age_start, age_end, smoking_rate):
        return df.withColumn('smokesource2',
                             when((col('sex') == 1) & (col('age') >= age_start) & (col('age') <= age_end), smoking_rate)
                             .otherwise(col('smokesource2')))

    age_groups = [(int(group1_data[2]), int(group1_data[3])),
                  (int(group2_data[2]), int(group2_data[3])),
                  (int(group3_data[2]), int(group3_data[3]))]

    for i, (age_start, age_end) in enumerate(age_groups):
        df_fe2 = update_smoking_rates_males(df_fe2, age_start, age_end, men_smoke_rates[i])

    # Age group 4 is slightly different
    df_fe2 = df_fe2.withColumn('smokesource2',
                               when((col('sex') == 1) & (col('age') >= age_start4), men_smoke_rates[3])
                               .otherwise(col('smokesource2')))

    # Calculate average of smokesource1 and smokesource2 columns and replace smoke column
    df_fe2 = df_fe2.withColumn('smoke', (col('smokesource1') + col('smokesource2')) / 2)

    # Drop smokesource1 and smokesource2 columns
    df_fe2 = df_fe2.drop('smokesource1', 'smokesource2')

    # Convert the cleaned Spark DataFrame to a Pandas DataFrame for XCom serialization
    df_merge = df_fe2.toPandas()

    # Stop the Spark session
    spark.stop()

    # Push the cleaned DataFrame as a Pandas DataFrame to XCom
    ti.xcom_push(key='df_scrape_merge', value=df_merge)

    return df_merge


def lr_model_df1(**kwargs):

    from sklearn.model_selection import train_test_split
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score

    ti = kwargs['ti']

    # Retrieve the DataFrame from XCom
    df_fe1 = ti.xcom_pull(task_ids='feature_eng_1')

    # Separate features and target variable
    X = df_fe1.drop(columns=['target'])
    y = df_fe1['target']

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Initialize and train the logistic regression model
    lr = LogisticRegression()
    lr.fit(X_train, y_train)

    # Make predictions on the test set
    lr_predictions = lr.predict(X_test)

    # Calculate the accuracy of the model
    lr1_accuracy = accuracy_score(y_test, lr_predictions)

    return lr1_accuracy



def lr_model_df2(**kwargs):

    from sklearn.model_selection import train_test_split
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score
    from sklearn.preprocessing import StandardScaler

    ti = kwargs['ti']

    # Retrieve the DataFrame from XCom
    df_fe2 = ti.xcom_pull(task_ids='feature_eng_2')

    # Separate features and target variable
    X = df_fe2.drop(columns=['target'])
    y = df_fe2['target']

    # Standardize the features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

    # Initialize and train the logistic regression model
    lr = LogisticRegression()
    lr.fit(X_train, y_train)

    # Make predictions on the test set
    lr_predictions = lr.predict(X_test)

    # Calculate the accuracy of the model
    lr2_accuracy = accuracy_score(y_test, lr_predictions)

    return lr2_accuracy





def lr_model_dfmerge(**kwargs):

    from sklearn.model_selection import train_test_split
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score

    # Access the XCom value using the task instance
    ti = kwargs['ti']
    df_merge = ti.xcom_pull(task_ids='merge')

    # Separate features and target variable
    X = df_merge.drop(columns=['target'])
    y = df_merge['target']

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Initialize and train the logistic regression model
    lr = LogisticRegression()
    lr.fit(X_train, y_train)

    # Make predictions on the test set
    lr_predictions = lr.predict(X_test)

    # Calculate the accuracy of the model
    lrm_accuracy = accuracy_score(y_test, lr_predictions)

    return lrm_accuracy



def svm_model_df1(**kwargs):
    
    from sklearn.model_selection import train_test_split
    from sklearn.svm import SVC
    from sklearn.metrics import accuracy_score
    
    # Access the XCom value using the task instance
    ti = kwargs['ti']
    df_fe1 = ti.xcom_pull(task_ids='feature_eng_1')
    
    # Separate features and target variable
    X = df_fe1.drop(columns=['target'])
    y = df_fe1['target']
    
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Initialize and train the SVM model
    svm = SVC()
    svm.fit(X_train, y_train)
    
    # Make predictions on the test set
    svm_predictions = svm.predict(X_test)
    
    # Calculate the accuracy of the model
    svm1_accuracy = accuracy_score(y_test, svm_predictions)

    return svm1_accuracy


def svm_model_df2(**kwargs):
    
    from sklearn.model_selection import train_test_split
    from sklearn.svm import SVC
    from sklearn.metrics import accuracy_score
    
    # Access the XCom value using the task instance
    ti = kwargs['ti']
    df_fe2 = ti.xcom_pull(task_ids='feature_eng_2')

    # Separate features and target variable
    X = df_fe2.drop(columns=['target'])
    y = df_fe2['target']
    
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Initialize and train the SVM model
    svm = SVC()
    svm.fit(X_train, y_train)
    
    # Make predictions on the test set
    svm_predictions = svm.predict(X_test)
    
    # Calculate the accuracy of the model
    svm2_accuracy = accuracy_score(y_test, svm_predictions)

    return svm2_accuracy

def svm_model_dfmerge(**kwargs):

    from sklearn.model_selection import train_test_split
    from sklearn.svm import SVC
    from sklearn.metrics import accuracy_score
    
    # Access the XCom value using the task instance
    ti = kwargs['ti']
    df_merge = ti.xcom_pull(task_ids='merge')
    
    # Separate features and target variable
    X = df_merge.drop(columns=['target'])
    y = df_merge['target']
    
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Initialize and train the SVM model
    svm = SVC()
    svm.fit(X_train, y_train)
    
    # Make predictions on the test set
    svm_predictions = svm.predict(X_test)
    
    # Calculate the accuracy of the model
    svmm_accuracy = accuracy_score(y_test, svm_predictions)

    return svmm_accuracy


def best_model(**kwargs):
    # Unpack the task instance
    ti = kwargs['ti']

    # Extract accuracy scores from XCom
    lr1_acc = ti.xcom_pull(task_ids='lr_model_df1')
    lr2_acc = ti.xcom_pull(task_ids='lr_model_df2')
    lrm_acc = ti.xcom_pull(task_ids='lr_model_dfmerge')
    svm1_acc = ti.xcom_pull(task_ids='svm_model_df1')
    svm2_acc = ti.xcom_pull(task_ids='svm_model_df2')
    svmm_acc = ti.xcom_pull(task_ids='svm_model_dfmerge')

    # Collect accuracies in a dictionary
    accuracies = {
        'Logistic Regression Model 1': lr1_acc,
        'Logistic Regression Model 2': lr2_acc,
        'Logistic Regression Merged Model': lrm_acc,
        'SVM Model 1': svm1_acc,
        'SVM Model 2': svm2_acc,
        'SVM Merged Model': svmm_acc
    }

    # Identify the best model by accuracy
    best_model_name, best_accuracy = max(accuracies.items(), key=lambda x: x[1])

    # Construct a report message
    report_message = f"The best model is '{best_model_name}' with an accuracy of {best_accuracy}"

    return (best_model_name, best_accuracy), report_message



def evaluate_on_test(**kwargs):
    from sklearn.model_selection import train_test_split
    from sklearn.linear_model import LogisticRegression
    from sklearn.svm import SVC
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
    import pandas as pd

    # Retrieve the best model information from XCom
    ti = kwargs['ti']
    best_model_info, best_model_message = ti.xcom_pull(task_ids='best_model')
    best_model_name = best_model_info[0]
    print(f"Evaluating the best model: {best_model_name}")

    # Function to train and evaluate a given model
    def train_and_evaluate_model(X, y, model):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
        return predictions, y_test

    # Select and initialize the appropriate model based on the best model name
    if 'lr' in best_model_name:
        model = LogisticRegression()
    elif 'svm' in best_model_name:
        model = SVC()

    # Retrieve the appropriate dataset based on the best model name
    if 'df1' in best_model_name:
        df = ti.xcom_pull(task_ids='feature_eng_1')
    elif 'df2' in best_model_name:
        df = ti.xcom_pull(task_ids='feature_eng_2')
    elif 'dfmerge' in best_model_name:
        df = ti.xcom_pull(task_ids='merge')

    # Prepare features and target
    X = df.drop(columns=['target'])
    y = df['target']

    # Train and evaluate the model
    predictions, y_test = train_and_evaluate_model(X, y, model)

    # Calculate evaluation metrics
    accuracy = accuracy_score(y_test, predictions)
    precision = precision_score(y_test, predictions, average='binary')
    recall = recall_score(y_test, predictions, average='binary')
    f1 = f1_score(y_test, predictions, average='binary')

    # Print evaluation metrics
    print(f"Accuracy of the best model on the testing data: {accuracy}")
    print(f"Precision of the best model on the testing data: {precision}")
    print(f"Recall of the best model on the testing data: {recall}")
    print(f"F1 Score of the best model on the testing data: {f1}")

    # Return evaluation metrics
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
    }


    

    


# Instantiate the DAG
dag = DAG(
    'HW4',
    default_args=default_args,
    description='EDA with feature engineering and model selection',
    schedule_interval=PARAMS['workflow']['workflow_schedule_interval'],
    tags=["de300"]
)

load_data_task = PythonOperator(
    task_id='load_data',
    python_callable=load_data,
    dag=dag
)

clean_impute_sklearn_task = PythonOperator(
    task_id='clean_impute_sklearn',
    python_callable=clean_impute_sklearn,
    provide_context=True,
    dag=dag
)

feature_eng_1_task = PythonOperator(
    task_id='feature_eng_1',
    python_callable=feature_eng_1,
    provide_context=True,
    dag=dag
)


clean_impute_spark_task = PythonOperator(
    task_id='clean_impute_spark',
    python_callable=clean_impute_spark,
    provide_context=True,
    dag=dag
)

feature_eng_2_task = PythonOperator(
    task_id='feature_eng_2',
    python_callable=feature_eng_2,
    provide_context=True,
    dag=dag
)


web_scrape_task = PythonOperator(
    task_id='web_scrape',
    python_callable=web_scrape,
    provide_context=True,
    dag=dag
)

merge_task = PythonOperator(
    task_id='merge',
    python_callable=merge,
    provide_context=True,
    dag=dag
)


lr1_task = PythonOperator(
    task_id='lr_model_df1',
    python_callable=lr_model_df1,
    provide_context=True,
    dag=dag
)


lr2_task = PythonOperator(
    task_id='lr_model_df2',
    python_callable=lr_model_df2,
    provide_context=True,
    dag=dag
)

lrmerge_task = PythonOperator(
    task_id='lr_model_dfmerge',
    python_callable=lr_model_dfmerge,
    provide_context=True,
    dag=dag
)


svm1_task = PythonOperator(
    task_id='svm_model_df1',
    python_callable=svm_model_df1,
    provide_context=True,
    dag=dag
)


svm2_task = PythonOperator(
    task_id='svm_model_df2',
    python_callable=svm_model_df2,
    provide_context=True,
    dag=dag
)


svm_merge_task = PythonOperator(
    task_id='svm_model_dfmerge',
    python_callable=svm_model_dfmerge,
    provide_context=True,
    dag=dag
)


best_model_task = PythonOperator(
    task_id='best_model',
    python_callable=best_model,
    provide_context=True,
    dag=dag
)


evaluate_on_test_task = PythonOperator(
    task_id='evaluate_on_test',
    python_callable=evaluate_on_test,
    provide_context=True,
    dag=dag
)

load_data_task >> [clean_impute_sklearn_task, clean_impute_spark_task] 
clean_impute_sklearn_task >> feature_eng_1_task >> [lr1_task, svm1_task, merge_task]
clean_impute_spark_task >> feature_eng_2_task >> [lr2_task, svm2_task, merge_task]
web_scrape_task >> merge_task >> [lrmerge_task, svm_merge_task]
[lr1_task, svm1_task,lr2_task, svm2_task,lrmerge_task, svm_merge_task] >> best_model_task
best_model_task >> evaluate_on_test_task