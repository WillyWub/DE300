# In[44]:


from pyspark.sql import SparkSession
from pyspark.sql.functions import col, when, isnan, mean, lit, udf, sum
from pyspark.sql.types import FloatType, IntegerType, DoubleType, ArrayType
from pyspark.ml.feature import Imputer, VectorAssembler, StringIndexer
from pyspark.ml.classification import LogisticRegression, DecisionTreeClassifier, RandomForestClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
from pyspark.ml.linalg import Vectors, DenseVector
import requests
from bs4 import BeautifulSoup
import json
import numpy as np
import math
import re
from scrapy import Selector
import random
import os



def main():

    spark = SparkSession.builder \
        .appName("HeartDiseasePrediction") \
        .getOrCreate()


    # In[45]:

    df = spark.read.csv("s3://hw3wilson/data/heart_disease.csv", header=True, inferSchema=True)


    # In[46]:


    retain = [
        'age', 'sex', 'painloc', 'painexer', 'cp', 'trestbps', 'smoke',
        'fbs', 'prop', 'nitr', 'pro', 'diuretic', 'thaldur', 'thalach', 'exang',
        'oldpeak', 'slope'
    ]
    selected_columns_with_target = retain + ['target']
    df = df.select(*selected_columns_with_target)

    # Replace NaN in 'painloc' and 'painexer' with their modes
    painloc_mode = df.groupBy('painloc').count().orderBy('count', ascending=False).first()['painloc']
    painexer_mode = df.groupBy('painexer').count().orderBy('count', ascending=False).first()['painexer']
    df = df.fillna({'painloc': painloc_mode, 'painexer': painexer_mode})

    mean_above_100 = df.filter(col('trestbps') > 100).select(mean('trestbps')).collect()[0][0]
    df = df.withColumn('trestbps', when((col('trestbps') < 100) | col('trestbps').isNull() | isnan(col('trestbps')), mean_above_100).otherwise(col('trestbps')))

    # Replace values in 'oldpeak' below 0 or above 4 with the mean of values between 0 and 4
    mean_between_0_and_4 = df.filter((col('oldpeak') >= 0) & (col('oldpeak') <= 4)).select(mean('oldpeak')).collect()[0][0]
    df = df.withColumn('oldpeak', when((col('oldpeak') < 0) | (col('oldpeak') > 4) | col('oldpeak').isNull() | isnan(col('oldpeak')), mean_between_0_and_4).otherwise(col('oldpeak')))

    # Replace NaN in 'thaldur' and 'thalach' with their means
    mean_thaldur = df.select(mean('thaldur')).collect()[0][0]
    mean_thalach = df.select(mean('thalach')).collect()[0][0]
    df = df.fillna({'thaldur': mean_thaldur, 'thalach': mean_thalach})

    # Replace values greater than 1 in specified columns with NaN and fill NaN with mode
    columns_to_replace = ['fbs', 'prop', 'nitr', 'pro', 'diuretic']
    for column in columns_to_replace:
        mode_value = df.groupBy(column).count().orderBy('count', ascending=False).first()[column]
        df = df.withColumn(column, when(col(column) > 1, lit(None)).otherwise(col(column)))
        df = df.fillna({column: mode_value})

    exang_mode = df.groupBy('exang').count().orderBy('count', ascending=False).first()['exang']
    slope_mode = df.groupBy('slope').count().orderBy('count', ascending=False).first()['slope']
    df = df.fillna({'exang': exang_mode, 'slope': slope_mode})


    # In[47]:


    df = df.withColumn("source_1", df["smoke"]).withColumn("source_2", df["smoke"])

    # Source 1: Scrape data and create the smoking dictionary
    url = "https://www.abs.gov.au/statistics/health/health-conditions-and-risks/smoking-and-vaping/latest-release"
    response = requests.get(url)
    soup = BeautifulSoup(response.content, 'html.parser')
    part_of_caption = "Proportion of people 15 years and over who were current daily smokers by age, 2011"  # replace with your caption
    div = None

    for d in soup.find_all('div', {'class': 'chart-data-wrapper'}):
        caption = d.find('pre', {'class': 'chart-caption'}).text
        if part_of_caption in caption:
            div = d
            break

    # Extract chart data
    data = json.loads(div.find('pre', {'class': 'chart-data'}).text)
    smoking_2022 = np.array(data[7]).flatten() / 100
    age_dict = {
        (15, 17): smoking_2022[0],
        (18, 24): smoking_2022[1],
        (25, 34): smoking_2022[2],
        (35, 44): smoking_2022[3],
        (45, 54): smoking_2022[4],
        (55, 64): smoking_2022[5],
        (65, 74): smoking_2022[6],
        (75, 1000): smoking_2022[7]
    }

    # Impute missing values in 'source_1'
    for age_range, rate in age_dict.items():
        df = df.withColumn(
            "source_1",
            when(
                (isnan(col("source_1")) | (col("source_1").isNull())) &
                (col("age").between(age_range[0], age_range[1])),
                when(
                    lit(np.random.rand()) <= rate, lit(1.0)
                ).otherwise(lit(0.0))
            ).otherwise(col("source_1"))
        )

    tobacco_data_url = "https://www.cdc.gov/tobacco/data_statistics/fact_sheets/adult_data/cig_smoking/index.htm"
    response = requests.get(tobacco_data_url)
    html_content = response.content
    selector = Selector(text=html_content)

    row_section = selector.xpath("//div[@class='row '][3]")
    unordered_lists = row_section.xpath("//ul[@class='block-list']")

    sex_rates = unordered_lists[0].xpath(".//li/text()").extract()  # Smoking rates by sex
    age_rates = unordered_lists[1].xpath(".//li/text()").extract()  # Smoking rates by age group

    male_smoking_rate = float(re.search(r"\((\d+(\.\d+)?)%", sex_rates[0]).group(1))
    female_smoking_rate = float(re.search(r"\((\d+(\.\d+)?)%", sex_rates[1]).group(1))

    age_based_rates = {}
    for rate_text in age_rates:
        age_info = re.search(r"aged (\d+–\d+|\d+)", rate_text).group(1)
        smoking_percentage = float(re.search(r"\((\d+(\.\d+)?)%", rate_text).group(1))

        if "–" in age_info:
            age_range = list(map(int, age_info.split("–")))
            age_based_rates[(age_range[0], age_range[1])] = smoking_percentage
        else:
            age_based_rates[(int(age_info), float('inf'))] = smoking_percentage

    male_adjusted_rates = {
        age_range: rate * (male_smoking_rate / female_smoking_rate)
        for age_range, rate in age_based_rates.items()
    }

    female_rates = {key: value / 100 for key, value in age_based_rates.items()}
    male_rates = {key: value / 100 for key, value in male_adjusted_rates.items()}

    for age_range, rate in male_rates.items():
        df = df.withColumn(
            "source_2",
            when(
                (isnan(col("source_2")) | (col("source_2").isNull())) &
                (col("age").between(age_range[0], age_range[1])) &
                (col("sex") == 1),
                when(
                    lit(np.random.rand()) <= rate, lit(1.0)
                ).otherwise(lit(0.0))
            ).otherwise(col("source_2"))
        )

    for age_range, rate in female_rates.items():
        df = df.withColumn(
            "source_2",
            when(
                (isnan(col("source_2")) | (col("source_2").isNull())) &
                (col("age").between(age_range[0], age_range[1])) &
                (col("sex") == 0),
                when(
                    lit(np.random.rand()) <= rate, lit(1.0)
                ).otherwise(lit(0.0))
            ).otherwise(col("source_2"))
        )

    df = df.withColumn(
        "smoke",
        when(
            (isnan(col("smoke")) | (col("smoke").isNull())),
            when(
                (col("source_1") + col("source_2")) >= 1, lit(1.0)
            ).otherwise(lit(0.0))
        ).otherwise(col("smoke"))
    )



    # In[52]:

    print(df)
    feature_columns = retain + ['source_1', 'source_2']

    assembler = VectorAssembler(inputCols=feature_columns, outputCol="features")
    df = assembler.transform(df)
    df = df.withColumnRenamed("target", "label")
    train, test = df.randomSplit([0.9, 0.1], seed=42)
    def evaluate_model(predictions):
        evaluator = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="accuracy")
        accuracy = evaluator.evaluate(predictions)
        return accuracy

    # Logistic Regression
    lr = LogisticRegression(labelCol='label', featuresCol='features')
    lr_model = lr.fit(train)
    lr_predictions = lr_model.transform(test)
    lr_accuracy = evaluate_model(lr_predictions)

    # Decision Tree
    dt = DecisionTreeClassifier(labelCol='label', featuresCol='features')
    dt_model = dt.fit(train)
    dt_predictions = dt_model.transform(test)
    dt_accuracy = evaluate_model(dt_predictions)

    # Random Forest
    rf = RandomForestClassifier(labelCol='label', featuresCol='features')
    rf_model = rf.fit(train)
    rf_predictions = rf_model.transform(test)
    rf_accuracy = evaluate_model(rf_predictions)

    # Determine the best model
    best_model_name = "Logistic Regression" if lr_accuracy > dt_accuracy and lr_accuracy > rf_accuracy else \
                    "Decision Tree" if dt_accuracy > rf_accuracy else \
                    "Random Forest"

    best_model_accuracy = max(lr_accuracy, dt_accuracy, rf_accuracy)

    print(f"The best model is {best_model_name} with an accuracy of {best_model_accuracy:.2f}")

    # Evaluate on test data for the best model
    if best_model_name == "Logistic Regression":
        best_model = lr_model
    elif best_model_name == "Decision Tree":
        best_model = dt_model
    else:
        best_model = rf_model

    predictions = best_model.transform(test)
    test_accuracy = evaluate_model(predictions)
    print(f"Test set accuracy: {test_accuracy:.2f}")

    from pyspark.mllib.evaluation import MulticlassMetrics
    predictionAndLabels = predictions.select("prediction", "label").rdd.map(lambda x: (float(x[0]), float(x[1])))
    metrics = MulticlassMetrics(predictionAndLabels)
    print("Classification report on test set:")
    print(metrics.confusionMatrix().toArray())

    # Stop the Spark session
    spark.stop()
main()

