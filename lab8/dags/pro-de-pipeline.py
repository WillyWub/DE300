from airflow import DAG
from airflow.operators.python_operator import PythonOperator, BranchPythonOperator
#from airflow.providers.sftp.operators.sftp import SFTPOperator
from airflow.providers.postgres.operators.postgres import PostgresOperator
from airflow.operators.dummy_operator import DummyOperator
#from airflow.contrib.hooks.ssh_hook import SSHHook
from airflow.utils.dates import days_ago
from airflow.hooks.base_hook import BaseHook
import boto3
import pandas as pd
from io import StringIO
import tomli
import pathlib
from sqlalchemy import create_engine

# read the parameters from toml
CONFIG_BUCKET = "lab7wilson"
CONFIG_FILE_KEY = "config.toml"


TABLE_NAMES = {
    "original_data": "wine",
    "clean_data": "wine_clean_data",
    "train_data": "wine_train_data",
    "test_data": "wine_test_data",
    "normalization_data": "normalization_values",
    "max_fe": "max_fe_features",
    "product_fe": "product_fe_features"
}

ENCODED_SUFFIX = "_encoded"

# Define the default args dictionary for DAG
default_args = {
    'owner': 'Wilson',
    'depends_on_past': False,
    'start_date': days_ago(1),
    'retries': 1,
}


def read_config_from_s3() -> dict:
    s3_client = boto3.client('s3')
    obj = s3_client.get_object(Bucket=CONFIG_BUCKET, Key=CONFIG_FILE_KEY)
    config_data = obj['Body'].read().decode('utf-8')
    params = tomli.loads(config_data)
    return params

# Usage
PARAMS = read_config_from_s3()

def create_db_connection():
    """
    Create a database connection to the PostgreSQL RDS instance using SQLAlchemy.
    """
   
    conn_uri = f"{PARAMS['db']['db_alchemy_driver']}://{PARAMS['db']['username']}:{PARAMS['db']['password']}@{PARAMS['db']['host']}:{PARAMS['db']['port']}/{PARAMS['db']['db_name']}"

    # Create a SQLAlchemy engine and connect
    engine = create_engine(conn_uri)
    connection = engine.connect()

    return connection



def from_table_to_df(input_table_names: list[str], output_table_names: list[str]):
    """
    Decorator to open a list of tables input_table_names, load them in df and pass the dataframe to the function; on exit, it deletes tables in output_table_names
    The function has key = dfs with the value corresponding the list of the dataframes 

    The function must return a dictionary with key dfs; the values must be a list of dictionaries with keys df and table_name; Each df is written to table table_name
    """

    def decorator(func):
        def wrapper(*args, **kwargs):
            import pandas as pd

            """
            load tables to dataframes
            """
            if input_table_names is None:
                raise ValueError('input_table_names cannot be None')
            
            _input_table_names = None
            if isinstance(input_table_names, str):
                _input_table_names = [input_table_names]
            else:
                _input_table_names = input_table_names

            import pandas as pd
            
            print(f'Loading input tables to dataframes: {_input_table_names}')

            # open the connection
            conn = create_db_connection()

            # read tables and convert to dataframes
            dfs = []
            for table_name in _input_table_names:
                df = pd.read_sql(f"SELECT * FROM {table_name}", conn)
                dfs.append(df)

            if isinstance(input_table_names, str):
                dfs = dfs[0]

            """
            call the main function
            """

            kwargs['dfs'] = dfs
            kwargs['output_table_names'] = output_table_names
            result = func(*args, **kwargs)

            """
            delete tables
            """

            print(f'Deleting tables: {output_table_names}')
            if output_table_names is None:
                _output_table_names = []
            elif isinstance(output_table_names, str):
                _output_table_names = [output_table_names]
            else:
                _output_table_names = output_table_names
            
            print(f"Dropping tables {_output_table_names}")
            for table_name in _output_table_names:
                conn.execute(f"DROP TABLE IF EXISTS {table_name}")

            """
            write dataframes in result to tables
            """

            for pairs in result['dfs']:
                df = pairs['df']
                table_name = pairs['table_name']
                df.to_sql(table_name, conn, if_exists="replace", index=False)
                print(f"Wrote to table {table_name}")

            conn.close()
            result.pop('dfs')

            return result
        return wrapper
    return decorator



def add_data_to_table_func(**kwargs):
    """
    Insert data from a CSV file stored in S3 to a database table.
    """
    # Create a database connection
    conn = create_db_connection()
    
    # Set the S3 bucket and file key
    s3_bucket = PARAMS['files']['s3_bucket']
    s3_key = PARAMS['files']['s3_file_key']
    
    # Create an S3 client
    s3_client = boto3.client('s3')
    
    # Get the object from the S3 bucket
    response = s3_client.get_object(Bucket=s3_bucket, Key=s3_key)
    
    # Read the CSV file directly from the S3 object's byte stream into a DataFrame
    csv_content = response['Body'].read().decode('utf-8')
    df = pd.read_csv(StringIO(csv_content))
    
    # Write the DataFrame to SQL
    df.to_sql(TABLE_NAMES['original_data'], con=conn, if_exists="replace", index=False)

    # Close the database connection
    conn.close()

    return {'status': 1}

@from_table_to_df(TABLE_NAMES['original_data'], None)
def clean_data_func(**kwargs):
    """
    data cleaning: drop none, remove outliers based on z-scores
    apply label encoding on categorical variables: assumption is that every string column is categorical
    """

    import pandas as pd
    from sklearn.preprocessing import LabelEncoder

    data_df = kwargs['dfs']

    # Drop rows with missing values
    data_df = data_df.dropna()

    # Remove outliers using Z-score 
    numeric_columns = [v for v in data_df.select_dtypes(include=['float64', 'int64']).columns if v != PARAMS['ml']['labels']]
    for column in numeric_columns:
        values = (data_df[column] - data_df[column].mean()).abs() / data_df[column].std() - PARAMS['ml']['outliers_std_factor']
        data_df = data_df[values < PARAMS['ml']['tolerance']]

    # label encoding
    label_encoder = LabelEncoder()
    string_columns = [v for v in data_df.select_dtypes(exclude=['float64', 'int64']).columns if v != PARAMS['ml']['labels']]
    for v in string_columns:
        data_df[v + ENCODED_SUFFIX] = label_encoder.fit_transform(data_df[v])

    return {
        'dfs': [
            {'df': data_df, 
             'table_name': TABLE_NAMES['clean_data']
             }]
        }

@from_table_to_df(TABLE_NAMES['clean_data'], None)
def normalize_data_func(**kwargs):
    """
    normalization
    split to train/test
    """
    
    from sklearn.preprocessing import MinMaxScaler
    from sklearn.model_selection import train_test_split
    import pandas as pd

    df = kwargs['dfs']

    # Split the data into training and test sets
    df_train, df_test = train_test_split(df, test_size=PARAMS['ml']['train_test_ratio'], random_state=42)

    # Normalize numerical columns
    normalization_values = [] # 
    for column in [v for v in df_train.select_dtypes(include=['float64', 'int64']).columns if v != PARAMS['ml']['labels']]:
        scaler = MinMaxScaler()
        df_train[column] = scaler.fit_transform(df_train[column].values.reshape(-1, 1))
        normalization_values.append((column, scaler.data_min_[0], scaler.data_max_[0], scaler.scale_[0]))
    normalization_df = pd.DataFrame(data=normalization_values, columns=["name", "min", "max", "scale"])

    return {
        'dfs': [
            {
                'df': df_train, 
                'table_name': TABLE_NAMES['train_data']
            },
            {
                'df': df_test,
                'table_name': TABLE_NAMES['test_data']   
            },
            {
                'df': normalization_df,
                'table_name': TABLE_NAMES['normalization_data']
            }
            ]
        }

@from_table_to_df(TABLE_NAMES['clean_data'], None)
def eda_func(**kwargs):
    """
    print basic statistics
    """

    import pandas as pd

    df = kwargs['dfs']
    
    print(df.describe())

    return { 'dfs': [] }

@from_table_to_df(TABLE_NAMES['train_data'], None)
def fe_max_func(**kwargs):
    """
    add features that are max of two features 
    """

    import pandas as pd

    df = kwargs['dfs']

    # Create new features that are products of all pairs of features
    features = [v for v in df.select_dtypes(include=['float64', 'int64']).columns if v != PARAMS['ml']['labels']]
    new_features_df = pd.DataFrame()
    for i in range(len(features)):
        for j in range(i+1, len(features)):
            new_features_df['max_'+features[i]+'_'+features[j]] = df[[features[i], features[j]]].max(axis=1)

    return {
        'dfs': [
            {'df': new_features_df, 
             'table_name': TABLE_NAMES['max_fe']
             }]
        }

@from_table_to_df(TABLE_NAMES['train_data'], None)
def fe_product_func(**kwargs):
    """
    add features that are products of two features
    """
    
    import pandas as pd

    df = kwargs['dfs']

    # Create new features that are products of all pairs of features
    features = [v for v in df.select_dtypes(include=['float64', 'int64']).columns if v != PARAMS['ml']['labels']]
    new_features_df = pd.DataFrame()
    for i in range(len(features)):
        for j in range(i+1, len(features)):
            new_features_df[features[i]+'*'+features[j]] = df[features[i]] * df[features[j]]

    # NOTE: normalization should be done

    return {
        'dfs': [
            {'df': new_features_df, 
             'table_name': TABLE_NAMES['product_fe']
             }]
        }

def train_model(dfs):
    """
    train logistic regression
    """

    import pandas as pd
    from sklearn.model_selection import train_test_split
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score

    org_df = dfs[1]
    fe_df = dfs[0]

    # combine dataframes
    df = pd.concat([org_df, fe_df], axis=1)

    # Split the data into training and validation sets
    string_columns = [v for v in df.select_dtypes(exclude=['float64', 'int64']).columns if v != PARAMS['ml']['labels']]
    df = df.drop(string_columns, axis=1)

    Y = df[PARAMS['ml']['labels']]
    X = df.drop(PARAMS['ml']['labels'], axis=1)
    
    X_train, X_val, y_train, y_val = train_test_split(X, Y, test_size=PARAMS['ml']['train_test_ratio'], random_state=42)

    # Create an instance of Logistic Regression model
    model = LogisticRegression()

    # Train the model
    model.fit(X_train, y_train)

    # Make predictions on the val set
    y_pred = model.predict(X_val)

    # Calculate the accuracy of the model
    accuracy = accuracy_score(y_val, y_pred)
    print("Accuracy:", accuracy)        

    return accuracy

@from_table_to_df([TABLE_NAMES['product_fe'], TABLE_NAMES['train_data']], TABLE_NAMES["product_fe"])
def product_train_func(**kwargs):
    """
    train logistic regression on product features
    """

    dfs = kwargs['dfs']
 
    accuracy = train_model(dfs)

    return {
        "accuracy": accuracy,
        'dfs': []
    }

@from_table_to_df([TABLE_NAMES['train_data']], None)
def production_train_func(**kwargs):
    """
    train logistic regression on the production model which is not using any additional features
    """

    import pandas as pd

    dfs = kwargs['dfs']
    null_df = pd.DataFrame()
    dfs.append(null_df)

    accuracy = train_model(dfs)

    return {
        "accuracy": accuracy,
        'dfs': []
    }

@from_table_to_df([TABLE_NAMES['max_fe'], TABLE_NAMES['train_data']], TABLE_NAMES["max_fe"])
def max_train_func(**kwargs):
    """
    train logistic regression on max features
    """

    dfs = kwargs['dfs']
 
    accuracy = train_model(dfs)

    return {
        "accuracy": accuracy,
        'dfs': []
    }

feature_operations = ["max", "product"] # used when we automatically create tasks
def encode_task_id(feature_operation: str):
    return f'{feature_type}_evaluation'

def decide_which_model(**kwargs):
    """
    perform testing on the best model; if the best model not better than the production model, do nothing
    """
    
    ti = kwargs['ti']
    max_train_return_value = ti.xcom_pull(task_ids='max_train')
    product_train_return_value = ti.xcom_pull(task_ids='product_train')
    production_train_return_value = ti.xcom_pull(task_ids='production_train')
    
    print(f"Accuracies (product, max, production) {product_train_return_value['accuracy']}, {max_train_return_value['accuracy']}, {production_train_return_value['accuracy']}")

    if max(max_train_return_value['accuracy'], product_train_return_value['accuracy']) - production_train_return_value['accuracy'] <  -PARAMS['ml']['tolerance']:
        return "do_nothing"
    elif max_train_return_value['accuracy'] > product_train_return_value['accuracy']:
        return encode_task_id("max")
    else:
        return encode_task_id("product")

def extensive_evaluation_func(train_df, test_df, fe_type: str, **kwargs):
    """
    train the model on the entire validation data set
    test the final model on test; evaluation also on perturbed test data set
    """
    
    import pandas as pd
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score
    import numpy as np
    from scipy.stats import norm

    model = LogisticRegression()

    # Train the model
    y_train = train_df[PARAMS['ml']['labels']]
    X_train = train_df.drop(PARAMS['ml']['labels'], axis=1)
    model.fit(X_train, y_train)

    def accuracy_on_test(perturb:str = True) -> float:
        X_test = test_df.drop(PARAMS['ml']['labels'], axis=1)
        y_test = test_df[PARAMS['ml']['labels']]

        if perturb == True:
            # we are also perturbing categorical features which is fine since the perturbation is small and thus should not have affect on such features
            X_test = X_test.apply(lambda x: x + np.random.normal(0, PARAMS['ml']['perturbation_std'], len(x)))

        y_pred = model.predict(X_test)
        # Calculate the accuracy of the model
        accuracy = accuracy_score(y_test, y_pred)

        return accuracy

    accuracy = accuracy_on_test(perturb=False)
    print(f"Accuracy on test {accuracy}")

    # we stop when given confidence in accuracy is achieved
    accuracies = []
    for i in range(PARAMS['ml']['max_perturbation_iterations']):
        # Make predictions on the test set
        accuracy = accuracy_on_test()
        accuracies.append(accuracy)

        # compute the confidence interval; break if in the range
        average = np.mean(accuracies)
        std_error = np.std(accuracies) / np.sqrt(len(accuracies))
        confidence_interval = norm.interval(PARAMS['ml']['confidence_level'], loc=average, scale=std_error)
        confidence = confidence_interval[1] - confidence_interval[0]
        if confidence <= 2 * std_error:
            break
    else:
        print(f"Max number of trials reached. Average accuracy on perturbed test {average} with confidence {confidence} and std error of {2 * std_error}")

    print(f"Average accuracy on perturbed test {average}")
    
@from_table_to_df([TABLE_NAMES['train_data'], TABLE_NAMES['test_data']], None)
def max_evaluation_func(**kwargs):
    dfs = kwargs['dfs']
    extensive_evaluation_func(dfs[0], dfs[1], "max")

    return {'dfs': []}

@from_table_to_df([TABLE_NAMES['train_data'], TABLE_NAMES['test_data']], None)
def product_evaluation_func(**kwargs):
    dfs = kwargs['dfs']
    extensive_evaluation_func(dfs[0], dfs[1], "product")

    return {'dfs': []}

# Instantiate the DAG
dag = DAG(
    'Classify',
    default_args=default_args,
    description='Classify with feature engineering/model selection',
    schedule_interval=PARAMS['workflow']['workflow_schedule_interval'],
    tags=["de300"]
)

drop_tables = PostgresOperator(
    task_id="drop_tables",
    postgres_conn_id=PARAMS['db']['db_connection'],
    sql=f"""
    DROP SCHEMA public CASCADE;
    CREATE SCHEMA public;
    GRANT ALL ON SCHEMA public TO {PARAMS['db']['username']};
    GRANT ALL ON SCHEMA public TO public;
    COMMENT ON SCHEMA public IS 'standard public schema';
    """,
    dag=dag
)

add_data_to_table = PythonOperator(
    task_id='add_data_to_table',
    python_callable=add_data_to_table_func,
    provide_context=True,
    dag=dag
)

clean_data = PythonOperator(
    task_id='clean_data',
    python_callable=clean_data_func,
    provide_context=True,
    dag=dag
)

normalize_data = PythonOperator(
    task_id='normalize_data',
    python_callable=normalize_data_func,
    provide_context=True,
    dag=dag
)

eda = PythonOperator(
    task_id='EDA',
    python_callable=eda_func,
    provide_context=True,
    dag=dag
)

fe_max = PythonOperator(
    task_id='add_max_features',
    python_callable=fe_max_func,
    provide_context=True,
    dag=dag
)

fe_product = PythonOperator(
    task_id='add_product_features',
    python_callable=fe_product_func,
    provide_context=True,
    dag=dag
)

product_train = PythonOperator(
    task_id='product_train',
    python_callable=product_train_func,
    provide_context=True,
    dag=dag
)

max_train = PythonOperator(
    task_id='max_train',
    python_callable=max_train_func,
    provide_context=True,
    dag=dag
)

production_train = PythonOperator(
    task_id='production_train',
    python_callable=production_train_func,
    provide_context=True,
    dag=dag
)

model_selection = BranchPythonOperator(
    task_id='model_selection',
    python_callable=decide_which_model,
    provide_context=True,
    dag=dag,
)

dummy_task = DummyOperator(
    task_id='do_nothing',
    dag=dag,
)

evaluation_tasks = []
for feature_type in feature_operations:
    encoding = encode_task_id(feature_type)
    evaluation_tasks.append(PythonOperator(
        task_id=encoding,
        python_callable=locals()[f'{encoding}_func'],
        provide_context=True,
        dag=dag
    ))

drop_tables >> add_data_to_table >> clean_data >> normalize_data
clean_data >> eda
normalize_data >> [fe_max, fe_product]
fe_product >> product_train
fe_max >> max_train
normalize_data >> production_train
[product_train, max_train, production_train] >> model_selection
model_selection >> [dummy_task, *evaluation_tasks]