
from sqlalchemy import create_engine
import pandas as pd


def insert_to_table(data: pd.DataFrame, conn_string:str, table_name:str):
    db = create_engine(conn_string)
    conn = db.connect()
    data.to_sql(table_name, conn, if_exists="replace", index=False)
    conn.close()
