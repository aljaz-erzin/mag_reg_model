import pyodbc
from sqlalchemy.engine import URL
from sqlalchemy import create_engine, text
import os
from dotenv import load_dotenv

def close_connection_to_db(db_conn):
    db_conn.close()

def connect_to_db():
    load_dotenv(encoding='utf16')
    server_name = os.getenv('SERVER_NAME')
    databa_name = os.getenv('DATABASE_NAME')
    #uid = os.getenv('SQL_USER')
    #pwd = os.getenv('SQL_PASSWORD')
    connection_string = 'Driver={ODBC Driver 17 for SQL Server};Server='+server_name+';DATABASE='+databa_name+';PORT=1433;Trusted_Connection=yes;'#UID='+uid+';PWD='+pwd+';'
    connection_url = URL.create("mssql+pyodbc", query={"odbc_connect": connection_string})
    engine = create_engine(connection_url).connect()
    return engine


def get_db_info():
    load_dotenv(encoding='utf16')
    return os.getenv('PES_TABLE'), os.getenv('PRE_TABLE'), os.getenv('FIM_TABLE'), os.getenv('UNDS_TABLE'), os.getenv(
        'TRMS_TABLE'), os.getenv('MLA_TABLE'), os.getenv('FEA_TABLE'), os.getenv('COR_TABLE'), os.getenv('PRE_TABLE_UNKNOWN'), os.getenv('MODEL_STORAGE'), os.getenv('OUTPUT_FILE')


pes_table, pre_table, fim_table, unds_table, trms_table, mla_table, fea_table, cor_table, pre_table_unknown, model_storage, output_file = get_db_info()