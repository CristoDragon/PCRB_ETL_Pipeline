from sqlalchemy import create_engine
from sqlalchemy.engine import Engine
from sqlalchemy.exc import SQLAlchemyError
import configparser
import urllib
import pandas as pd
import os
import numpy as np
from jellyfish import soundex
from functools import wraps
import logging
from snowflake.connector.pandas_tools import write_pandas
import snowflake.connector

# Date: 11/05/2024
# Author: CMU Capstone Team (Dragon, Michael, Nirvik, Karl)
# Description: This module contains code relevant to the data collection part in machine learning pipeline,
# which includes functions to set up database connection, load data, and cache data.

def create_db_connection(config: configparser, config_section: str) -> Engine:
    """Create a database connection using configuration details.
    
    Args:
        config (ConfigParser): The configuration parser object.
        config_section (str): The section in the configuration file to look for database settings.

    Returns:
        Engine: A SQLAlchemy engine connected to the specified database.

    Raises:
        ValueError: If required sections or keys are missing in the configuration.
    """
    # Check if the configuration section exists
    if config_section not in config.sections():
        logging.error(f"Configuration section '{config_section}' not found in the configuration file.")
        raise ValueError(f"Configuration section '{config_section}' not found in the configuration file.")

    try:
        # Attempt to retrieve the configuration settings
        params = config[config_section]
        server = params['server']
        database = params['database']
    except KeyError as e:
        # Handle missing keys
        logging.error(f"Required configuration key '{e.args[0]}' is missing in the section '{config_section}'.")
        raise ValueError(f"Required configuration key '{e.args[0]}' is missing in the section '{config_section}'.")
    
    # Create a connection string
    params = urllib.parse.quote_plus(
        f'DRIVER={{ODBC Driver 17 for SQL Server}};'
        f'SERVER={server};'
        f'DATABASE={database};'
        f'Trusted_Connection=yes;'
    )

    # Create the connection
    engine = create_engine(f"mssql+pyodbc:///?odbc_connect={params}")
    logging.info(f"Established database connection to {config_section}")
    return engine

def cache_to_csv(filepath):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            if os.path.exists(filepath):
                df = pd.read_csv(filepath, low_memory=False)
            else:
                df = func(*args, **kwargs)
                df.to_csv(filepath, index=False)
            return df
        return wrapper
    return decorator

@cache_to_csv(filepath="df_pcrb.csv")
def load_data_pcrb(engine: Engine, query: str) -> pd.DataFrame:
    """
    Load data from the database.

    Args:
        engine (sqlalchemy.engine.Engine): the SQL engine to a specified database
        query (str): the SQL query to be executed

    Returns:
        pd.DataFrame: the data from executing the SQL query as a pandas DataFrame

    Raises:
        ValueError: If the query results in an empty DataFrame.
        SQLAlchemyError: If there is an issue executing the query (e.g., syntax error, connection issue).
    """
    logging.info(f"Executing the SQL query to pcrb dataset: {query}")
    
    try:
        # Execute the query and load data into a DataFrame
        data_frame = pd.read_sql(query, engine)
        # Check if the data frame is empty
        if data_frame.empty:
            warning_msg = "The query returned an empty dataset. Please check the query or the database state."
            logging.error(warning_msg)
            raise ValueError(warning_msg)
        return data_frame
    except SQLAlchemyError as e:
        # Handle errors from the database engine, such as a bad query or connection issues
        error_msg = f"An error occurred while executing the query: {e}"
        logging.error(error_msg)
        raise e
    except Exception as e:
        # Handle any other exceptions that might occur
        logging.error(f"An unexpected error occurred: {e}")
        raise e

@cache_to_csv(filepath="df_dnb.csv")
def load_data_dnb(engine: Engine, query: str) -> pd.DataFrame:
    """
    Load data from the database.

    Args:
        engine (sqlalchemy.engine.Engine): the SQL engine to a specified database
        query (str): the SQL query to be executed

    Returns:
        pd.DataFrame: the data from executing the SQL query as a pandas DataFrame

    Raises:
        ValueError: If the query results in an empty DataFrame.
        SQLAlchemyError: If there is an issue executing the query (e.g., syntax error, connection issue).
    """
    logging.info(f"Executing the SQL query to dnb dataset: {query}")
    
    try:
        # Execute the query and load data into a DataFrame
        data_frame = pd.read_sql(query, engine)
        # Check if the data frame is empty
        if data_frame.empty:
            warning_msg = "The query returned an empty dataset. Please check the query or the database state."
            logging.error(warning_msg)
            raise ValueError(warning_msg)
        return data_frame
    except SQLAlchemyError as e:
        # Handle errors from the database engine, such as a bad query or connection issues
        error_msg = f"An error occurred while executing the query: {e}"
        logging.error(error_msg)
        raise e
    except Exception as e:
        # Handle any other exceptions that might occur
        logging.error(f"An unexpected error occurred: {e}")
        raise e

def load_data_abbreviation(file_path: str) -> pd.DataFrame:
    """
    Load abbreviation data from an Excel file.

    Args:
        file_path (str): The path to the Excel file containing abbreviation data.

    Returns:
        pd.DataFrame: A DataFrame containing the loaded data from the Excel file.

    Raises:
        FileNotFoundError: If no Excel file is found at the provided `file_path`.
        ValueError: If there is an error loading the Excel file or the file does not contain at least three columns.
    """
    # Check if the Excel file exists
    if not os.path.exists(file_path):
        logging.error(f"No file found at the specified path: {file_path}")
        raise FileNotFoundError(f"No file found at the specified path: {file_path}")

    # Attempt to load the Excel file
    try:
        df = pd.read_excel(file_path, sheet_name='Sheet1', skiprows=3)
    except Exception as e:
        logging.error(f"Error loading Excel file: {str(e)}")
        raise ValueError(f"Error loading Excel file: {str(e)}")

    # Ensure that the DataFrame has at least three columns to match expected column names
    if df.shape[1] < 3:
        logging.error("The Excel sheet does not contain enough columns based on expected column names")
        raise ValueError("The Excel sheet does not contain enough columns based on expected column names")
    
    logging.info(f"Loaded street abbreviation data from '{file_path}'")
    return df


def create_abbreviation_mapping(file_path: str, col_names: list) -> dict:
    """
    Create a mapping dictionary for street abbreviations from an Excel file.

    Args:
        file_path (str): The path to the Excel file containing the abbreviation data.
        col_names (list): A list of column names to be used for primary names, common abbreviations, and standard abbreviations.

    Returns:
        dict: A dictionary where keys are primary names and common abbreviations, and values are the corresponding standard abbreviations.

    Raises:
        FileNotFoundError: If no Excel file is found at the provided `file_path`.
        ValueError: If there is an error processing the data or if expected columns are missing.
    """
    # Load the street abbreviation excel file into a dataframe
    df = load_data_abbreviation(file_path)

    # Set DataFrame columns
    df.columns = col_names[:3]

    # Create a dictionary for abbreviation mappings
    abbrev_mapping = {}
    for _, row in df.iterrows():
        primary = row[col_names[0]]
        common = row[col_names[1]]
        standard = row[col_names[2]]
        
        if not pd.isna(primary) and not pd.isna(standard):
            abbrev_mapping[primary] = standard
        if not pd.isna(common) and not pd.isna(standard):
            abbrev_mapping[common] = standard

    logging.info(f"Created an abbreviation mapping based on '{file_path}'")
    return abbrev_mapping

def load_common_words(file_path: str) -> set:
    """
    Load common words to be removed from business names from a specified file.

    Args:
        file_path (str): Path to the file containing common words.

    Returns:
        set: A set of common words, converted to uppercase.

    Raises:
        FileNotFoundError: If the file does not exist at the provided path.
        ValueError: If the file path is empty or not a string.
        IOError: If an error occurs during file reading.
    """
    if not isinstance(file_path, str) or not file_path:
        logging.error("The file path must be a non-empty string.")
        raise ValueError("The file path must be a non-empty string.")

    if not os.path.exists(file_path):
        logging.error(f"No file found at the specified path: {file_path}")
        raise FileNotFoundError(f"No file found at the specified path: {file_path}")

    try:
        with open(file_path, 'r') as file:
            logging.info(f"Loading common words data from '{file_path}'")
            return set(word.strip().upper() for word in file)
    except IOError as e:
        logging.error(f"Error reading file at {file_path}: {e}")
        raise IOError(f"Error reading file at {file_path}: {e}")

def load_legal_suffixes(file_path: str) -> set:
    """
    Load legal suffixes to be removed from business names from a specified file.

    Args:
        file_path (str): Path to the file containing legal suffixes.

    Returns:
        set: A set of legal suffixes, converted to uppercase.

    Raises:
        FileNotFoundError: If the file does not exist at the provided path.
        ValueError: If the file path is empty or not a string.
        IOError: If an error occurs during file reading.
    """
    if not isinstance(file_path, str) or not file_path:
        logging.error("The file path must be a non-empty string.")
        raise ValueError("The file path must be a non-empty string.")

    if not os.path.exists(file_path):
        logging.error(f"No file found at the specified path: {file_path}")
        raise FileNotFoundError(f"No file found at the specified path: {file_path}")

    try:
        with open(file_path, 'r') as file:
            logging.info(f"Loading legal suffixes data from '{file_path}'")
            return set(suffix.strip().upper() for suffix in file)
    except IOError as e:
        logging.error(f"Error reading file at {file_path}: {e}")
        raise IOError(f"Error reading file at {file_path}: {e}")
    
    
def save_df_as_pickle(df, filename: str):
    """
    Save a DataFrame to a pickle file.

    Args:
        df (pd.DataFrame): The DataFrame to save.
        filename (str): The path where the DataFrame will be saved.

    Raises:
        TypeError: If 'df' is not a pandas DataFrame.
        ValueError: If 'filename' is not a string or is empty.
        IOError: If an I/O error occurs during saving.
    """
    if not isinstance(df, pd.DataFrame):
        logging.error("The provided 'df' must be a pandas DataFrame.")
        raise TypeError("The provided 'df' must be a pandas DataFrame.")
    
    if not isinstance(filename, str) or not filename:
        logging.error("The filename must be a non-empty string.")
        raise ValueError("The filename must be a non-empty string.")

    try:
        df.to_pickle(filename)
        logging.info(f"Saved dataframe to {filename}")
    except IOError as e:
        logging.error(f"Failed to save DataFrame to {filename}: {e}")
        raise IOError(f"Failed to save DataFrame to {filename}: {e}")

def load_df_from_pickle(filename: str):
    """
    Load a DataFrame from a pickle file.

    Args:
        filename (str): The path from where the DataFrame will be loaded.

    Returns:
        pd.DataFrame: The loaded DataFrame.

    Raises:
        ValueError: If 'filename' is not a string or is empty.
        FileNotFoundError: If the file does not exist at the provided path.
        IOError: If an I/O error occurs during loading.
    """
    if not isinstance(filename, str) or not filename:
        logging.error("The filename must be a non-empty string.")
        raise ValueError("The filename must be a non-empty string.")

    try:
        return pd.read_pickle(filename)
    except FileNotFoundError:
        logging.error(f"No file found at the specified path: {filename}")
        raise FileNotFoundError(f"No file found at the specified path: {filename}")
    except IOError as e:
        logging.error(f"Failed to load DataFrame from {filename}: {e}")
        raise IOError(f"Failed to load DataFrame from {filename}: {e}")
    
def create_snowflake_conn(config: configparser, config_section: str) -> snowflake.connector:
    """Create a database connection using configuration details.
    
    Args:
        config (ConfigParser): The configuration parser object.
        config_section (str): The section in the configuration file to look for database settings.

    Returns:
        connection: A snowflake connection connected to the specified database.

    Raises:
        ValueError: If required sections or keys are missing in the configuration.
    """
    # Check if the configuration section exists
    if config_section not in config.sections():
        logging.error(f"Configuration section '{config_section}' not found in the configuration file.")
        raise ValueError(f"Configuration section '{config_section}' not found in the configuration file.")

    try:
        # Attempt to retrieve the configuration settings
        params = config[config_section]
        user = params['user']
        password = params['password']
        account = params['account']
        warehouse = params['warehouse']
        database = params['database']
        schema = params['schema']
        print(user, password, account, warehouse, database, schema)
    except KeyError as e:
        # Handle missing keys
        logging.error(f"Required configuration key '{e.args[0]}' is missing in the section '{config_section}'.")
        raise ValueError(f"Required configuration key '{e.args[0]}' is missing in the section '{config_section}'.")
    # Create the connection
    snowflake_connection = snowflake.connector.connect(
        user=user,
        password=password,
        account=account,
        warehouse=warehouse,
        database=database,
        schema=schema
    )
    logging.info(f"Established database connection to {config_section}")
    return snowflake_connection

def write_df_to_snowflake(df: pd.DataFrame, connection: snowflake.connector, table_name: str):
    """
    Write a DataFrame to a Snowflake table.

    Args:
        df (pd.DataFrame): The DataFrame to write to the Snowflake table.
        connection (snowflake.connector): The Snowflake connection object.
        table_name (str): The name of the table in Snowflake.

    Raises:
        ValueError: If the DataFrame is empty or if the table name is empty.
        IOError: If an I/O error occurs during writing.
    """
    if df.empty:
        logging.error("The DataFrame is empty. Nothing to write.")
        raise ValueError("The DataFrame is empty. Nothing to write.")

    if not table_name:
        logging.error("The table name must be a non-empty string.")
        raise ValueError("The table name must be a non-empty string.")

    try:
        write_pandas(connection, df, table_name)
        logging.info(f"Successfully wrote DataFrame to Snowflake table: {table_name}")
    except Exception as e:
        logging.error(f"Failed to write DataFrame to Snowflake table: {e}")
        raise IOError(f"Failed to write DataFrame to Snowflake table: {e}")
    
def execute_query_snowflake(connection: snowflake.connector, query: str):
    """
    Execute a query on a Snowflake database.

    Args:
        connection (snowflake.connector): The Snowflake connection object.
        query (str): The query to execute.

    Raises:
        ValueError: If the query is empty.
        IOError: If an I/O error occurs during execution.
    """
    if not query:
        logging.error("The query is empty. Nothing to execute.")
        raise ValueError("The query is empty. Nothing to execute.")
    # Create a cursor
    cursor = connection.cursor()
    try:
        cursor.execute(query)
        logging.info("Table created successfully!")
    except Exception as e:
        logging.error(f"An error occurred while executing a query in snowflake: {e}")

def construct_create_table_query(table_name: str, df: pd.DataFrame) -> str:
    """
    Construct a CREATE TABLE query for Snowflake database.

    Args:
        table_name (str): The name of the table to be created.
        columns (dict): A dictionary where keys are column names and values are column data types.

    Returns:
        str: The CREATE TABLE query for Snowflake database.
    """
    # Generate Snowflake CREATE TABLE SQL from DataFrame
    create_table_query = f"CREATE TABLE IF NOT EXISTS {table_name} ("
    for column, dtype in df.dtypes.items():
        if dtype == 'int64':
            snowflake_type = 'INTEGER'
        elif dtype == 'float64':
            snowflake_type = 'FLOAT'
        elif dtype == 'object':
            snowflake_type = 'STRING'
        elif dtype == 'datetime64[ns]':
            snowflake_type = 'TIMESTAMP'
        else:
            snowflake_type = 'STRING'  # Default type

        create_table_query += f"{column} {snowflake_type}, "
    create_table_query = create_table_query.rstrip(", ") + ");"
    return create_table_query