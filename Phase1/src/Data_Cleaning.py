import pandas as pd
import re
import numpy as np
from rapidfuzz import fuzz, process
from jellyfish import soundex
import logging
from functools import wraps
import os
import jellyfish
from collections import defaultdict
from multiprocessing import Pool, cpu_count, Manager
import time
from datetime import datetime
from src.Model_Evaluation import log_matching_metrics, get_latest_evaluation_metrics

# Date: 11/11/2024
# Author: CMU Capstone Team (Dragon, Michael, Nirvik, Karl)
# Description: This module contains code relevant to the data cleaning part in machine learning pipeline.

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

def common_standardization(text: str, abbrev_mapping: dict, directions: dict) -> str:
    """
    Apply common standardization rules for addresses and street names.
    
    Args:
        text (str): The text string to standardize.
        abbrev_mapping (dict): A dictionary mapping words to abbreviations.
        directions (dict): A dictionary mapping directions to abbreviations (e.g., 'North' -> 'N').

    Returns:
        str: The standardized text.
    """
    if pd.isna(text):
        return text
    try:
        text = text.upper()
        text = re.sub(r'^\s*\([^)]*\)\s*', '', text)
        # Replace multiple dots with space
        text = re.sub(r'\.+', ' ', text)
        for direction, abbr in directions.items():
            text = re.sub(r'\b' + direction + r'\b', abbr, text)
        for word, abbr in abbrev_mapping.items():
            if word:
                # Add spaces around abbreviation replacement
                text = re.sub(r'\b' + word + r'\b', f' {abbr} ', text)
        text = text.replace('.', '')
        text = ' '.join(text.split())  # This normalizes all spacing
        return text
    except Exception as e:
        logging.error(f"An error occurred during standardization: {e}")
        raise ValueError(f"An error occurred while standardizing the text: {e}")

def standardize_full_address(address, abbrev_mapping, directions, preserve_commas=False):
    """
    Standardize a full address string.

    Args:
        address (str): The full address to standardize.
        abbrev_mapping (dict): A dictionary mapping words to abbreviations.
        directions (dict): A dictionary mapping directions to abbreviations (e.g., 'North' -> 'N').
        preserve_commas (bool): If False, removes commas from the address. Default is False.

    Returns:
        str: The standardized address.
    """
    try:
        # Standardize the address using the common_standardization function
        address = common_standardization(address, abbrev_mapping, directions)
        # Check if address is None or empty after standardization
        if pd.isna(address) or address == "":
            return address
        # Remove commas if preserve_commas is False
        if not preserve_commas:
            address = address.replace(',', '')
        return address
    except Exception as e:
        logging.error(f"An error occurred during address standardization: {e}")
        raise ValueError(f"An error occurred while standardizing the address: {e}")

def standardize_street_only(street, abbrev_mapping, directions):
    """
    Standardize only the street portion of an address.

    Args:
        street (str): The street name to standardize.
        abbrev_mapping (dict): A dictionary mapping words to abbreviations.
        directions (dict): A dictionary mapping directions to abbreviations (e.g., 'North' -> 'N').

    Returns:
        str: The standardized street name.
    """
    try:
        # Apply standardization using common_standardization function
        standardized_street = common_standardization(street, abbrev_mapping, directions)
        return standardized_street
    except Exception as e:
        logging.error(f"An error occurred during street standardization: {e}")
        raise ValueError(f"An error occurred while standardizing the street: {e}")

def standardize_street_only_no_numbers(street: str, abbrev_mapping: dict, directions: dict) -> str:
    """
    Standardize the street name without numbers and suite information.

    Args:
        street (str): The street name to standardize.
        abbrev_mapping (dict): A dictionary mapping words to abbreviations.
        directions (dict): A dictionary mapping directions to abbreviations (e.g., 'North' -> 'N').

    Returns:
        str: The standardized street name with numbers and suite information removed.
    """
    
    try:
        # Standardize the street name using the standardize_street_only function
        street = standardize_street_only(street, abbrev_mapping, directions)
        # If the result is None or NaN, return as is
        if pd.isna(street):
            return street
        # Remove numbers, suite, and office information
        street = re.sub(r'\b(STE|FL|OFC)\.?\s+\d+\w*', '', street)
        street = re.sub(r'\b(SUITE|FLOOR|OFFICE)\s+\d+\w*', '', street)
        street = re.sub(r'^\d+(-\d+)?\s+', '', street)
        street = re.sub(r'\bUNIT\s+\d+\w*', '', street)
        street = re.sub(r'#\s*\d+\w*', '', street)
        # Normalize spacing
        street = ' '.join(street.split())
        return street
    except Exception as e:
        logging.error(f"An error occurred during street standardization (no numbers): {e}")
        raise ValueError(f"An error occurred while standardizing the street name without numbers: {e}")

def replace_numbered_words(name: str, number_words: dict)-> str:
    """
    Replace numbered words like ordinals or numbers with their word form.

    Args:
        name (str): The input string containing numbers or ordinals to replace.
        number_words (dict): Dictionary mapping numbers and ordinals to their word form.

    Returns:
        str: The input string with numbers and ordinals replaced by words.
    """
    try:
        if pd.isna(name):
            return name

        # Replace ordinal numbers
        pattern = r'\b(\d+)(ST|ND|RD|TH)\b'
        def ordinal_replace(match):
            full_match = match.group(0)
            return number_words.get(full_match, full_match)
        name = re.sub(pattern, ordinal_replace, name)

        # Replace regular numbers
        pattern = r'\b(\d+)\b'
        def number_replace(match):
            full_match = match.group(0)
            return number_words.get(full_match, full_match)
        name = re.sub(pattern, number_replace, name)

        return name
    except Exception as e:
        logging.error(f"An error occurred during numbered word replacement: {e}")
        raise ValueError(f"An error occurred while replacing numbered words: {e}")

def separate_suffix(name: str, legal_suffixes: set) -> str:
    """
    Separate legal suffixes that are directly attached to words in a given name.

    This function inserts a space between words and their legal suffixes if they are attached directly.
    For example, 'Inc' in 'CompanyInc' becomes 'Company Inc'.

    Args:
        name (str): The name string from which to separate suffixes.
        legal_suffixes (set): A set of legal suffixes to separate.

    Returns:
        str: The modified name with suffixes separated.
    """
    # Sort suffixes by length in descending order to match longer suffixes first
    for suffix in sorted(legal_suffixes, key=len, reverse=True):
        pattern = r'(\w+)(' + re.escape(suffix) + r')(?!\w)'
        name = re.sub(pattern, r'\1 \2', name)
    return name

def remove_common_words(name: str, common_words: set) -> str:
    """
    Remove common words from a given name.

    Args:
        name (str): The name string from which to remove common words.
        common_words (set): A set of common words to remove.

    Returns:
        str: The modified name with common words removed.
    """
    for word in common_words:
        name = re.sub(r'\b' + re.escape(word) + r'\b', '', name)
    name = ' '.join(name.split())
    return name

def standardize_business_name(name: str, common_words: set, legal_suffixes: set, abbrev_mapping: dict, directions: dict, number_words: dict, remove_spaces=False):
    """Standardize business name with configurable options."""
    if pd.isna(name):
        return name

    # Convert to uppercase
    name = name.upper()

    # Convert numbers to words
    name = replace_numbered_words(name, number_words)

    # Separate attached suffixes and remove them
    name = separate_suffix(name, legal_suffixes)

    # Remove legal suffixes
    for suffix in legal_suffixes:
        name = re.sub(r'\b' + re.escape(suffix) + r'\b', '', name)

    # Replace directional words
    for direction, abbr in directions.items():
        name = re.sub(r'\b' + direction + r'\b', abbr, name)

    # Replace abbreviations
    for word, abbr in abbrev_mapping.items():
        if word:
            name = re.sub(r'\b' + re.escape(word) + r'\b', abbr, name)

    # Remove common words
    name = remove_common_words(name, common_words)

    # Remove punctuation and special characters
    name = re.sub(r'[^0-9A-Z\s]', '', name)
    name = name.replace('&', 'AND')

    # Remove extra whitespace
    name = ' '.join(name.split())

    # Remove all spaces if specified
    if remove_spaces:
        name = name.replace(' ', '')

    return name

def standardize_city(city, reference_cities):
    """Standardize city names using multiple methods."""
    if pd.isna(city):
        return None

    city = city.upper().strip()

    if city in reference_cities:
        return city

    # Try fuzzy first
    fuzzy_result = process.extractOne(
        city, 
        reference_cities,
        scorer=fuzz.token_sort_ratio,
        score_cutoff=85
    )
    if fuzzy_result:
        return fuzzy_result[0]

    # Try soundex if fuzzy fails
    city_soundex = soundex(city)
    matches = [ref_city for ref_city in reference_cities 
              if soundex(ref_city) == city_soundex]

    if matches:
        if len(matches) > 1:
            result = process.extractOne(city, matches, scorer=fuzz.token_sort_ratio)
            return result[0]
        return matches[0]

    return None

def apply_standardization_to_dataframe(df: pd.DataFrame, address_cols: list, abbrev_mapping: dict, directions: dict, reference_cities=None):
    """
    Apply address and street standardization to a DataFrame.

    Args:
        df (pd.DataFrame): The DataFrame to standardize.
        address_cols (list): List of columns for street, city, state, and zip.
        abbrev_mapping (dict): A dictionary mapping words to abbreviations.
        directions (dict): A dictionary mapping directions to abbreviations (e.g., 'North' -> 'N').
        reference_cities (set, optional): Set of reference cities to use for city standardization.

    Returns:
        pd.DataFrame: The DataFrame with standardized address columns.
    """
    
    try:
        # Check if required columns exist
        missing_cols = [col for col in address_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(f"The following address columns are missing in the DataFrame: {missing_cols}")
        # Initialize new columns to be created
        new_cols = ['standardized_address', 'standardized_street_only', 'standardized_street_no_numbers', 'standardized_city']
        # Standardize the full address
        df[new_cols[0]] = df.apply(
            lambda x: standardize_full_address(
                f"{x[address_cols[0]]}, {x[address_cols[1]]}, {x[address_cols[2]]} {x[address_cols[3]]}",
                abbrev_mapping,
                directions,
                preserve_commas=False
            ),
            axis=1
        )
        logging.info(f"Standardized the column '{new_cols[0]}'.")

        # Standardize only the street portion
        df[new_cols[1]] = df[address_cols[0]].apply(
            lambda x: standardize_street_only(x, abbrev_mapping, directions)
        )
        logging.info(f"Standardized the column '{new_cols[1]}'.")

        # Standardize street without numbers
        df[new_cols[2]] = df[address_cols[0]].apply(
            lambda x: standardize_street_only_no_numbers(x, abbrev_mapping, directions)
        )
        logging.info(f"Standardized the column '{new_cols[2]}'.")

        # Standardize the city if reference_cities are provided
        if reference_cities is not None:
            df[new_cols[3]] = df[address_cols[1]].apply(
                lambda x: standardize_city(x, reference_cities)
            )
            logging.info(f"Standardized the column '{new_cols[3]}'.")

        return df

    except Exception as e:
        logging.error(f"An error occurred during address standardization: {e}")
        raise ValueError(f"An error occurred while applying standardization to the DataFrame: {e}")

def create_pcrb_col_FEIN_Address(df_pcrb: pd.DataFrame, cols: list) -> pd.DataFrame:
    try:
        # Verify all columns in cols exist in the DataFrame
        missing_cols = [col for col in cols[1:] if col not in df_pcrb.columns]
        if missing_cols:
            logging.error(f"Missing columns in DataFrame: {missing_cols}")
            raise ValueError(f"The following columns are missing in the DataFrame: {missing_cols}")
        # Create FEIN_Address directly without full_address
        df_pcrb[cols[0]] = (df_pcrb[cols[1]].astype(str) + '_' + df_pcrb[cols[2]].fillna(''))
        logging.info(f"Created pcrb column '{cols[0]}' by concatenating '{cols[1]}' and '{cols[2]}'.")
    except Exception as e:
        logging.error(f"An error occurred while creating FEIN_Address: {e}")
        raise ValueError(f"An error occurred while creating FEIN_Address: {e}")
    return df_pcrb

@cache_to_csv(filepath="df_pcrb_v1.csv")
def apply_pcrb_standardization(df_pcrb: pd.DataFrame, df_dnb: pd.DataFrame, abbrev_mapping: dict, directions: dict, cols_address: list, col_city: str) -> pd.DataFrame:
    """
    Apply address standardization to specified columns in the PCRB DataFrame, referencing cities from the DNB DataFrame.

    Args:
        df_pcrb (pd.DataFrame): The PCRB DataFrame to standardize.
        df_dnb (pd.DataFrame): The DNB DataFrame containing reference city information.
        abbrev_mapping (dict): A dictionary mapping words to abbreviations.
        directions (dict): A dictionary mapping directions to abbreviations (e.g., 'North' -> 'N').
        cols_address (list): List of address columns to standardize.

    Returns:
        pd.DataFrame: The standardized PCRB DataFrame.
    """
    try:
        # Ensure 'physical_city' exists in df_dnb
        if col_city not in df_dnb.columns:
            raise ValueError(f"The DNB DataFrame must contain '{col_city}' column for reference.")
        # Extract unique cities from df_dnb as reference
        reference_cities = set(df_dnb[col_city].unique())
        logging.info("Extracted reference cities from DNB DataFrame.")
        # Apply the standardization function to the specified columns in df_pcrb
        df_pcrb = apply_standardization_to_dataframe(
            df_pcrb,
            cols_address,
            abbrev_mapping,
            directions,
            reference_cities
        )
        logging.info("Completed address standardization to PCRB DataFrame.")
        return df_pcrb
    except Exception as e:
        logging.error(f"An error occurred while applying address standardization to PCRB DataFrame: {e}")
        raise ValueError(f"An error occurred while applying address standardization to PCRB DataFrame: {e}")

@cache_to_csv(filepath="df_dnb_v1.csv")
def apply_dnb_standardization(df_dnb: pd.DataFrame, abbrev_mapping: dict, directions: dict, cols_address: list) -> pd.DataFrame:
    """
    Apply address standardization to specified columns in the DNB DataFrame.

    Args:
        df_dnb (pd.DataFrame): The DNB DataFrame to standardize.
        abbrev_mapping (dict): A dictionary mapping words to abbreviations.
        directions (dict): A dictionary mapping directions to abbreviations (e.g., 'North' -> 'N').
        cols_address (list): List of address columns to standardize.

    Returns:
        pd.DataFrame: The standardized DNB DataFrame.
    """
    try:
        # Apply the standardization function to the specified columns
        df_dnb = apply_standardization_to_dataframe(
            df_dnb,
            cols_address,
            abbrev_mapping,
            directions
        )
        logging.info("Completed address standardization to dnb dataset.")
        return df_dnb

    except Exception as e:
        logging.error(f"An error occurred while applying address standardization to dnb dataset: {e}")
        raise ValueError(f"An error occurred while applying address standardization to dnb dataset: {e}")

def prioritize_address_code(group):
    """
    # Step 1: Function to prioritize records based on AddressTypeCode.
    Logic: 
        This function prioritizes records with AddressTypeCode 2. 
        If AddressTypeCode 2 does not exist, it returns only one record for that PrimaryFEIN (the smallest AddressTypeCode).
        After prioritization, any duplicate enteries on FEIN_Adress will be removed to ensure uniqueness.

        - If any records have 'AddressTypeCode 2', all such records are treated as the highest priority, and only returns such rows.
          Any other row with different AddressTypeCode (1,3,4,5) are ignored in this case.

        - If no records have 'AddressTypeCode 2', the function will only return one record with the smallest 'AddressTypeCode'.
          If there are multiple records with the same smallest AddressTypeCode, only first record is returned to prevent duplicates (.head(1)).
    
    """
    # Initialize the column name for 'AddressTypeCode'
    address_type_code = 'AddressTypeCode'
    # Check for any record in the group (PrimaryFEIN) has AddressTypeCode = 2.
    code_2 = group[group[address_type_code] == 2]

    if not code_2.empty:
        # Return all records with AddressTypeCode = 2.
        return code_2
    else: # If records do not have AddressTypeCode = 2, return only one row of the grouping (PrimaryFEIN) with the smallest AddressType Code.
          # '.head(1) ensures only one record is returned. Some cases had multiple records with same AddressTypeCode.
        return group[group[address_type_code] == group[address_type_code].min()].head(1)

@cache_to_csv(filepath="df_pcrb_v3.csv")
def apply_hierarchy_method(df_pcrb: pd.DataFrame, col_names: list, RUN_ID: str) -> pd.DataFrame:
    """
    Apply a hierarchy method to prioritize rows based on the AddressTypeCode within each PrimaryFEIN group,
    and then remove duplicate FEIN_Address entries.

    Args:
        df_pcrb (pd.DataFrame): The PCRB DataFrame to process.
        col_names (list): List of column names, including:
            - col_names[9]: Column name for PrimaryFEIN
            - col_names[10]: Column name for FEIN_Address
        RUN_ID (str): Unique identifier for the pipeline run.

    Returns:
        pd.DataFrame: The DataFrame after applying the hierarchy method and removing duplicates.
    """
    try:
        # Record the initial size of df_pcrb
        df_rows_before = df_pcrb.shape[0]
        logging.info(f"Initial size of PCRB dataset: {df_rows_before} rows.")

        # Step 2: Apply function to each group of PrimaryFEIN
        # '.groupby(['PrimaryFEIN']': Groups the data by PrimaryFEIN. Each group contains all rows for a particular PrimaryFEIN (e.g., all rows for Walmart).
        # 'group_keys=False': This ensures that grouped records that are removed, do not get added back into the index of the DataFrame.
        # '.apply(prioritize_address_code)': For each PrimaryFEIN, apply the function.
        # If any rows in the group have 'AddressTypeCode = 2', all rows are returned for that PrimaryFEIN, all other rows with different AddressTypeCode are ignored for that PrimaryFEIN.
        # If no rows in that group have 'AddressTypeCode = 2', the function will only return one row. Specifically, the smallest AddressTypeCode. 
        transform_df_2 = df_pcrb.groupby([col_names[9]], group_keys=False).apply(prioritize_address_code)
        rows_after_hierarchy = transform_df_2.shape[0]
        logging.info("Hierarchy method applied to prioritize rows within each PrimaryFEIN group.")

        # Step 3: Remove any duplicate of 'FEIN_Address' as this is now unique.
        rows_before_drop_duplicates = rows_after_hierarchy
        final_df_2 = transform_df_2.drop_duplicates(subset=[col_names[10]])
        rows_after_drop_duplicates = final_df_2.shape[0]
        log_matching_metrics("Apply Hierarchy Method",df_rows_before,rows_after_hierarchy, RUN_ID)
        log_matching_metrics("Drop Duplicates",rows_before_drop_duplicates,rows_after_drop_duplicates, RUN_ID)
        return final_df_2
    except Exception as e:
        logging.error(f"An error occurred while applying the hierarchy method: {e}")
        raise ValueError(f"An error occurred while applying the hierarchy method: {e}")


def standardize_pcrb_FEIN(df: pd.DataFrame, col_FEIN: str) -> pd.DataFrame:
    try:
        # Standardize PrimaryFEIN
        df[col_FEIN] = (df[col_FEIN]
                             .replace(r'^\s*$', np.nan, regex=True)  # Handle whitespace-only strings
                             .fillna('000000000')
                             .str.strip()  # Remove leading/trailing spaces
                             .str.replace(r'\s+', '', regex=True)  # Remove spaces within the number
                             .str.replace('-', '')  # Remove hyphens
                             .replace('', '000000000')
                             .astype('Int64'))
        logging.info(f"Standardized pcrb column '{col_FEIN}'.")
    except Exception as e:
        logging.error(f"An error occurred while standardizing pcrb column '{col_FEIN}': {e}")
        raise ValueError(f"An error occurred while standardizing pcrb column '{col_FEIN}': {e}")
    return df

def standardize_pcrb_city(df: pd.DataFrame, col_city: str, fill_str: str) -> pd.DataFrame:
    try:
        # Standardize City
        df[col_city] = df[col_city].fillna(fill_str).replace('', fill_str)
        logging.info(f"Standardized pcrb column '{col_city}' by replacing null and empty values with '{fill_str}'.")
    except Exception as e:
        logging.error(f"An error occurred while standardizing pcrb column '{col_city}': {e}")
        raise ValueError(f"An error occurred while standardizing pcrb column '{col_city}': {e}")
    return df
    
def standardize_pcrb_zipcode(df: pd.DataFrame, col_zipcode: str, fill_str: str) -> pd.DataFrame:
    try:
        # Standardize Zipcode
        df[col_zipcode] = df[col_zipcode].fillna(fill_str).replace('', fill_str)
        logging.info(f"Standardized pcrb column '{col_zipcode}' by replacing null and empty values with '{fill_str}'.")
    except Exception as e:
        logging.error(f"An error occurred while standardizing pcrb column '{col_zipcode}': {e}")
        raise ValueError(f"An error occurred while standardizing pcrb column '{col_zipcode}': {e}")
    return df
    
def standardize_pcrb_str_address(df: pd.DataFrame, col_address: str, fill_str: str) -> pd.DataFrame:
    try:
        # Standardize Street Address
        address_patterns = ['STATE OF PA', 'STATE OF DE', 'PA', 'DE', 'NO SPECIFIC LOCATION']
        df[col_address] = df[col_address].replace(address_patterns, fill_str)
        # Standardize StreetAddress_Rev if it exists in the DataFrame
        col_address_rev = f'{col_address}_Rev'
        if col_address_rev in df.columns:
            df[col_address_rev] = df[col_address_rev].replace(address_patterns, fill_str)
        logging.info(f"Standardized pcrb columns '{col_address}' (and '{col_address_rev}' if present) by replacing patterns {address_patterns} with {fill_str}.")
    except Exception as e:
        logging.error(f"An error occurred while standardizing pcrb columns '{col_address}' and '{col_address_rev}': {e}")
        raise ValueError(f"An error occurred while standardizing pcrb columns '{col_address}' and '{col_address_rev}': {e}")
    return df

def standardize_pcrb_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Standardize PCRB values for FEIN, City, Zipcode, and Street Address.

    Args:
        df (pd.DataFrame): The DataFrame containing PCRB data.

    Returns:
        pd.DataFrame: The standardized DataFrame.
    """
    cols = ['PrimaryFEIN', 'City', 'Zipcode', 'StreetAddress']
    # Verify all columns in cols exist in the DataFrame
    missing_cols = [col for col in cols if col not in df.columns]
    if missing_cols:
        logging.error(f"Missing columns in DataFrame: {missing_cols}")
        raise ValueError(f"The following columns are missing in the DataFrame: {missing_cols}")
    # Standardize 'PrimaryFEIN' column in pcrb dataset
    df = standardize_pcrb_FEIN(df, cols[0])
    # Standardize columns 'city' and 'zipcode'
    df = standardize_pcrb_city(df, cols[1], 'MISSING')
    df = standardize_pcrb_zipcode(df, cols[2], '99999')
    # Standardize column 'StreetAddress'
    df = standardize_pcrb_str_address(df, cols[3], 'MISSING')
    return df

def add_record_ids(df_pcrb: pd.DataFrame, df_dnb: pd.DataFrame, col_id: str) -> tuple:
    """
    Add row IDs to both PCRB dataset and DNB dataset.

    Args:
        df_pcrb (pd.DataFrame): the PCRB dataset.
        df_dnb (pd.DataFrame): the DNB dataset.
        col_id (str): the column name to be created.

    Returns:
        tuple: updated PCRB and DNB datasets with new column for row IDs.
    """

    # Check if the datasets are not empty
    if df_pcrb is None or df_pcrb.empty:
        logging.error("df_pcrb DataFrame is empty")
        raise ValueError("df_pcrb DataFrame is empty")
    if df_dnb is None or df_dnb.empty:
        logging.error("df_dnb DataFrame is empty")
        raise ValueError("df_dnb DataFrame is empty")
    
    # Check if col_id already exists in either DataFrame
    if col_id in df_pcrb.columns:
        logging.error(f"Column '{col_id}' already exists in df_pcrb")
        raise ValueError(f"Column '{col_id}' already exists in df_pcrb")
    if col_id in df_dnb.columns:
        logging.error(f"Column '{col_id}' already exists in df_dnb")
        raise ValueError(f"Column '{col_id}' already exists in df_dnb")

    try:
        # Add row IDs
        df_pcrb[col_id] = range(1, len(df_pcrb) + 1)
        logging.info(f"Added column '{col_id}' to pcrb dataset")
        df_dnb[col_id] = range(1, len(df_dnb) + 1)
        logging.info(f"Added column '{col_id}' to dnb dataset")
    except Exception as e:
        logging.exception(f"An error occurred while adding record IDs: {e}")
        raise RuntimeError(f"An error occurred while adding record IDs: {e}")

    return df_pcrb, df_dnb

def convert_cols_to_int(df: pd.DataFrame, cols: list, replace_null: bool) -> pd.DataFrame:
    """
    Converts specified columns in a DataFrame to integer type, with options for replacing nulls.

    Args:
        df (pd.DataFrame): The input DataFrame.
        cols (list): List of columns to convert to integer.
        replace_null (bool): If True, replaces empty strings with NaN values before conversion.

    Returns:
        pd.DataFrame: The DataFrame with specified columns converted to integer type.

    Raises:
        TypeError: If df is not a DataFrame, or cols is not a list, or replace_null is not a boolean.
        ValueError: If any column in cols does not exist in df or contains non-numeric data.
    """
    
    # Check that df is a DataFrame
    if not isinstance(df, pd.DataFrame):
        logging.error("The df argument is not a pandas DataFrame.")
        raise TypeError("The df argument must be a pandas DataFrame.")
    
    # Check that cols is a list
    if not isinstance(cols, list):
        logging.error("The cols argument is not a list of column names.")
        raise TypeError("The cols argument must be a list of column names.")
    
    # Verify all columns in cols exist in the DataFrame
    missing_cols = [col for col in cols if col not in df.columns]
    if missing_cols:
        logging.error(f"Missing columns in DataFrame: {missing_cols}")
        raise ValueError(f"The following columns are missing in the DataFrame: {missing_cols}")
    
    # Iterate over each column to handle conversion and null replacement
    for column in cols:
        try:
            # Replace empty strings with NaN if replace_null is True
            if replace_null:
                df[column] = df[column].replace(r'^\s*$', np.nan, regex=True)
            # Attempt to convert column to integer, handling errors if non-numeric data is present
            df[column] = pd.to_numeric(df[column], errors='coerce').astype('Int64')
        except ValueError as ve:
            logging.error(f"Column '{column}' contains non-numeric values and cannot be converted to integer.")
            raise ValueError(f"Column '{column}' contains non-numeric values and cannot be converted to integer.") from ve
        except Exception as e:
            logging.error(f"Unexpected error while converting column '{column}' to integer: {e}")
            raise RuntimeError(f"An unexpected error occurred while converting column '{column}' to integer: {e}")

    return df

def convert_cols_to_str(df: pd.DataFrame, cols: list) -> pd.DataFrame:
    """
    Converts specified columns in a DataFrame to string type.

    Args:
        df (pd.DataFrame): The input DataFrame.
        cols (list): List of columns to convert to string.

    Returns:
        pd.DataFrame: The DataFrame with specified columns converted to string type.

    Raises:
        TypeError: If df is not a DataFrame or if cols is not a list.
        ValueError: If any column in cols does not exist in df.
    """
    
    # Check that df is a DataFrame
    if not isinstance(df, pd.DataFrame):
        logging.error("The df argument is not a pandas DataFrame.")
        raise TypeError("The df argument must be a pandas DataFrame.")
    
    # Check that cols is a list
    if not isinstance(cols, list):
        logging.error("The cols argument is not a list of column names.")
        raise TypeError("The cols argument must be a list of column names.")
    
    # Verify all columns in cols exist in the DataFrame
    missing_cols = [col for col in cols if col not in df.columns]
    if missing_cols:
        logging.error(f"Missing columns in DataFrame: {missing_cols}")
        raise ValueError(f"The following columns are missing in the DataFrame: {missing_cols}")
    
    # Iterate over each column to handle conversion to string
    for column in cols:
        try:
            # Attempt to convert column to string type
            df[column] = df[column].astype(pd.StringDtype())
        except Exception as e:
            logging.error(f"Unexpected error while converting column '{column}' to string: {e}")
            raise RuntimeError(f"An unexpected error occurred while converting column '{column}' to string: {e}")

    return df

def convert_pcrb_cols_type(df_pcrb: pd.DataFrame, cols_to_int: list, cols_to_str: list) -> pd.DataFrame:
    """
    Converts specified columns in the PCRB DataFrame to integer and string types.

    Args:
        df_pcrb (pd.DataFrame): The PCRB dataset to process.
        cols_to_int (list): List of columns to convert to integer type.
        cols_to_str (list): List of columns to convert to string type.

    Returns:
        pd.DataFrame: The DataFrame with specified columns converted to the respective types.

    Raises:
        ValueError: If an error occurs in the conversion process, the error will be logged and raised.
    """
    
    try:
        # Convert specified columns to integer type
        df_pcrb = convert_cols_to_int(df_pcrb, cols_to_int, True)
        logging.info(f"Converted pcrb columns {cols_to_int} to integer type.")
    except Exception as e:
        logging.error(f"Failed to convert columns {cols_to_int} to integer type: {e}")
        raise ValueError(f"Failed to convert columns {cols_to_int} to integer type: {e}")
    
    try:
        # Convert specified columns to string type
        df_pcrb = convert_cols_to_str(df_pcrb, cols_to_str)
        logging.info(f"Converted pcrb columns {cols_to_str} to integer type.")
    except Exception as e:
        logging.error(f"Failed to convert columns {cols_to_str} to string type: {e}")
        raise ValueError(f"Failed to convert columns {cols_to_str} to string type: {e}")

    return df_pcrb

def convert_dnb_cols_type(df_dnb: pd.DataFrame, cols_to_int: list, cols_to_str: list) -> pd.DataFrame:
    """
    Converts specified columns in the DNB DataFrame to integer and string types, including
    special handling for the fifth column in the cols_to_str list (extracts the first 5 digits).

    Args:
        df_dnb (pd.DataFrame): The DNB dataset to process.
        cols_to_int (list): List of columns to convert to integer type.
        cols_to_str (list): List of columns to convert to string type, with special handling for the 5th column.

    Returns:
        pd.DataFrame: The DataFrame with specified columns converted to the respective types.

    Raises:
        ValueError: If an error occurs in the conversion process, the error will be logged and raised.
    """
    
    try:
        # Convert specified columns to integer type
        df_dnb = convert_cols_to_int(df_dnb, cols_to_int, False)
        logging.info(f"Converted dnb columns {cols_to_int} to integer type.")
    except Exception as e:
        logging.error(f"Failed to convert columns {cols_to_int} to integer type: {e}")
        raise ValueError(f"Failed to convert columns {cols_to_int} to integer type: {e}")

    try:
        # Special handling for the 5th column in cols_to_str - extract the first 5 digits as string
        if len(cols_to_str) > 4:  # Ensure that the 5th column exists in the list
            df_dnb[cols_to_str[4]] = df_dnb[cols_to_str[4]].astype(str).str.extract(r'(\d{5})')[0]
            
    except Exception as e:
        logging.error(f"Failed to extract first 5 digits from column '{cols_to_str[4]}': {e}")
        raise ValueError(f"Failed to extract first 5 digits from column '{cols_to_str[4]}': {e}")

    try:
        # Convert other specified columns to string type
        df_dnb = convert_cols_to_str(df_dnb, cols_to_str)
        logging.info(f"Converted dnb columns {cols_to_str} to string type.")
    except Exception as e:
        logging.error(f"Failed to convert columns {cols_to_str} to string type: {e}")
        raise ValueError(f"Failed to convert columns {cols_to_str} to string type: {e}")

    return df_dnb



@cache_to_csv(filepath="df_pcrb_v2.csv")
def create_pcrb_cols(df_pcrb: pd.DataFrame, new_cols: list, col_insured_name: str, common_words: set, legal_suffixes: set, abbrev_mapping: dict, directions: dict, number_words: dict) -> pd.DataFrame:
    """
    Create standardized columns for PCRB DataFrame based on the insured name.

    Args:
        df_pcrb (pd.DataFrame): The PCRB DataFrame to process.
        new_cols (list): List of names for the new columns to be created.
        col_insured_name (str): Column name for the insured name.
        common_words (set): Set of common words to remove during standardization.
        legal_suffixes (set): Set of legal suffixes to remove during standardization.
        abbrev_mapping (dict): Dictionary mapping words to abbreviations.
        directions (dict): Dictionary mapping directions to abbreviations.
        number_words (dict): Dictionary mapping numbers as words to their numeric form.

    Returns:
        pd.DataFrame: The updated DataFrame with new standardized columns.
    """
    
    try:
        # Ensure the col_insured_name column exists
        if col_insured_name not in df_pcrb.columns:
            logging.error(f"The column '{col_insured_name}' is missing in the pcrb DataFrame.")
            raise ValueError(f"The column '{col_insured_name}' is missing in the pcrb DataFrame.")

        # Create the first standardized column without removing spaces
        df_pcrb[new_cols[0]] = df_pcrb[col_insured_name].apply(
            lambda x: standardize_business_name(
                x, common_words, legal_suffixes, abbrev_mapping, directions, number_words, remove_spaces=False
            )
        )
        logging.info(f"Created column '{new_cols[0]}' with standardized business names (spaces preserved).")

        # Create the second standardized column with spaces removed
        df_pcrb[new_cols[1]] = df_pcrb[col_insured_name].apply(
            lambda x: standardize_business_name(
                x, common_words, legal_suffixes, abbrev_mapping, directions, number_words, remove_spaces=True
            )
        )
        logging.info(f"Created column '{new_cols[1]}' with standardized business names (spaces removed).")
        logging.info("Completed creating all the business name columns in pcrb dataset")
        return df_pcrb

    except Exception as e:
        logging.error(f"An error occurred while creating PCRB columns: {e}")
        raise ValueError(f"An error occurred while creating PCRB columns: {e}")

@cache_to_csv(filepath="df_dnb_v2.csv")
def create_dnb_cols(df_dnb: pd.DataFrame, new_cols: list, dnb_cols: str, common_words: set, legal_suffixes: set, abbrev_mapping: dict, directions: dict, number_words: dict) -> pd.DataFrame:
    """
    Create standardized columns for DNB DataFrame based on specified DNB columns.

    Args:
        df_dnb (pd.DataFrame): The DNB DataFrame to process.
        new_cols (list): List of names for the new columns to be created.
        dnb_cols (list): List of existing DNB columns to standardize.
        common_words (set): Set of common words to remove during standardization.
        legal_suffixes (set): Set of legal suffixes to remove during standardization.
        abbrev_mapping (dict): Dictionary mapping words to abbreviations.
        directions (dict): Dictionary mapping directions to abbreviations.
        number_words (dict): Dictionary mapping numbers as words to their numeric form.

    Returns:
        pd.DataFrame: The updated DataFrame with new standardized columns.
    """
    
    try:
        # Check if the required columns in dnb_cols exist in df_dnb
        missing_cols = [col for col in dnb_cols if col not in df_dnb.columns]
        if missing_cols:
            logging.error(f"The following columns are missing in the DataFrame: {missing_cols}")
            raise ValueError(f"The following columns are missing in the DataFrame: {missing_cols}")
        
        # Standardize the first DNB column without removing spaces
        df_dnb[new_cols[0]] = df_dnb[dnb_cols[0]].apply(
            lambda x: standardize_business_name(
                x, common_words, legal_suffixes, abbrev_mapping, directions, number_words, remove_spaces=False
            )
        )
        logging.info(f"Created column '{new_cols[0]}' from '{dnb_cols[0]}' with spaces preserved.")

        # Standardize the first DNB column with spaces removed
        df_dnb[new_cols[1]] = df_dnb[dnb_cols[0]].apply(
            lambda x: standardize_business_name(
                x, common_words, legal_suffixes, abbrev_mapping, directions, number_words, remove_spaces=True
            )
        )
        logging.info(f"Created column '{new_cols[1]}' from '{dnb_cols[0]}' with spaces removed.")

        # Standardize the second DNB column without removing spaces
        df_dnb[new_cols[2]] = df_dnb[dnb_cols[1]].apply(
            lambda x: standardize_business_name(
                x, common_words, legal_suffixes, abbrev_mapping, directions, number_words, remove_spaces=False
            )
        )
        logging.info(f"Created column '{new_cols[2]}' from '{dnb_cols[1]}' with spaces preserved.")
        logging.info("Completed creating all the business name columns in dnb dataset")
        return df_dnb

    except Exception as e:
        logging.error(f"An error occurred while creating DNB columns: {e}")
        raise ValueError(f"An error occurred while creating DNB columns: {e}")


# Preprocessing function with exception handling
def preprocess(name):
    """
    Preprocesses a given name by stripping whitespace and converting it to a string.

    Args:
        name (str): The name to preprocess.

    Returns:
        str: The preprocessed name, or an empty string if an exception occurs.
    """
    try:
        return str(name).strip()

    except Exception:
        return ""

# Function to build Soundex map
def build_soundex_map(names):
    """
    Builds a Soundex map from a list of names. Each name is converted to its Soundex code
    and grouped accordingly.

    Args:
        names (list): A list of business names.

    Returns:
        defaultdict: A dictionary where the keys are Soundex codes and the values are lists of names.
    """

    soundex_map = defaultdict(list)

    for name in names:
        soundex_code = jellyfish.soundex(name)
        soundex_map[soundex_code].append(name)

    return soundex_map

# Function to match a single PCRB name
def match_pcrb_name(args):
    """
    Matches a single PCRB name against possible DNB business names based on Soundex and fuzzy matching.

    Args:

        args (tuple): A tuple containing:

            - pcrb_name (str): The name from PCRB to match.

            - soundex_to_dnb_map (dict): A map of Soundex codes to DNB names.

            - dnb_original_map (dict): A map of processed DNB names to their original names.

            - matched_dnb_proxy (dict): A shared dictionary to track already matched DNB names.

    Returns:

        tuple: A tuple of (pcrb_name, matched_dnb_name, match_type). If no match is found, returns (pcrb_name, None, None).

    """
    pcrb_name, soundex_to_dnb_map, dnb_original_map, matched_dnb_proxy = args
    soundex_code = jellyfish.soundex(pcrb_name)

    possible_dnb = soundex_to_dnb_map.get(soundex_code, [])

    if not possible_dnb:
        return (pcrb_name, None, None)  # No possible match

    # Get the best possible match
    match = process.extractOne(
        pcrb_name,
        possible_dnb,
        scorer=fuzz.token_set_ratio
    )

    if match:
        match_name, score, _ = match
        dnb_orig = dnb_original_map.get(match_name)

        if dnb_orig and dnb_orig not in matched_dnb_proxy:
            matched_dnb_proxy[dnb_orig] = True  # Mark as matched

            # Determine match type based on the score
            if score >= 90:
                match_type = 'high'

            elif 80 <= score < 90:
                match_type = 'med'

            elif 75 <= score < 80:
                match_type = 'low'

            else:
                match_type = None  # Below threshold

            return (pcrb_name, dnb_orig, match_type)

    return (pcrb_name, None, None)  # No suitable match found

def generate_maps(pcrb_business, dnb_business):
    """
    Generates similarity maps for business names between PCRB and DNB datasets using Soundex and fuzzy matching.

    Args:

        pcrb_business (set): A set of unique business names from the PCRB dataset.

        dnb_business (set): A set of unique business names from the DNB dataset, including global ultimate names.

    Returns:

        tuple: Three dictionaries representing high, medium, and low similarity matches. Each dictionary maps PCRB names

               to their corresponding best-matched DNB names.

    Note:

        - If running on Windows, ensure this function is called within the `if __name__ == "__main__":` block due to multiprocessing.

    """

    pcrb_business_name = set(pcrb_business)
    dnb_business_name = set(dnb_business)

    pcrb_only_business = pcrb_business_name - dnb_business_name
    dnb_only_business = dnb_business_name - pcrb_business_name

    pcrb_only_business = list(pcrb_only_business)
    dnb_only_business = list(dnb_only_business)

    # Preprocess PCRB and DNB names
    pcrb_processed = [preprocess(biz) for biz in pcrb_only_business]
    dnb_processed = [preprocess(biz) for biz in dnb_only_business]

    # Build Soundex map for DNB
    soundex_to_dnb_map = build_soundex_map(dnb_processed)

    # Build mapping from processed DNB names to original DNB names
    dnb_original_map = {name: name for name in dnb_processed}

    # Initialize Manager for shared data structures
    manager = Manager()
    matched_dnb = manager.dict()  # To track matched DNB names

    # Prepare arguments for multiprocessing
    matching_args = [
        (pcrb, soundex_to_dnb_map, dnb_original_map, matched_dnb)
        for pcrb in pcrb_processed
    ]

    # Initialize dictionaries to store matches
    high_similarity_matches = {}
    low_similarity_matches = {}
    med_similarity_matches = {}

    # Function to aggregate results
    def aggregate_result(result):
        """
        Aggregates the result of matching by adding matches to the appropriate similarity dictionary.

        Args:
            result (tuple): The result tuple of (pcrb_name, dnb_name, match_type).
        """

        pcrb, dnb, match_type = result

        if match_type == 'high':
            high_similarity_matches[pcrb] = dnb

        elif match_type == 'med':
            med_similarity_matches[pcrb] = dnb

        elif match_type == 'low':
            low_similarity_matches[pcrb] = dnb

    # Use multiprocessing Pool for parallel processing
    with Pool(processes=cpu_count()) as pool:
        for result in pool.imap_unordered(match_pcrb_name, matching_args, chunksize=1000):
            aggregate_result(result)

    # Return the similarity maps
    return high_similarity_matches, med_similarity_matches, low_similarity_matches 












########################################################################################################################################
# The methods below are not used by the current pipeline due to most recent updates in the matching solution.

# def standardize_address(address: str, abbrev_mapping: dict, directions: dict) -> str:
#     """
#     Standardize address format with improved handling of edge cases.

#     This function standardizes an address by converting it to uppercase, replacing full directional words with abbreviations,
#     replacing full street suffix words or common abbreviations with their postal service standard abbreviations,
#     and cleaning up punctuation and extra spaces.

#     Args:
#         address (str): The address string to be standardized.
#         abbrev_mapping (dict): A dictionary mapping full words and common abbreviations to standard abbreviations.
#         directions (dict): A dictionary mapping full directional words to their abbreviations.

#     Returns:
#         str: The standardized address.

#     Raises:
#         TypeError: If 'address' is not a string or if 'abbrev_mapping' or 'directions' are not dictionaries.
#         ValueError: If 'address' is empty.
#     """
#     if not isinstance(abbrev_mapping, dict) or not isinstance(directions, dict):
#         raise TypeError("Both 'abbrev_mapping' and 'directions' must be dictionaries.")

#     if pd.isna(address):
#         return address
    
#     # Convert to uppercase
#     address = address.upper()

#     # Remove leading parentheses and their contents
#     address = re.sub(r'^\s*\([^)]*\)\s*', '', address)
    
#     # Replace full directional words with abbreviations
#     for direction, abbr in directions.items():
#         address = re.sub(r'\b' + direction + r'\b', abbr, address)
    
#     # Replace full street suffix words or common abbreviations with their postal service standard abbreviations
#     for word, abbr in abbrev_mapping.items():
#         if word:  # Ensure the word is not NaN
#             address = re.sub(r'\b' + word + r'\b', abbr, address)
    
#     # Remove periods and commas
#     address = address.replace('.', '').replace(',', '').replace('#', '')
    
#     # Remove extra spaces
#     address = ' '.join(address.split())
    
#     return address

# def standardize_address_df(df: pd.DataFrame, col_names: list, abbrev_mapping: dict, directions: dict, df_name: str) -> pd.DataFrame:
#     """
#     Standardize addresses in a DataFrame using provided column names and an abbreviation mapping.

#     Args:
#         df (pd.DataFrame): DataFrame containing the addresses to be standardized.
#         col_names (list): List of column names; col_names[3] should contain the addresses to standardize,
#                           col_names[4] will store the standardized addresses.
#         abbrev_mapping (dict): A dictionary with abbreviations used to standardize addresses.

#     Returns:
#         pd.DataFrame: DataFrame with the standardized address in col_names[4].

#     Raises:
#         ValueError: If col_names does not have enough elements or the specified columns do not exist in the DataFrame.
#         KeyError: If the DataFrame does not contain one of the specified columns.
#         TypeError: If 'abbrev_mapping' is not a dictionary.
#     """
#     # Check input validity
#     if not isinstance(df, pd.DataFrame):
#         raise TypeError("The provided 'df' must be a pandas DataFrame.")

#     if not isinstance(abbrev_mapping, dict):
#         raise TypeError("'abbrev_mapping' must be a dictionary.")

#     if len(col_names) < 5:
#         raise ValueError("The 'col_names' list must contain at least 5 elements for addressing column indices.")

#     # Apply address standardization
#     try:
#         df[col_names[4]] = df[col_names[3]].apply(lambda x: standardize_address(x, abbrev_mapping, directions))
#     except Exception as e:
#         raise Exception(f"An error occurred during address standardization: {str(e)}")

#     print(f"Completed standardizing the column '{col_names[4]}' in {df_name}")
#     return df

# def standardize_business_name_no_spaces(name, common_words: set, legal_suffixes: set, abbrev_mapping: dict, directions: dict) -> str:
#     """
#     Standardize the format of a business name by removing spaces, handling abbreviations,
#     removing legal suffixes, and cleaning up common words and special characters.

#     Args:
#         name (str): The business name to standardize.
#         common_words (set): A set of common words to remove from the business name.
#         legal_suffixes (set): A set of legal suffixes to remove from the business name.
#         abbrev_mapping (dict): A dictionary mapping full words to their abbreviations.
#         directions (dict): A dictionary mapping directional words to their abbreviations.

#     Returns:
#         str: The standardized business name with all spaces removed.
#     """
#     if pd.isna(name):
#         return name
#     # Convert to uppercase
#     name = name.upper()
#     # Separate attached suffixes
#     name = separate_suffix(name, legal_suffixes)
#     # Remove legal suffixes
#     for suffix in legal_suffixes:
#         name = re.sub(r'\b' + re.escape(suffix) + r'\b', '', name)
#     # Replace full directional words with abbreviations
#     for direction, abbr in directions.items():
#         name = re.sub(r'\b' + direction + r'\b', abbr, name)
#     # Replace full street suffix words or common abbreviations with their postal service standard abbreviations
#     for word, abbr in abbrev_mapping.items():
#         if word:  # Ensure the word is not NaN
#             name = re.sub(r'\b' + re.escape(word) + r'\b', abbr, name)
#     # Remove punctuation and special characters
#     name = re.sub(r'[^\w]', '', name)
#     # Replace & with 'AND'
#     name = name.replace('&', 'AND')
#     # Remove common words
#     for word in common_words:
#         name = re.sub(r'\b' + re.escape(word) + r'\b', '', name)
#     return name

# def standardize_business_name_pcrb(df_pcrb: pd.DataFrame, col_names: list, common_words: set, legal_suffixes: set, abbrev_mapping: dict, directions: dict) -> pd.DataFrame:
#     """
#     Standardize business names in the pcrb DataFrame using specified rules and mappings.

#     This function applies transformations to clean and standardize business names by removing spaces,
#     applying abbreviations, removing legal suffixes, replacing directions, and removing common words.

#     Args:
#         df_pcrb (pd.DataFrame): DataFrame containing the business names to standardize.
#         col_names (list): List of column names used in the DataFrame where col_names[6] is the source
#                           column for business names and col_names[5] will store the standardized names.
#         common_words (set): Set of common words to remove from business names.
#         legal_suffixes (set): Set of legal suffixes to remove from business names.
#         abbrev_mapping (dict): Dictionary mapping full words to their abbreviations.
#         directions (dict): Dictionary mapping full directional words to their abbreviations.

#     Returns:
#         pd.DataFrame: The DataFrame with standardized business names.

#     Raises:
#         ValueError: If 'col_names' does not have enough elements, or the DataFrame does not contain necessary columns.
#         TypeError: If any of the set or dictionary inputs are not of the correct type.
#     """
#     # Check the adequacy and types of inputs
#     if not isinstance(df_pcrb, pd.DataFrame):
#         raise TypeError("The provided 'df_pcrb' must be a pandas DataFrame.")
#     if not isinstance(col_names, list) or len(col_names) < 7:
#         raise ValueError("The 'col_names' list must contain at least 7 elements.")
#     if not isinstance(common_words, set) or not isinstance(legal_suffixes, set) or not isinstance(abbrev_mapping, dict) or not isinstance(directions, dict):
#         raise TypeError("'common_words', 'legal_suffixes', 'abbrev_mapping', and 'directions' must be set or dictionary types accordingly.")
#     if col_names[6] not in df_pcrb.columns:
#         raise ValueError(f"The pcrb DataFrame is missing column '{col_names[6]}'")
#     # Apply standardization function
#     try:
#         df_pcrb[col_names[5]] = df_pcrb[col_names[6]].apply(lambda x: standardize_business_name_no_spaces(x, common_words, legal_suffixes, abbrev_mapping, directions))
#     except Exception as e:
#         raise Exception(f"An error occurred during standardizing business names in pcrb dataset: {str(e)}")
#     print(f"Completed standardizing the column '{col_names[5]}' in pcrb dataset")
#     return df_pcrb


            
# def standardize_business_name_dnb(df_dnb: pd.DataFrame, col_names: list, common_words: set, legal_suffixes: set, abbrev_mapping: dict, directions: dict) -> pd.DataFrame:
#     """
#     Standardize business names in the dnb DataFrame using specified rules and mappings.

#     This function applies transformations to clean and standardize business names by removing spaces,
#     applying abbreviations, removing legal suffixes, replacing directions, and removing common words.

#     Args:
#         df_pcrb (pd.DataFrame): DataFrame containing the business names to standardize.
#         col_names (list): List of column names used in the DataFrame where col_names[6] is the source
#                           column for business names and col_names[5] will store the standardized names.
#         common_words (set): Set of common words to remove from business names.
#         legal_suffixes (set): Set of legal suffixes to remove from business names.
#         abbrev_mapping (dict): Dictionary mapping full words to their abbreviations.
#         directions (dict): Dictionary mapping full directional words to their abbreviations.

#     Returns:
#         pd.DataFrame: The DataFrame with standardized business names.

#     Raises:
#         ValueError: If 'col_names' does not have enough elements, or the DataFrame does not contain necessary columns.
#         TypeError: If any of the set or dictionary inputs are not of the correct type.
#     """
#     # Check the adequacy and types of inputs
#     if not isinstance(df_dnb, pd.DataFrame):
#         raise TypeError("The provided 'df_pcrb' must be a pandas DataFrame.")
#     if not isinstance(col_names, list) or len(col_names) < 7:
#         raise ValueError("The 'col_names' list must contain at least 7 elements.")
#     if not isinstance(common_words, set) or not isinstance(legal_suffixes, set) or not isinstance(abbrev_mapping, dict) or not isinstance(directions, dict):
#         raise TypeError("'common_words', 'legal_suffixes', 'abbrev_mapping', and 'directions' must be set or dictionary types accordingly.")
#     if col_names[7] not in df_dnb.columns:
#         raise ValueError(f"The pcrb DataFrame is missing column '{col_names[6]}'")
#     try:     
#         df_dnb[col_names[5]] = df_dnb[col_names[7]].apply(lambda x: standardize_business_name_no_spaces(x, common_words, legal_suffixes, abbrev_mapping, directions))
#     except Exception as e:
#         raise Exception(f"An error occurred during standardizing business names in dnb dataset: {str(e)}")
#     print(f"Completed standardizing the column '{col_names[5]}' in dnb dataset")
#     return df_dnb

# def append_FEIN_address(df_pcrb: pd.DataFrame, col_names: list) -> pd.DataFrame:
#     """
#     Appends a new column to a DataFrame that combines FEIN and address data.

#     This function creates a new column in the DataFrame by concatenating FEIN and address columns,
#     separated by an underscore, with appropriate handling for missing values.

#     Args:
#         df_pcrb (pd.DataFrame): DataFrame containing the FEIN and address data.
#         col_names (list): List of column names where col_names[9] is FEIN and col_names[4] is the address.

#     Returns:
#         pd.DataFrame: The DataFrame with an appended column combining FEIN and address.

#     Raises:
#         ValueError: If 'col_names' does not have enough elements or the DataFrame does not contain necessary columns.
#         TypeError: If 'df_pcrb' is not a pandas DataFrame or 'col_names' is not a list.
#     """
#     # Validate input types
#     if not isinstance(df_pcrb, pd.DataFrame):
#         raise TypeError("The provided 'df_pcrb' must be a pandas DataFrame.")
#     if not isinstance(col_names, list) or len(col_names) < 10:
#         raise ValueError("The 'col_names' list must contain at least 10 elements.")
#     # Ensure required columns are in the DataFrame
#     required_columns = [col_names[9], col_names[4]]  # FEIN and address columns
#     missing_columns = [col for col in required_columns if col not in df_pcrb.columns]
#     if missing_columns:
#         raise ValueError(f"The DataFrame is missing required columns: {', '.join(missing_columns)}")
#     # Append new column
#     df_pcrb[col_names[8]] = df_pcrb[col_names[9]].fillna('') + '_' + df_pcrb[col_names[4]].fillna('')
#     print(f"Appended column '{col_names[8]}' to pcrb dataset")
#     return df_pcrb

# # Create a column full_address
# df_pcrb = data_cleaning.append_full_address_pcrb(df_pcrb, self.col_names[3])
# df_dnb = data_cleaning.append_full_address_dnb(df_dnb, self.col_names[3])
# # Standardize the full_address for df_pcrb and df_dnb
# df_pcrb = data_cleaning.standardize_address_df(df_pcrb, self.col_names, abbrev_mapping, self.directions, "pcrb dataset")
# df_dnb = data_cleaning.standardize_address_df(df_dnb, self.col_names, abbrev_mapping, self.directions, "dnb dataset")
# # Standardize business names for df_pcrb and df_dnb
# df_pcrb = data_cleaning.standardize_business_name_pcrb(df_pcrb, self.col_names, common_words, legal_suffixes, abbrev_mapping, self.directions)
# df_dnb = data_cleaning.standardize_business_name_dnb(df_dnb, self.col_names, common_words, legal_suffixes, abbrev_mapping, self.directions)
# Create FEIN_Address column for df_pcrb
# df_pcrb = data_cleaning.append_FEIN_address(df_pcrb, self.col_names)
# # Apply the hierarchy method on pcrb dataset to reduce the number of rows to work with
# df_pcrb = data_cleaning.apply_hierarchy_method(df_pcrb, self.col_names)