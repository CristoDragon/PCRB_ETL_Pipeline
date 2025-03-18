import pandas as pd
import numpy as np
from sqlalchemy import create_engine
from fuzzywuzzy import fuzz
import re
from tqdm import tqdm
import configparser
import traceback
from functools import wraps
import src.Data_Collection as data_collection
import src.Data_Cleaning as data_cleaning
import src.Model_Building as model_building
import logging
from src.Model_Evaluation import log_matching_metrics, get_latest_evaluation_metrics


# Date: 11/11/2024
# Author: CMU Capstone Team (Dragon, Michael, Nirvik, Karl)
# Description: This class contains pipeline details and methods to execute cleaning, processing, and matching jobs.

class Pipeline:
    # Initialize all the instance variables
    def __init__(self, config: configparser, RUN_ID: str):
        self.config = config
        self.RUN_ID = RUN_ID
        self.db_pcrb = 'PCRB_DB'
        self.db_dnb = 'DNB_DB'
        self.db_snowflake = 'SNOWFLAKE_DB'
        self.pcrb_query = 'pcrb_query'
        self.dnb_query = 'dnb_query'
        self.col_names = [
            'Primary Name', 
            'Commonly Used Suffix or Abbreviation',
            'Postal Service Standard Abbreviation',
            'full_address',
            'standardized_address',
            'standardized_business_name',
            'PrimaryInsuredName',
            'business_name',
            'FEIN_Address',
            'PrimaryFEIN',
            'FEIN_Address',
            'AddressTypeCode',
            'record_id'
        ]
        self.directions = {
            'NORTH': 'N', 'SOUTH': 'S', 'EAST': 'E', 'WEST': 'W',
            'NORTHEAST': 'NE', 'NORTHWEST': 'NW', 'SOUTHEAST': 'SE', 'SOUTHWEST': 'SW'
        }
        self.business_name = 'BUSINESS_NAME'
        self.sql = "SQL"
        self.pcrb_cols_to_int = ['AddressTypeCode', 'NumberOfEmployees']
        self.pcrb_cols_to_str = ['Zipcode', 'FileNumber', 'PrimaryFEIN', 'PrimaryInsuredName', 
                                'StreetAddress', 'StreetAddress_Rev', 'City', 'State', 'InsuredPhoneNumber']
        self.dnb_cols_to_int = ['employees_total', 'employees_here']
        self.dnb_cols_to_str = ['physical_street_address', 'second_address_line',
                                'physical_city', 'physical_state_abbreviation', 'physical_zip', 
                                'tradestyle', 'second_tradestyle', 'global_ultimate_name', 
                                'domestic_ultimate_business_name']
        self.number_words = {
                '1ST': 'FIRST', '2ND': 'SECOND', '3RD': 'THIRD', '4TH': 'FOURTH',
                '5TH': 'FIFTH', '6TH': 'SIXTH', '7TH': 'SEVENTH', '8TH': 'EIGHTH',
                '9TH': 'NINTH', '10TH': 'TENTH', '1': 'ONE', '2': 'TWO', '3': 'THREE',
                '4': 'FOUR', '5': 'FIVE', '6': 'SIX', '7': 'SEVEN', '8': 'EIGHT',
                '9': 'NINE', '10': 'TEN'
            }
        self.new_cols_pcrb = ['standardized_business_name_with_spaces', 'standardized_business_name_no_spaces']
        self.new_cols_dnb = self.new_cols_pcrb + ['standardized_global_ultimate_name_with_spaces']
    
    def run(self):
        try:
            # 1. Data Collection:
            # Establish a database connection to SQL Server
            engine_pcrb = data_collection.create_db_connection(self.config, self.db_pcrb)
            engine_dnb = data_collection.create_db_connection(self.config, self.db_dnb)
            # Load the pcrb and dnb data from SQL Server
            df_pcrb = data_collection.load_data_pcrb(engine_pcrb, self.config[self.sql][self.pcrb_query])
            df_dnb = data_collection.load_data_dnb(engine_dnb, self.config[self.sql][self.dnb_query])
            # Load the abbreviations from the Excel file
            abbrev_mapping = data_collection.create_abbreviation_mapping(self.config['STREET_ABBR']['file_path'], self.col_names)
            # Load common words and legal suffixes from txt files
            common_words = data_collection.load_common_words(self.config[self.business_name]['common_words_file'])
            legal_suffixes = data_collection.load_legal_suffixes(self.config[self.business_name]['legal_suffixes_file'])
            
            # 2. Data Cleaning & Feature Engineering:
            # Add record IDs to pcrb and dnb dataset
            df_pcrb, df_dnb = data_cleaning.add_record_ids(df_pcrb, df_dnb, self.col_names[12])
            # Convert PCRB column types
            df_pcrb = data_cleaning.convert_pcrb_cols_type(df_pcrb, self.pcrb_cols_to_int, self.pcrb_cols_to_str)
            # Convert DNB column types
            df_dnb = data_cleaning.convert_dnb_cols_type(df_dnb, self.dnb_cols_to_int, self.dnb_cols_to_str)
            # Standardize PCRB values
            df_pcrb = data_cleaning.standardize_pcrb_values(df_pcrb)
            # Apply address standardization
            df_pcrb = data_cleaning.apply_pcrb_standardization(df_pcrb, df_dnb, abbrev_mapping, self.directions, ['StreetAddress', 'City', 'State', 'Zipcode'], 'physical_city')
            df_dnb = data_cleaning.apply_dnb_standardization(df_dnb, abbrev_mapping, self.directions, ['physical_street_address', 'physical_city', 'physical_state_abbreviation', 'physical_zip'])
            df_pcrb = data_cleaning.create_pcrb_col_FEIN_Address(df_pcrb, ['FEIN_Address', 'PrimaryFEIN', 'standardized_street_only'])
            # Standardize business names for pcrb dataset
            df_pcrb = data_cleaning.create_pcrb_cols(df_pcrb, self.new_cols_pcrb, self.col_names[6], common_words, legal_suffixes, abbrev_mapping, self.directions, self.number_words)
            # Standardize business names for dnb dataset
            df_dnb = data_cleaning.create_dnb_cols(df_dnb, self.new_cols_dnb, ['business_name', 'global_ultimate_name'], common_words, legal_suffixes, abbrev_mapping, self.directions, self.number_words)
            # Apply the hierarchy method on pcrb dataset to reduce the number of rows to work with
            df_pcrb = data_cleaning.apply_hierarchy_method(df_pcrb, self.col_names, self.RUN_ID)

            # # 3. Model Building for Matching Process
            # Data Preparation
            df_pcrb, df_dnb = model_building.clean_city_name(df_pcrb, df_dnb)
            # Create mappings for both datasets
            zip_to_cities_pcrb = model_building.create_zip_to_cities_mapping(df_pcrb)
            zip_to_cities_dnb = model_building.create_zip_to_cities_mapping(df_dnb)
            city_to_zips_pcrb = model_building.create_city_to_zips_mapping(df_pcrb)
            city_to_zips_dnb = model_building.create_city_to_zips_mapping(df_dnb)
            # Merge the mappings from both datasets
            zip_to_cities = model_building.get_mapping_ztc(zip_to_cities_pcrb, zip_to_cities_dnb)
            city_to_zips = model_building.get_mapping_ctz(city_to_zips_pcrb, city_to_zips_dnb)

            # Phase 1
            # group by Zip Code and count for each dataset after the mapping
            rec_before_match = len(df_pcrb)
            zip_count_pcrb = df_pcrb.groupby('Zip_UPDATED').size().reset_index(name='count_pcrb')
            zip_count_dnb = df_dnb.groupby('Zip_UPDATED').size().reset_index(name='count_dnb')
            # Merge the counts on Zip Code
            zip_counts = pd.merge(zip_count_pcrb, zip_count_dnb, on='Zip_UPDATED', how='outer').fillna(0)
            zip_counts['count_pcrb'] = zip_counts['count_pcrb'].astype(int)
            zip_counts['count_dnb'] = zip_counts['count_dnb'].astype(int)
            # Calculate total_comparisons
            zip_counts['total_comparisons'] = zip_counts['count_pcrb'] * zip_counts['count_dnb']
            # Filter out zip codes with total comparisons <= 20
            zip_counts_filtered = zip_counts[zip_counts['total_comparisons'] > 0]
            # Sort by total_comparisons in ascending order
            zip_counts_sorted = zip_counts_filtered.sort_values(by='total_comparisons', ascending=True)
            # Select top zip codes (you can adjust this number as needed)
            top_zip_codes_df = zip_counts_sorted
            top_zip_codes = top_zip_codes_df['Zip_UPDATED'].tolist()
            logging.info("Top zip codes by total comparisons:")
            logging.info(top_zip_codes_df)
            # Usage
            zip_counts = model_building.calculate_zip_counts(df_pcrb, df_dnb, 'Zip_UPDATED', True)
            total_comparisons_needed = zip_counts['total_comparisons'].sum() // 5
            batches = model_building.batch_processing(zip_counts, total_comparisons_needed, 'Zip_UPDATED')
            # Output the batches or process them as needed
            for i, batch in enumerate(batches):
                logging.info(f"Batch {i+1}: {batch}")
            #final_matched_df1, final_city_stats_df1, final_unmatched_pcrb_df1, final_unmatched_dnb_df1 = model_building.get_result_dfs_phase1(batches, df_pcrb, df_dnb, zip_to_cities)
            final_matched_df1 = model_building.get_result_dfs_phase1(batches, df_pcrb, df_dnb, zip_to_cities)
            rec_matched = len(final_matched_df1)
            df_pcrb, df_dnb = model_building.update_dfs(df_pcrb, df_dnb, final_matched_df1)

            # Phase 2
            # Usage
            zip_counts = model_building.calculate_zip_counts(df_pcrb, df_dnb, 'City_UPDATED', False)
            total_comparisons_needed = zip_counts['total_comparisons'].sum() // 30
            # You define 'desired_number_of_batches'
            batches = model_building.batch_processing(zip_counts, total_comparisons_needed, 'City_UPDATED')
            # Output the batches or process them as needed
            for i, batch in enumerate(batches):
                print(f"Batch {i+1}: {batch}")
            #final_matched_df2, final_city_stats_df2, final_unmatched_pcrb_df2, final_unmatched_dnb_df2 = model_building.get_result_dfs_phase2(batches, df_pcrb, df_dnb, city_to_zips)
            final_matched_df2 = model_building.get_result_dfs_phase2(batches, df_pcrb, df_dnb, city_to_zips)
            rec_matched += len(final_matched_df2)
            df_pcrb, df_dnb = model_building.update_dfs(df_pcrb, df_dnb, final_matched_df2)

            # Phase 3
            # Usage
            zip_counts = model_building.calculate_zip_counts(df_pcrb, df_dnb, 'Zip_UPDATED', False)
            total_comparisons_needed = zip_counts['total_comparisons'].sum() // 10  # You define 'desired_number_of_batches'
            batches = model_building.batch_processing(zip_counts, total_comparisons_needed, 'Zip_UPDATED')
            # Output the batches or process them as needed
            for i, batch in enumerate(batches):
                print(f"Batch {i+1}: {batch}")
            final_matched_df3, final_city_stats_df3, final_unmatched_pcrb_df3, final_unmatched_dnb_df3 = model_building.get_result_dfs_phase3(batches, df_pcrb, df_dnb, zip_to_cities)
            # final_city_stats_df3.to_csv("Phase_3_Zip_Stats.csv")
            with pd.ExcelWriter('Dashboard.xlsx', mode='a', if_sheet_exists='replace') as writer:
                final_city_stats_df3.to_excel(writer, sheet_name='Phase 3')
                logging.info("final_city_stats_df has been saved to 'Phase 3' sheet in Dashboard.xlsx")
            num_of_unmatched = len(final_unmatched_pcrb_df3)
            rec_matched += len(final_matched_df3)
            log_matching_metrics( "Number of matched rows",rec_before_match, rec_matched, self.RUN_ID)
            log_matching_metrics( "Number of unmatched rows",rec_before_match, num_of_unmatched, self.RUN_ID)
            df_pcrb, df_dnb = model_building.update_dfs(df_pcrb, df_dnb, final_matched_df3)

            # The following code has not been tested yet due to Duo authentication issues
            # # Save the matched and unmatched records to the database
            # # Create a connection to snowflake database
            # conn_snowflake = data_collection.create_snowflake_conn(self.config, self.db_snowflake)
            # query_create_table = data_collection.construct_create_table_query('MATCHED_PCRB_DNB', final_matched_df3)
            # logging.info(query_create_table)
            # data_collection.execute_query_snowflake(conn_snowflake, query_create_table)
            # # Save the matched records to the snowflake database
            # data_collection.write_df_to_snowflake(final_matched_df3, conn_snowflake, 'MATCHED_PCRB_DNB')
            # # Close the Snowflake connection
            # conn_snowflake.close()
            # logging.info("Snowflake connection is closed.")
        except Exception as e:
            logging.error(f"Error in run(): {e}\n{traceback.format_exc()}")