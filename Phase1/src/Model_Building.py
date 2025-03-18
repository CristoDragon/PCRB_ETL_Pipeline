import os
from functools import wraps
import pandas as pd
import numpy as np 
from sklearn.metrics.pairwise import cosine_similarity
import re
import time
from rapidfuzz import fuzz
from sentence_transformers import SentenceTransformer, util
import warnings
import torch
import pickle
from collections import defaultdict
import gc
import logging

# Date: 10/26/2024
# Author: CMU Capstone Team (Dragon, Michael, Nirvik, Karl)
# Description: This module contains code relevant to the model building part in machine learning pipeline,
# which includes functions to do the matching process between pcrb and dnb dataset.

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

def clean_city_name(df_pcrb: pd.DataFrame, df_dnb: pd.DataFrame) -> tuple:
    df_pcrb['City_UPDATED'] = df_pcrb['standardized_city'].str.upper().str.strip()
    df_pcrb['Zip_UPDATED'] = df_pcrb['Zipcode']
    df_pcrb['Street_UPDATED'] = df_pcrb['standardized_street_only'].str.upper().str.strip()
    df_pcrb['BusinessName_UPDATED'] = df_pcrb['standardized_business_name_with_spaces'].str.upper().str.strip()
    df_dnb['City_UPDATED'] = df_dnb['physical_city'].str.upper().str.strip()
    df_dnb['Street_UPDATED'] = df_dnb['standardized_street_only'].str.upper().str.strip()
    df_dnb['Zip_UPDATED'] = df_dnb['physical_zip']
    df_dnb['BusinessName_UPDATED'] = df_dnb['standardized_business_name_with_spaces'].str.upper().str.strip()
    df_dnb['GlobalName_UPDATED'] = df_dnb['standardized_global_ultimate_name_with_spaces'].str.upper().str.strip()
    return df_pcrb, df_dnb

# Create mappings from Zipcode to Cities and vice versa
def create_zip_to_cities_mapping(df, zip_col='Zip_UPDATED', city_col='City_UPDATED') -> defaultdict:
    zip_to_cities = defaultdict(set)
    for _, row in df.iterrows():
        zip_code = row[zip_col]
        city = row[city_col]
        if pd.notnull(zip_code) and pd.notnull(city):
            zip_to_cities[zip_code].add(city)
    return zip_to_cities

def create_city_to_zips_mapping(df, city_col='City_UPDATED', zip_col='Zip_UPDATED') -> defaultdict:
    city_to_zips = defaultdict(set)
    for _, row in df.iterrows():
        zip_code = row[zip_col]
        city = row[city_col]
        if pd.notnull(zip_code) and pd.notnull(city):
            city_to_zips[city].add(zip_code)
    return city_to_zips

def get_mapping_ztc(zip_to_cities_pcrb: defaultdict, zip_to_cities_dnb: defaultdict) -> defaultdict:
    zip_to_cities = defaultdict(set)
    for zip_code in set(zip_to_cities_pcrb.keys()).union(zip_to_cities_dnb.keys()):
        zip_to_cities[zip_code] = zip_to_cities_pcrb.get(zip_code, set()).union(zip_to_cities_dnb.get(zip_code, set()))
    return zip_to_cities

def get_mapping_ctz(city_to_zips_pcrb: defaultdict, city_to_zips_dnb: defaultdict) -> defaultdict:
    city_to_zips = defaultdict(set)
    for city in set(city_to_zips_pcrb.keys()).union(city_to_zips_dnb.keys()):
        city_to_zips[city] = city_to_zips_pcrb.get(city, set()).union(city_to_zips_dnb.get(city, set()))
    return city_to_zips

def fuzzy_match(pcrb_dover, dnb_dover, street_threshold=90, business_name_threshold=90, global_name_threshold=90, method_name='Method 1'):
    """
    This function performs fuzzy matching between pcrb_dover and dnb_dover datasets.
    """
    # Start time tracking
    start_time = time.time()
    logging.info(f"PCRB Dataset Size Before Fuzzy Matching: {len(pcrb_dover)}")
    logging.info(f"DnB Dataset Size Before Fuzzy Matching: {len(dnb_dover)}")
    # Prepare data lists for fuzzy matching
    pcrb_street = pcrb_dover['Street_UPDATED'].tolist()
    pcrb_business = pcrb_dover['BusinessName_UPDATED'].tolist()
    dnb_street = dnb_dover['Street_UPDATED'].tolist()
    dnb_business = dnb_dover['BusinessName_UPDATED'].tolist()
    dnb_global = dnb_dover['GlobalName_UPDATED'].tolist()
    # Initialize dictionaries to keep track of best matches
    best_matches_i = {}  # Key: pcrb index, Value: (dnb index, match score, use_global, street_score, business_score)
    matched_records_2 = set()  # To avoid multiple matches to the same record in dnb_dover
    # Iterate over each record in pcrb_dover
    for i in range(len(pcrb_street)):
        street1 = pcrb_street[i]
        business1 = pcrb_business[i]
        best_match = None
        best_business_score = -1
        best_street_score = -1
        best_j = None
        for j in range(len(dnb_street)):
            if j in matched_records_2:
                continue
            street2 = dnb_street[j]
            business2 = dnb_business[j]
            global_name2 = dnb_global[j]
            street_score = fuzz.ratio(street1, street2)
            if street_score >= street_threshold:
                business_score = fuzz.ratio(business1, business2)
                if business_score >= business_name_threshold:
                    match_score = business_score
                    use_global = False
                else:
                    if pd.notnull(global_name2):
                        global_score = fuzz.ratio(business1, global_name2)
                        if global_score >= global_name_threshold:
                            match_score = global_score
                            use_global = True
                            business_score = global_score
                        else:
                            continue  # Business name doesn't meet threshold
                    else:
                        continue  # No global name available
                # Candidate meets all thresholds
                # Check if this candidate has a better business score
                if business_score > best_business_score:
                    best_business_score = business_score
                    best_street_score = street_score
                    best_match = (j, match_score, use_global, street_score, business_score)
                    best_j = j
                # If business scores are equal, check street score
                elif business_score == best_business_score:
                    if street_score > best_street_score:
                        best_street_score = street_score
                        best_match = (j, match_score, use_global, street_score, business_score)
                        best_j = j
        if best_match is not None:
            best_matches_i[i] = best_match
            matched_records_2.add(best_j)
    # Convert the best fuzzy matches to a DataFrame
    matched_pairs = [(i, val[0], val[1], val[2], val[3], val[4]) for i, val in best_matches_i.items()]
    matched_pairs_df = pd.DataFrame(matched_pairs, columns=['pcrb_index', 'dnb_index', 'Similarity_Score',
                                                            'Used_Global', 'Street_Similarity',
                                                            'BusinessName_Similarity'])
    # Retrieve actual data for the matched records using .iloc
    matched_pcrb_dover = pcrb_dover.iloc[matched_pairs_df['pcrb_index']].copy()
    matched_dnb_dover = dnb_dover.iloc[matched_pairs_df['dnb_index']].copy()
    # Reset indices for matched data (not the main DataFrames)
    matched_pcrb_dover = matched_pcrb_dover.reset_index(drop=True)
    matched_dnb_dover = matched_dnb_dover.reset_index(drop=True)
    matched_pairs_df = matched_pairs_df.reset_index(drop=True)
    matched_pcrb_dover = matched_pcrb_dover.add_prefix('pcrb_')
    matched_dnb_dover = matched_dnb_dover.add_prefix('dnb_')
    # Concatenate matched data
    matched_df = pd.concat([matched_pcrb_dover, matched_dnb_dover], axis=1)
    # Add matching details
    matched_df['Similarity_Score'] = matched_pairs_df['Similarity_Score']
    matched_df['Used_Global'] = matched_pairs_df['Used_Global']
    matched_df['Street_Similarity'] = matched_pairs_df['Street_Similarity']
    matched_df['BusinessName_Similarity'] = matched_pairs_df['BusinessName_Similarity']
    # Add 'matched_method' column
    matched_df['matched_method'] = method_name
    matched_df['matched'] = True
    # Records in PCRB dataset that did not get a fuzzy match
    matched_indices_pcrb = np.array(list(best_matches_i.keys()))
    unmatched_indices_pcrb = np.setdiff1d(np.arange(len(pcrb_dover)), matched_indices_pcrb)
    unmatched_pcrb_dover = pcrb_dover.iloc[unmatched_indices_pcrb].copy()
    unmatched_pcrb_dover['matched'] = False
    # Records in DnB dataset that did not get a fuzzy match
    matched_indices_dnb = np.array(list(matched_records_2))
    unmatched_indices_dnb = np.setdiff1d(np.arange(len(dnb_dover)), matched_indices_dnb)
    unmatched_dnb_dover = dnb_dover.iloc[unmatched_indices_dnb].copy()
    unmatched_dnb_dover['matched'] = False
    # logging.info matched records and percentage
    num_matched_records = len(matched_df)
    percentage_matched = num_matched_records / len(pcrb_dover) * 100 if len(pcrb_dover) > 0 else 0
    logging.info(f"Matched Records: {num_matched_records}")
    logging.info(f"Percentage of records matched: {percentage_matched:.2f}%")
    # logging.info adjusted dataset sizes
    logging.info(f"PCRB Dataset Size After Fuzzy Matching: {len(unmatched_pcrb_dover)}")
    logging.info(f"DnB Dataset Size After Fuzzy Matching: {len(unmatched_dnb_dover)}")
    # Compute total time
    end_time = time.time()
    total_time = end_time - start_time
    logging.info(f"{method_name} completed in {total_time:.2f} seconds.")
    # Return matched and unmatched datasets along with unmatched indices
    return matched_df, unmatched_pcrb_dover, unmatched_dnb_dover, unmatched_indices_pcrb, unmatched_indices_dnb


def compute_embeddings(pcrb_dover, dnb_dover):
    """
    This function computes the embeddings for the PCRB and DnB datasets.
    """
    # Start time tracking
    start_time = time.time()

    # Load the model
    model = SentenceTransformer('paraphrase-MiniLM-L6-v2')

    # Generate embeddings for PCRB dataset
    embeddings_1_street = model.encode(pcrb_dover['Street_UPDATED'].tolist(), convert_to_tensor=True)
    embeddings_1_business = model.encode(pcrb_dover['BusinessName_UPDATED'].tolist(), convert_to_tensor=True)

    # Convert embeddings to numpy arrays
    embeddings_1_street = embeddings_1_street.cpu().numpy()
    embeddings_1_business = embeddings_1_business.cpu().numpy()

    # Generate embeddings for DnB dataset
    embeddings_2_street = model.encode(dnb_dover['Street_UPDATED'].tolist(), convert_to_tensor=True)
    embeddings_2_business = model.encode(dnb_dover['BusinessName_UPDATED'].tolist(), convert_to_tensor=True)

    # Convert embeddings to numpy arrays
    embeddings_2_street = embeddings_2_street.cpu().numpy()
    embeddings_2_business = embeddings_2_business.cpu().numpy()

    # Prepare global name embeddings, handling nulls
    embeddings_2_global = []
    for name in dnb_dover['GlobalName_UPDATED']:
        if pd.notnull(name):
            embedding = model.encode(name, convert_to_tensor=True)
            embeddings_2_global.append(embedding.cpu().numpy())  # Convert to numpy array
        else:
            embeddings_2_global.append(None)

    # Compute total time
    end_time = time.time()
    total_time = end_time - start_time
    logging.info(f"Embeddings computed in {total_time:.2f} seconds.")
    
    # Return the embeddings
    return (embeddings_1_street, embeddings_1_business,
            embeddings_2_street, embeddings_2_business, embeddings_2_global)

def embedding_match(pcrb_dover, dnb_dover, embeddings_1_street, embeddings_1_business,
                   embeddings_2_street, embeddings_2_business, embeddings_2_global,
                   street_threshold=0.90, business_name_threshold=0.90, global_name_threshold=0.90,
                   method_name='Method 3'):
    """
    This function performs embedding-based matching between pcrb_dover and dnb_dover datasets using precomputed embeddings.
    """
    # Start time tracking
    start_time = time.time()
    # Compute cosine similarity matrices
    cosine_scores_street = util.cos_sim(embeddings_1_street, embeddings_2_street).numpy()
    cosine_scores_business = util.cos_sim(embeddings_1_business, embeddings_2_business).numpy()
    # Initialize dictionaries to keep track of best matches
    best_matches_i = {}  # Key: PCRB index, Value: (DnB index, match score, use_global, street_score, business_score)
    matched_records_2 = set()  # To avoid multiple matches to the same record in DnB
    # Iterate over each record in PCRB dataset
    for i in range(len(embeddings_1_street)):
        best_match = None
        best_business_score = -1
        best_street_score = -1
        best_j = None
        for j in range(len(embeddings_2_street)):
            if j in matched_records_2:
                continue  # Skip if already matched
            street_score = cosine_scores_street[i, j]
            if street_score >= street_threshold:
                business_score = cosine_scores_business[i, j]
                if business_score >= business_name_threshold:
                    match_score = business_score
                    use_global = False
                else:
                    # Check global name
                    if embeddings_2_global[j] is not None:
                        global_similarity = util.cos_sim(embeddings_1_business[i], embeddings_2_global[j]).item()
                        if global_similarity >= global_name_threshold:
                            match_score = global_similarity
                            use_global = True
                            business_score = global_similarity  # Use global similarity as business_score
                        else:
                            continue  # No match on global name
                    else:
                        continue  # No global name available
                # Candidate meets all thresholds
                # Check if this candidate has a better business score
                if business_score > best_business_score:
                    best_business_score = business_score
                    best_street_score = street_score
                    best_match = (j, match_score, use_global, street_score, business_score)
                    best_j = j
                # If business scores are equal, check street score
                elif business_score == best_business_score:
                    if street_score > best_street_score:
                        best_street_score = street_score
                        best_match = (j, match_score, use_global, street_score, business_score)
                        best_j = j
        # Record the best match found for this record
        if best_match is not None:
            best_matches_i[i] = best_match
            matched_records_2.add(best_j)
    # Convert the best matches to a DataFrame
    matched_pairs = [(i, val[0], val[1], val[2], val[3], val[4]) for i, val in best_matches_i.items()]
    matched_pairs_df = pd.DataFrame(matched_pairs, columns=['pcrb_index', 'dnb_index', 'Similarity_Score',
                                                            'Used_Global', 'Street_Similarity',
                                                            'BusinessName_Similarity'])
    # Retrieve actual data for the matched records using original indices
    matched_pcrb_dover = pcrb_dover.iloc[matched_pairs_df['pcrb_index']].copy()
    matched_dnb_dover = dnb_dover.iloc[matched_pairs_df['dnb_index']].copy()
    # Reset indices for matched data (not for the main DataFrames)
    matched_pcrb_dover = matched_pcrb_dover.reset_index(drop=True)
    matched_dnb_dover = matched_dnb_dover.reset_index(drop=True)
    matched_pairs_df = matched_pairs_df.reset_index(drop=True)
    # Add embeddings to matched dataframes
    matched_pcrb_dover['Embedding_Street'] = list(embeddings_1_street[matched_pairs_df['pcrb_index']])
    matched_pcrb_dover['Embedding_BusinessName'] = list(embeddings_1_business[matched_pairs_df['pcrb_index']])
    matched_dnb_dover['Embedding_Street'] = list(embeddings_2_street[matched_pairs_df['dnb_index']])
    matched_dnb_dover['Embedding_BusinessName'] = list(embeddings_2_business[matched_pairs_df['dnb_index']])
    matched_dnb_dover['Embedding_GlobalName'] = [embeddings_2_global[idx] for idx in matched_pairs_df['dnb_index']]
    # Add prefixes to column names
    matched_pcrb_dover = matched_pcrb_dover.add_prefix('pcrb_')
    matched_dnb_dover = matched_dnb_dover.add_prefix('dnb_')
    # Concatenate matched data
    matched_df = pd.concat([matched_pcrb_dover, matched_dnb_dover], axis=1)
    # Add matching details
    matched_df['Similarity_Score'] = matched_pairs_df['Similarity_Score']
    matched_df['Used_Global'] = matched_pairs_df['Used_Global']
    matched_df['Street_Similarity'] = matched_pairs_df['Street_Similarity']
    matched_df['BusinessName_Similarity'] = matched_pairs_df['BusinessName_Similarity']
    # Add 'matched_method' column with the provided method_name
    matched_df['matched_method'] = method_name
    matched_df['matched'] = True
    # Records in PCRB dataset that did not get a match
    matched_indices_pcrb = np.array(list(best_matches_i.keys()))
    unmatched_indices_pcrb = np.setdiff1d(np.arange(len(pcrb_dover)), matched_indices_pcrb)
    unmatched_pcrb_dover = pcrb_dover.iloc[unmatched_indices_pcrb].copy()
    unmatched_pcrb_dover['matched'] = False
    # Add embeddings to unmatched PCRB records
    unmatched_pcrb_dover['Embedding_Street'] = list(embeddings_1_street[unmatched_indices_pcrb])
    unmatched_pcrb_dover['Embedding_BusinessName'] = list(embeddings_1_business[unmatched_indices_pcrb])
    # Records in DnB dataset that did not get a match
    matched_indices_dnb = np.array(list(matched_records_2))
    unmatched_indices_dnb = np.setdiff1d(np.arange(len(dnb_dover)), matched_indices_dnb)
    unmatched_dnb_dover = dnb_dover.iloc[unmatched_indices_dnb].copy()
    unmatched_dnb_dover['matched'] = False
    # Add embeddings to unmatched DnB records
    unmatched_dnb_dover['Embedding_Street'] = list(embeddings_2_street[unmatched_indices_dnb])
    unmatched_dnb_dover['Embedding_BusinessName'] = list(embeddings_2_business[unmatched_indices_dnb])
    unmatched_dnb_dover['Embedding_GlobalName'] = [embeddings_2_global[idx] for idx in unmatched_indices_dnb]
    # Compute total time
    end_time = time.time()
    total_time = end_time - start_time
    # Return matched and unmatched datasets along with indices
    return matched_df, unmatched_pcrb_dover, unmatched_dnb_dover, unmatched_indices_pcrb, unmatched_indices_dnb

def batch_processing(zip_counts, batch_size: int, col_append: str) -> list:
    # Sort ZIP codes by total comparisons
    zip_counts_sorted = zip_counts.sort_values(by='total_comparisons', ascending=True)
    # Filter to keep only the batches we need
    batches = []
    cumulative_sum = 0
    current_batch = []
    for _, row in zip_counts_sorted.iterrows():
        current_batch.append(row[col_append])
        cumulative_sum += row['total_comparisons']
        if cumulative_sum >= batch_size:
            batches.append(current_batch)
            current_batch = []
            cumulative_sum = 0
    # Catch any remaining ZIPs that didn't make the full batch size
    if current_batch:
        batches.append(current_batch)
    return batches

# Assuming df_pcrb and df_dnb are your two datasets already loaded
def calculate_zip_counts(df_pcrb: pd.DataFrame, df_dnb: pd.DataFrame, col_groupby: str, phase1: bool):
    # Calculate count per ZIP in each DataFrame
    zip_count_pcrb = df_pcrb.groupby(col_groupby).size().reset_index(name='count_pcrb')
    zip_count_dnb = df_dnb.groupby(col_groupby).size().reset_index(name='count_dnb')
    # Merge the counts on Zip Code
    zip_counts = pd.merge(zip_count_pcrb, zip_count_dnb, on=col_groupby, how='outer').fillna(0)
    zip_counts['count_pcrb'] = zip_counts['count_pcrb'].astype(int)
    zip_counts['count_dnb'] = zip_counts['count_dnb'].astype(int)
    # Calculate total comparisons for each ZIP code
    zip_counts['total_comparisons'] = zip_counts['count_pcrb'] * zip_counts['count_dnb']
    if not phase1:
        zip_counts = zip_counts[zip_counts['total_comparisons'] > 0]
    return zip_counts
    


def process_zip_codes_phase1(batch_zip_codes, df_pcrb, df_dnb, zip_to_cities, fuzzy_match, ):
    matched_dfs = []
    city_stats = []
    unmatched_pcrb_records = []
    unmatched_dnb_records = []
    for idx, zip_code in enumerate(batch_zip_codes):
        logging.info(f"Processing zip code {zip_code} ({idx+1}/{len(batch_zip_codes)})")
        
        # Start timing for this zip_code
        zip_start_time = time.time()

        # Initialize a dictionary to hold stats for this zip_code
        zip_stat = {'Zipcode': zip_code, 'Total_PCRB_Records': 0, 'Total_DNB_Records': 0, 'Total_Matched_Records': 0}

        # Filter datasets for the current zip code
        associated_cities = zip_to_cities.get(zip_code, set())
        pcrb_batch_df = df_pcrb[
            (df_pcrb['Zip_UPDATED'] == zip_code) & (df_pcrb['City_UPDATED'].isin(associated_cities))
        ]
        dnb_batch_df = df_dnb[
            (df_dnb['Zip_UPDATED'] == zip_code) & (df_dnb['City_UPDATED'].isin(associated_cities))
        ]
        # Get total number of PCRB and DNB records for the zip_code
        total_pcrb_records_zip = len(pcrb_batch_df)
        total_dnb_records_zip = len(dnb_batch_df)
        zip_stat['Total_PCRB_Records'] = total_pcrb_records_zip
        zip_stat['Total_DNB_Records'] = total_dnb_records_zip
        if total_pcrb_records_zip == 0 or total_dnb_records_zip == 0:
            logging.info("No records to match in this batch.")
            continue
        total_matched_records_zip = 0  # Initialize matched counter for the zip code

        ############################################### Method 1 #####################################################
        method_start_time = time.time()
        matched_df_method1, unmatched_pcrb_batch_df, unmatched_dnb_batch_df, _, _ = fuzzy_match(
            pcrb_batch_df, dnb_batch_df,
            street_threshold=95,
            business_name_threshold=80,
            global_name_threshold=80,
            method_name='Method 1'
        )
        method_end_time = time.time()
        method_elapsed_time = method_end_time - method_start_time
        num_matched_method1 = len(matched_df_method1)
        total_matched_records_zip += num_matched_method1
        zip_stat['Method1_Matched_Records'] = num_matched_method1
        zip_stat['Method1_Time'] = method_elapsed_time
        matched_dfs.append(matched_df_method1)
        logging.info(f"Method 1 matched records: {num_matched_method1}, Time taken: {method_elapsed_time:.2f} seconds")

        # Update datasets for unmatched records
        pcrb_batch_df = unmatched_pcrb_batch_df
        dnb_batch_df = unmatched_dnb_batch_df
        if len(pcrb_batch_df) == 0 or len(dnb_batch_df) == 0:
            logging.info("No unmatched records left after Method 1 in this batch.")
            if not pcrb_batch_df.empty:
                unmatched_pcrb_records.append(pcrb_batch_df)
            if not dnb_batch_df.empty:
                unmatched_dnb_records.append(dnb_batch_df)
            if total_pcrb_records_zip > 0:
                match_percentage_zip = (total_matched_records_zip / total_pcrb_records_zip) * 100
            else:
                match_percentage_zip = 0
            zip_stat['Total_Matched_Records'] = total_matched_records_zip
            zip_stat['Match_Percentage'] = match_percentage_zip
            # End timing for this zip_code
            zip_end_time = time.time()
            zip_elapsed_time = zip_end_time - zip_start_time
            zip_stat['Total_Time'] = zip_elapsed_time
            city_stats.append(zip_stat)
            continue
        
        ############################################### Method 2 #####################################################
        method_start_time = time.time()
        matched_df_method2, unmatched_pcrb_batch_df, unmatched_dnb_batch_df, _, _ = fuzzy_match(
            pcrb_batch_df, dnb_batch_df,
            street_threshold=80,
            business_name_threshold=95,
            global_name_threshold=90,
            method_name='Method 2'
        )
        method_end_time = time.time()
        method_elapsed_time = method_end_time - method_start_time
        num_matched_method2 = len(matched_df_method2)
        total_matched_records_zip += num_matched_method2
        zip_stat['Method2_Matched_Records'] = num_matched_method2
        zip_stat['Method2_Time'] = method_elapsed_time
        matched_dfs.append(matched_df_method2)
        logging.info(f"Method 2 matched records: {num_matched_method2}, Time taken: {method_elapsed_time:.2f} seconds")
        
        # Update datasets for unmatched records
        pcrb_batch_df = unmatched_pcrb_batch_df
        dnb_batch_df = unmatched_dnb_batch_df
        if len(pcrb_batch_df) == 0 or len(dnb_batch_df) == 0:
            logging.info("No unmatched records left after Method 2 in this batch.")
            if not pcrb_batch_df.empty:
                unmatched_pcrb_records.append(pcrb_batch_df)
            if not dnb_batch_df.empty:
                unmatched_dnb_records.append(dnb_batch_df)
            if total_pcrb_records_zip > 0:
                match_percentage_zip = (total_matched_records_zip / total_pcrb_records_zip) * 100
            else:
                match_percentage_zip = 0
            zip_stat['Total_Matched_Records'] = total_matched_records_zip
            zip_stat['Match_Percentage'] = match_percentage_zip
            # End timing for this zip_code
            zip_end_time = time.time()
            zip_elapsed_time = zip_end_time - zip_start_time
            zip_stat['Total_Time'] = zip_elapsed_time
            city_stats.append(zip_stat)
            continue
        # Collect unmatched records
        if not pcrb_batch_df.empty:
            unmatched_pcrb_records.append(pcrb_batch_df)
        if not dnb_batch_df.empty:
            unmatched_dnb_records.append(dnb_batch_df)
        if total_pcrb_records_zip > 0:
            match_percentage_zip = (total_matched_records_zip / total_pcrb_records_zip) * 100
        else:
            match_percentage_zip = 0

        zip_stat['Total_Matched_Records'] = total_matched_records_zip
        zip_stat['Match_Percentage'] = match_percentage_zip

        # End timing for this zip_code
        zip_end_time = time.time()
        zip_elapsed_time = zip_end_time - zip_start_time
        zip_stat['Total_Time'] = zip_elapsed_time
        city_stats.append(zip_stat)

    ################################### Combine all matched DataFrames ################################
    final_matched_df = pd.concat(matched_dfs, ignore_index=True)
    # Combine all unmatched PCRB records
    if unmatched_pcrb_records:
        final_unmatched_pcrb_df = pd.concat(unmatched_pcrb_records, ignore_index=True).drop_duplicates(subset='record_id')
    else:
        final_unmatched_pcrb_df = pd.DataFrame()
        
    # Combine all unmatched DnB records
    if unmatched_dnb_records:
        final_unmatched_dnb_df = pd.concat(unmatched_dnb_records, ignore_index=True).drop_duplicates(subset='record_id')
    else:
        final_unmatched_dnb_df = pd.DataFrame()
    # Explicitly call garbage collector
    gc.collect()
    return final_matched_df, city_stats, final_unmatched_pcrb_df, final_unmatched_dnb_df
    # Define the fuzzy_match function, df_pcrb, df_dnb, and zip_to_cities as required before using this function.

def update_dfs(df_pcrb: pd.DataFrame, df_dnb: pd.DataFrame, final_matched_df: pd.DataFrame) -> tuple[pd.DataFrame]:
    # Step 1: logging.info Orginal Shape
    logging.info(f"PCRB Original Shape: {df_pcrb.shape}")
    logging.info(f"DnB Original Shape: {df_dnb.shape}")
    # Step 2: Create a set of matched DnB record IDs
    matched_pcrb_ids = set(final_matched_df['pcrb_record_id'])
    matched_dnb_ids = set(final_matched_df['dnb_record_id'])
    logging.info(f"PCRB Matched Records: {len(matched_pcrb_ids)}")
    logging.info(f"DnB Matched Records: {len(matched_dnb_ids)}")
    # Step 3: Update df_dnb to exclude already matched records
    df_pcrb = df_pcrb[~df_pcrb['record_id'].isin(matched_pcrb_ids)]
    df_dnb = df_dnb[~df_dnb['record_id'].isin(matched_dnb_ids)]
    logging.info(f"PCRB Adjusted Shape: {df_pcrb.shape}")
    logging.info(f"DnB Adjusted Shape: {df_dnb.shape}")
    return (df_pcrb, df_dnb)

@cache_to_csv("final_matched_df1.csv")
def get_result_dfs_phase1(batches, df_pcrb, df_dnb, zip_to_cities) -> tuple[pd.DataFrame]:
    # Loop over each batch and process
    all_matched_dfs = []
    all_city_stats = []
    all_unmatched_pcrb = []
    all_unmatched_dnb = []

    for i, batch in enumerate(batches):
        logging.info(f"Processing Batch {i+1}/{len(batches)}")
        matched_df, city_stats, unmatched_pcrb_df, unmatched_dnb_df = process_zip_codes_phase1(
            batch,
            df_pcrb,
            df_dnb,
            zip_to_cities,
            fuzzy_match  # Make sure this function is defined and available
        )
        # Store the results from each batch
        all_matched_dfs.append(matched_df)
        all_city_stats.extend(city_stats)
        all_unmatched_pcrb.append(unmatched_pcrb_df)
        all_unmatched_dnb.append(unmatched_dnb_df)
    # Optional: Save intermediate results to files or a database for recovery and analysis
    # Example: matched_df.to_csv(f"batch_{i+1}_matches.csv")
    # Combine all results after all batches are processed
    final_matched_df = pd.concat(all_matched_dfs, ignore_index=True)
    final_city_stats_df = pd.DataFrame(all_city_stats)
    final_unmatched_pcrb_df = pd.concat(all_unmatched_pcrb, ignore_index=True)
    final_unmatched_dnb_df = pd.concat(all_unmatched_dnb, ignore_index=True)
    # final_city_stats_df.to_csv("Phase_1_Zip_Stats.csv")
    with pd.ExcelWriter('Dashboard.xlsx', mode='a', if_sheet_exists='replace') as writer:
        final_city_stats_df.to_excel(writer, sheet_name='Phase1')
        logging.info("final_city_stats_df has been saved to 'Phase 1' sheet in Dashboard.xlsx")
    #return (final_matched_df, final_city_stats_df, final_unmatched_pcrb_df, final_unmatched_dnb_df)
    return final_matched_df



# Phase 2
def process_zip_codes_phase2(batch_zip_codes, df_pcrb, df_dnb, city_to_zips, fuzzy_match):
    matched_dfs = []
    city_stats = []
    unmatched_pcrb_records = []
    unmatched_dnb_records = []
    for idx, city in enumerate(batch_zip_codes):
        logging.info(f"Processing zip code {city} ({idx+1}/{len(batch_zip_codes)})")
        
        # Start timing for this zip_code
        zip_start_time = time.time()

        # Initialize a dictionary to hold stats for this zip_code
        zip_stat = {'City': city, 'Total_PCRB_Records': 0, 'Total_DNB_Records': 0, 'Total_Matched_Records': 0}

        # Filter datasets for the current zip code
        associated_cities = city_to_zips.get(city, set())
        pcrb_batch_df = df_pcrb[
            (df_pcrb['City_UPDATED'] == city) & (df_pcrb['Zip_UPDATED'].isin(associated_cities))
        ]
        dnb_batch_df = df_dnb[
            (df_dnb['City_UPDATED'] == city) & (df_dnb['Zip_UPDATED'].isin(associated_cities))
        ]
        # Get total number of PCRB and DNB records for the zip_code
        total_pcrb_records_zip = len(pcrb_batch_df)
        total_dnb_records_zip = len(dnb_batch_df)
        zip_stat['Total_PCRB_Records'] = total_pcrb_records_zip
        zip_stat['Total_DNB_Records'] = total_dnb_records_zip
        if total_pcrb_records_zip == 0 or total_dnb_records_zip == 0:
            logging.info("No records to match in this batch.")
            continue
        total_matched_records_zip = 0  # Initialize matched counter for the zip code

        ############################################### Method 1 #####################################################
        method_start_time = time.time()
        matched_df_method1, unmatched_pcrb_batch_df, unmatched_dnb_batch_df, _, _ = fuzzy_match(
            pcrb_batch_df, dnb_batch_df,
            street_threshold=95,
            business_name_threshold=80,
            global_name_threshold=80,
            method_name='Method 1'
        )
        method_end_time = time.time()
        method_elapsed_time = method_end_time - method_start_time
        num_matched_method1 = len(matched_df_method1)
        total_matched_records_zip += num_matched_method1
        zip_stat['Method1_Matched_Records'] = num_matched_method1
        zip_stat['Method1_Time'] = method_elapsed_time
        matched_dfs.append(matched_df_method1)
        logging.info(f"Method 1 matched records: {num_matched_method1}, Time taken: {method_elapsed_time:.2f} seconds")

        # Update datasets for unmatched records
        pcrb_batch_df = unmatched_pcrb_batch_df
        dnb_batch_df = unmatched_dnb_batch_df
        if len(pcrb_batch_df) == 0 or len(dnb_batch_df) == 0:
            logging.info("No unmatched records left after Method 1 in this batch.")
            if not pcrb_batch_df.empty:
                unmatched_pcrb_records.append(pcrb_batch_df)
            if not dnb_batch_df.empty:
                unmatched_dnb_records.append(dnb_batch_df)
            if total_pcrb_records_zip > 0:
                match_percentage_zip = (total_matched_records_zip / total_pcrb_records_zip) * 100
            else:
                match_percentage_zip = 0
            zip_stat['Total_Matched_Records'] = total_matched_records_zip
            zip_stat['Match_Percentage'] = match_percentage_zip
            # End timing for this zip_code
            zip_end_time = time.time()
            zip_elapsed_time = zip_end_time - zip_start_time
            zip_stat['Total_Time'] = zip_elapsed_time
            city_stats.append(zip_stat)
            continue
        
        ############################################### Method 2 #####################################################
        method_start_time = time.time()
        matched_df_method2, unmatched_pcrb_batch_df, unmatched_dnb_batch_df, _, _ = fuzzy_match(
            pcrb_batch_df, dnb_batch_df,
            street_threshold=80,
            business_name_threshold=95,
            global_name_threshold=90,
            method_name='Method 2'
        )
        method_end_time = time.time()
        method_elapsed_time = method_end_time - method_start_time
        num_matched_method2 = len(matched_df_method2)
        total_matched_records_zip += num_matched_method2
        zip_stat['Method2_Matched_Records'] = num_matched_method2
        zip_stat['Method2_Time'] = method_elapsed_time
        matched_dfs.append(matched_df_method2)
        logging.info(f"Method 2 matched records: {num_matched_method2}, Time taken: {method_elapsed_time:.2f} seconds")
        
        # Update datasets for unmatched records
        pcrb_batch_df = unmatched_pcrb_batch_df
        dnb_batch_df = unmatched_dnb_batch_df
        if len(pcrb_batch_df) == 0 or len(dnb_batch_df) == 0:
            logging.info("No unmatched records left after Method 2 in this batch.")
            if not pcrb_batch_df.empty:
                unmatched_pcrb_records.append(pcrb_batch_df)
            if not dnb_batch_df.empty:
                unmatched_dnb_records.append(dnb_batch_df)
            if total_pcrb_records_zip > 0:
                match_percentage_zip = (total_matched_records_zip / total_pcrb_records_zip) * 100
            else:
                match_percentage_zip = 0
            zip_stat['Total_Matched_Records'] = total_matched_records_zip
            zip_stat['Match_Percentage'] = match_percentage_zip
            # End timing for this zip_code
            zip_end_time = time.time()
            zip_elapsed_time = zip_end_time - zip_start_time
            zip_stat['Total_Time'] = zip_elapsed_time
            city_stats.append(zip_stat)
            continue
        # Collect unmatched records
        if not pcrb_batch_df.empty:
            unmatched_pcrb_records.append(pcrb_batch_df)
        if not dnb_batch_df.empty:
            unmatched_dnb_records.append(dnb_batch_df)
        if total_pcrb_records_zip > 0:
            match_percentage_zip = (total_matched_records_zip / total_pcrb_records_zip) * 100
        else:
            match_percentage_zip = 0

        zip_stat['Total_Matched_Records'] = total_matched_records_zip
        zip_stat['Match_Percentage'] = match_percentage_zip

        # End timing for this zip_code
        zip_end_time = time.time()
        zip_elapsed_time = zip_end_time - zip_start_time
        zip_stat['Total_Time'] = zip_elapsed_time
        city_stats.append(zip_stat)

    ################################### Combine all matched DataFrames ################################

    # Combine all Matched records
    if matched_dfs:
        final_matched_df = pd.concat(matched_dfs, ignore_index=True)
    else:
        final_matched_df = pd.DataFrame()

    # Combine all unmatched PCRB records
    if unmatched_pcrb_records:
        final_unmatched_pcrb_df = pd.concat(unmatched_pcrb_records, ignore_index=True).drop_duplicates(subset='record_id')
    else:
        final_unmatched_pcrb_df = pd.DataFrame()
        
    # Combine all unmatched DnB records
    if unmatched_dnb_records:
        final_unmatched_dnb_df = pd.concat(unmatched_dnb_records, ignore_index=True).drop_duplicates(subset='record_id')
    else:
        final_unmatched_dnb_df = pd.DataFrame()
    # Explicitly call garbage collector
    gc.collect()
    return final_matched_df, city_stats, final_unmatched_pcrb_df, final_unmatched_dnb_df

@cache_to_csv("final_matched_df2.csv")
def get_result_dfs_phase2(batches, df_pcrb, df_dnb, city_to_zips) -> tuple[pd.DataFrame]:
    # Loop over each batch and process
    all_matched_dfs = []
    all_city_stats = []
    all_unmatched_pcrb = []
    all_unmatched_dnb = []

    for i, batch in enumerate(batches):
        logging.info(f"Processing Batch {i+1}/{len(batches)}")
        matched_df, city_stats, unmatched_pcrb_df, unmatched_dnb_df = process_zip_codes_phase2(
            batch,
            df_pcrb,
            df_dnb,
            city_to_zips,
            fuzzy_match  # Make sure this function is defined and available
        )
        # Store the results from each batch
        all_matched_dfs.append(matched_df)
        all_city_stats.extend(city_stats)
        all_unmatched_pcrb.append(unmatched_pcrb_df)
        all_unmatched_dnb.append(unmatched_dnb_df)
        # Optional: Save intermediate results to files or a database for recovery and analysis
        #matched_df.to_csv(f"batch_{i+1}_matches.csv")
    # Combine all results after all batches are processed
    final_matched_df = pd.concat(all_matched_dfs, ignore_index=True)
    final_city_stats_df = pd.DataFrame(all_city_stats)
    final_unmatched_pcrb_df = pd.concat(all_unmatched_pcrb, ignore_index=True)
    final_unmatched_dnb_df = pd.concat(all_unmatched_dnb, ignore_index=True)
    # final_city_stats_df.to_csv("Phase_2_City_Stats.csv")
    with pd.ExcelWriter('Dashboard.xlsx', mode='a', if_sheet_exists='replace') as writer:
        final_city_stats_df.to_excel(writer, sheet_name='Phase 2')
        logging.info("final_city_stats_df has been saved to 'Phase 2' sheet in Dashboard.xlsx")
    #return (final_matched_df, final_city_stats_df, final_unmatched_pcrb_df, final_unmatched_dnb_df)
    return final_matched_df

# Phase 3
def process_zip_codes_phase3(batch_zip_codes, df_pcrb, df_dnb, zip_to_cities, fuzzy_match, embedding_match):
    matched_dfs = []
    city_stats = []
    unmatched_pcrb_records = []
    unmatched_dnb_records = []
    for idx, zip_code in enumerate(batch_zip_codes):
        logging.info(f"Processing zip code {zip_code} ({idx+1}/{len(batch_zip_codes)})")
        # Start timing for this zip_code
        zip_start_time = time.time()
        # Initialize a dictionary to hold stats for this zip_code
        zip_stat = {'Zipcode': zip_code, 'Total_PCRB_Records': 0, 'Total_DNB_Records': 0, 'Total_Matched_Records': 0}
        # Filter datasets for the current zip code
        associated_cities = zip_to_cities.get(zip_code, set())
        pcrb_batch_df = df_pcrb[
            (df_pcrb['Zip_UPDATED'] == zip_code) & (df_pcrb['City_UPDATED'].isin(associated_cities))
        ]
        dnb_batch_df = df_dnb[
            (df_dnb['Zip_UPDATED'] == zip_code) & (df_dnb['City_UPDATED'].isin(associated_cities))
        ]
        # Get total number of PCRB and DNB records for the zip_code
        total_pcrb_records_zip = len(pcrb_batch_df)
        total_dnb_records_zip = len(dnb_batch_df)
        zip_stat['Total_PCRB_Records'] = total_pcrb_records_zip
        zip_stat['Total_DNB_Records'] = total_dnb_records_zip
        if total_pcrb_records_zip == 0 or total_dnb_records_zip == 0:
            logging.info("No records to match in this batch.")
            continue
        total_matched_records_zip = 0  # Initialize matched counter for the zip code
        
        # Compute embeddings once for the batch
        warnings.filterwarnings("ignore", category=FutureWarning, module="transformers.tokenization_utils_base")
        embedding_start_time = time.time()
        (embeddings_1_street_full, embeddings_1_business_full,
            embeddings_2_street_full, embeddings_2_business_full,
            embeddings_2_global_full) = compute_embeddings(pcrb_batch_df, dnb_batch_df)
        embedding_end_time = time.time()
        embedding_elapsed_time = embedding_end_time - embedding_start_time
        zip_stat['Embedding_Compute_Time'] = embedding_elapsed_time
        # Initialize variables
        pcrb_current_df = pcrb_batch_df.copy()
        dnb_current_df = dnb_batch_df.copy()
        embeddings_1_street = embeddings_1_street_full
        embeddings_1_business = embeddings_1_business_full
        embeddings_2_street = embeddings_2_street_full
        embeddings_2_business = embeddings_2_business_full
        embeddings_2_global = embeddings_2_global_full
        ### Method 3: Embedding Match
        method_start_time = time.time()
        matched_df_method3, unmatched_pcrb_df, unmatched_dnb_df, unmatched_indices_pcrb, unmatched_indices_dnb = embedding_match(
            pcrb_current_df, dnb_current_df,
            embeddings_1_street, embeddings_1_business,
            embeddings_2_street, embeddings_2_business, embeddings_2_global,
            street_threshold=0.90,
            business_name_threshold=0.80,
            global_name_threshold=0.90,
            method_name='Method 3'
        )
        method_end_time = time.time()
        method_elapsed_time = method_end_time - method_start_time
        zip_stat['Method3_Time'] = method_elapsed_time
        num_matched_method3 = len(matched_df_method3)
        zip_stat['Method3_Matched_Records'] = num_matched_method3
        total_matched_records_zip += num_matched_method3
        matched_dfs.append(matched_df_method3)
        # Update embeddings and DataFrames for unmatched records
        pcrb_current_df = unmatched_pcrb_df
        dnb_current_df = unmatched_dnb_df
        embeddings_1_street = embeddings_1_street[unmatched_indices_pcrb]
        embeddings_1_business = embeddings_1_business[unmatched_indices_pcrb]
        embeddings_2_street = embeddings_2_street[unmatched_indices_dnb]
        embeddings_2_business = embeddings_2_business[unmatched_indices_dnb]
        embeddings_2_global = [embeddings_2_global[i] for i in unmatched_indices_dnb]
        if pcrb_current_df.empty or dnb_current_df.empty:
            logging.info("No unmatched records left after Method 3 in this batch.")
            if not pcrb_current_df.empty:
                unmatched_pcrb_records.append(pcrb_current_df)
            if not dnb_current_df.empty:
                unmatched_dnb_records.append(dnb_current_df)
            match_percentage_zip = (total_matched_records_zip / total_pcrb_records_zip) * 100
            zip_stat['Total_Matched_Records'] = total_matched_records_zip
            zip_stat['Match_Percentage'] = match_percentage_zip
            # End timing for this zip_code
            zip_end_time = time.time()
            zip_elapsed_time = zip_end_time - zip_start_time
            zip_stat['Total_Time'] = zip_elapsed_time
            city_stats.append(zip_stat)
            continue
        ### Method 4: Embedding Match with different thresholds
        method_start_time = time.time()
        matched_df_method4, unmatched_pcrb_df, unmatched_dnb_df, unmatched_indices_pcrb, unmatched_indices_dnb = embedding_match(
            pcrb_current_df, dnb_current_df,
            embeddings_1_street, embeddings_1_business,
            embeddings_2_street, embeddings_2_business, embeddings_2_global,
            street_threshold=0.60,
            business_name_threshold=0.80,
            global_name_threshold=0.80,
            method_name='Method 4'
        )
        method_end_time = time.time()
        method_elapsed_time = method_end_time - method_start_time
        zip_stat['Method4_Time'] = method_elapsed_time
        num_matched_method4 = len(matched_df_method4)
        zip_stat['Method4_Matched_Records'] = num_matched_method4
        total_matched_records_zip += num_matched_method4
        matched_dfs.append(matched_df_method4)
        # Update embeddings and DataFrames for unmatched records
        pcrb_current_df = unmatched_pcrb_df
        dnb_current_df = unmatched_dnb_df
        embeddings_1_street = embeddings_1_street[unmatched_indices_pcrb]
        embeddings_1_business = embeddings_1_business[unmatched_indices_pcrb]
        embeddings_2_street = embeddings_2_street[unmatched_indices_dnb]
        embeddings_2_business = embeddings_2_business[unmatched_indices_dnb]
        embeddings_2_global = [embeddings_2_global[i] for i in unmatched_indices_dnb]
        if pcrb_current_df.empty or dnb_current_df.empty:
            logging.info("No unmatched records left after Method 4 in this batch.")
            if not pcrb_current_df.empty:
                unmatched_pcrb_records.append(pcrb_current_df)
            if not dnb_current_df.empty:
                unmatched_dnb_records.append(dnb_current_df)
            match_percentage_zip = (total_matched_records_zip / total_pcrb_records_zip) * 100
            zip_stat['Total_Matched_Records'] = total_matched_records_zip
            zip_stat['Match_Percentage'] = match_percentage_zip
            # End timing for this zip_code
            zip_end_time = time.time()
            zip_elapsed_time = zip_end_time - zip_start_time
            zip_stat['Total_Time'] = zip_elapsed_time
            city_stats.append(zip_stat)
            continue
        ### Method 5: Embedding Match with different thresholds
        method_start_time = time.time()
        matched_df_method5, unmatched_pcrb_df, unmatched_dnb_df, unmatched_indices_pcrb, unmatched_indices_dnb = embedding_match(
            pcrb_current_df, dnb_current_df,
            embeddings_1_street, embeddings_1_business,
            embeddings_2_street, embeddings_2_business, embeddings_2_global,
            street_threshold=0.98,
            business_name_threshold=0.50,
            global_name_threshold=0.80,
            method_name='Method 5'
        )
        method_end_time = time.time()
        method_elapsed_time = method_end_time - method_start_time
        zip_stat['Method5_Time'] = method_elapsed_time
        num_matched_method5 = len(matched_df_method5)
        zip_stat['Method5_Matched_Records'] = num_matched_method5
        total_matched_records_zip += num_matched_method5
        matched_dfs.append(matched_df_method5)
        # Collect unmatched records after Method 5
        if not pcrb_current_df.empty:
            unmatched_pcrb_records.append(pcrb_current_df)
        if not dnb_current_df.empty:
            unmatched_dnb_records.append(dnb_current_df)
        # Record total matched records and match percentage
        match_percentage_zip = (total_matched_records_zip / total_pcrb_records_zip) * 100
        zip_stat['Total_Matched_Records'] = total_matched_records_zip
        zip_stat['Match_Percentage'] = match_percentage_zip
        # End timing for this zip_code
        zip_end_time = time.time()
        zip_elapsed_time = zip_end_time - zip_start_time
        zip_stat['Total_Time'] = zip_elapsed_time
        # Append zip stats
        city_stats.append(zip_stat)
    # Combine all matched DataFrames
    if matched_dfs:
        final_matched_df = pd.concat(matched_dfs, ignore_index=True)
    else:
        final_matched_df = pd.DataFrame()
    # Combine all unmatched PCRB records
    if unmatched_pcrb_records:
        final_unmatched_pcrb_df = pd.concat(unmatched_pcrb_records, ignore_index=True).drop_duplicates(subset='record_id')
    else:
        final_unmatched_pcrb_df = pd.DataFrame()
    # Combine all unmatched DnB records
    if unmatched_dnb_records:
        final_unmatched_dnb_df = pd.concat(unmatched_dnb_records, ignore_index=True).drop_duplicates(subset='record_id')
    else:
        final_unmatched_dnb_df = pd.DataFrame()
    # Explicitly call garbage collector
    gc.collect()
    return final_matched_df, city_stats, final_unmatched_pcrb_df, final_unmatched_dnb_df

def get_result_dfs_phase3(batches, df_pcrb, df_dnb, zip_to_cities) -> tuple[pd.DataFrame]:
    # Loop over each batch and process
    all_matched_dfs = []
    all_city_stats = []
    all_unmatched_pcrb = []
    all_unmatched_dnb = []
    # Replace the null values of 'Street_UPDATED' to 'Missing' in df_dnb
    df_dnb['Street_UPDATED'] = df_dnb['Street_UPDATED'].fillna("Missing").astype(pd.StringDtype())
    df_dnb['BusinessName_UPDATED'] = df_dnb['BusinessName_UPDATED'].fillna("Missing").astype(pd.StringDtype())
    df_pcrb['Street_UPDATED'] = df_pcrb['Street_UPDATED'].fillna("Missing").astype(pd.StringDtype())
    df_pcrb['BusinessName_UPDATED'] = df_pcrb['BusinessName_UPDATED'].fillna("Missing").astype(pd.StringDtype())
    for i, batch in enumerate(batches):
        logging.info(f"Processing Batch {i+1}/{len(batches)}")
        matched_df, city_stats, unmatched_pcrb_df, unmatched_dnb_df = process_zip_codes_phase3(
            batch,
            df_pcrb,
            df_dnb,
            zip_to_cities,
            fuzzy_match,  # Make sure this function is defined and available
            embedding_match
        )
        # Store the results from each batch
        all_matched_dfs.append(matched_df)
        all_city_stats.extend(city_stats)
        all_unmatched_pcrb.append(unmatched_pcrb_df)
        all_unmatched_dnb.append(unmatched_dnb_df)
        # Optional: Save intermediate results to files or a database for recovery and analysis
        #matched_df.to_csv(f"batch_{i+1}_matches_Phase_3.csv")
    # Combine all results after all batches are processed
    final_matched_df = pd.concat(all_matched_dfs, ignore_index=True)
    final_matched_df.to_csv("final_matched_df3.csv")
    final_city_stats_df = pd.DataFrame(all_city_stats)
    final_unmatched_pcrb_df = pd.concat(all_unmatched_pcrb, ignore_index=True)
    final_unmatched_dnb_df = pd.concat(all_unmatched_dnb, ignore_index=True)
    return (final_matched_df, final_city_stats_df, final_unmatched_pcrb_df, final_unmatched_dnb_df)
