import os
import logging
import pandas as pd
from datetime import datetime

# Date: 12/1/2024
# Author: CMU Capstone Team (Dragon, Michael, Nirvik, Karl)
# Description: This module contains code relevant to the data cleaning part in machine learning pipeline.

def get_latest_evaluation_metrics(RUN_ID):
    """
    Retrieve metrics from the latest model evaluation file for the given RUN_ID and print them.
    """
    try:
        # Ensure the directory exists
        metrics_dir = './evaluation_metrics/'
        if not os.path.exists(metrics_dir):
            os.makedirs(metrics_dir)
            logging.info(f"Directory '{metrics_dir}' created.")
 
        # Construct the filename using RUN_ID
        metrics_filepath = os.path.join(metrics_dir, f"model_evaluation_{RUN_ID}.csv")
 
        if not os.path.exists(metrics_filepath):
            logging.warning(f"No evaluation file found for RUN_ID {RUN_ID}.")
            print(f"No evaluation file found for RUN_ID {RUN_ID}.")
            return
 
        logging.info(f"Loading metrics from {metrics_filepath}")
 
        # Read and display the metrics
        try:
            metrics_df = pd.read_csv(metrics_filepath)
            print(metrics_df)
        except pd.errors.EmptyDataError:
            logging.error(f"The file {metrics_filepath} is empty or corrupted.")
            print(f"The file {metrics_filepath} is empty or corrupted.")
            return
 
    except Exception as e:
        logging.error(f"An error occurred: {e}")
        print(f"An error occurred: {e}")
 
def log_matching_metrics(description, rec_before_match, rec_matched, RUN_ID):
    """
    Logs and stores the percentage of records matched during a matching phase.
 
    Args:
        description (str): Description of the operation.
        rec_before_match (int): Total number of records before matching.
        rec_matched (int): Total number of matched records.
        RUN_ID (str): Unique identifier for the pipeline run.
    """
    try:
        # Calculate percentage of records matched
        percentage_matched = (rec_matched / rec_before_match) * 100 if rec_before_match > 0 else 0
 
        # Ensure the directory exists
        metrics_dir = './evaluation_metrics/'
        if not os.path.exists(metrics_dir):
            os.makedirs(metrics_dir)
            logging.info(f"Directory '{metrics_dir}' created.")
 
        # Prepare metrics for logging
        metrics_filepath = os.path.join(metrics_dir, f"model_evaluation_{RUN_ID}.csv")
        metrics = [
            {
                "Description of Operation": description,
                "Rows Before": rec_before_match,
                "Rows After": rec_matched,
                "Percentage Change": percentage_matched,
                "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }
        ]
 
        # Save metrics to the model evaluation file
        metrics_df = pd.DataFrame(metrics)
        if os.path.exists(metrics_filepath):
            metrics_df.to_csv(metrics_filepath, mode='a', header=False, index=False)
        else:
            metrics_df.to_csv(metrics_filepath, index=False)
 
        logging.info(f"Matching metrics saved to {metrics_filepath}.")
    except Exception as e:
        logging.error(f"Error logging matching metrics: {e}")
        raise