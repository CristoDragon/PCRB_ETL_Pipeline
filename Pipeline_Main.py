import pandas as pd
import numpy as np
from sqlalchemy import create_engine
from fuzzywuzzy import fuzz
import re
from tqdm import tqdm
import logging
import pyodbc
import urllib
import openpyxl
import configparser
import os
import time
import datetime
import sys
import src.Pipeline as pl
import uuid

# Date: 09/21/2024
# Description: This is the main program to run the pipeline to clean, process, and match the PCRB dataset and D&B dataset.

def main():
    # Configure basic settings for logging to a file
    logging.basicConfig(
        level=logging.DEBUG,
        filename="pipeline.log",
        encoding="utf-8",
        filemode="a",  # Append to the log file
        format="{asctime} - {levelname} - {message}",
        style="{",
        datefmt="%Y-%m-%d %H:%M"
    )
    # Create a console handler for output to the terminal
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.DEBUG)
    # Set a formatter for the console handler to match the file format
    console_formatter = logging.Formatter(
        "{asctime} - {levelname} - {message}",
        style="{",
        datefmt="%Y-%m-%d %H:%M"
    )
    console_handler.setFormatter(console_formatter)
    # Add the console handler to the root logger
    logging.getLogger().addHandler(console_handler)
    # Set logging level for sentence-transformers and transformers modules
    logging.getLogger("transformers").setLevel(logging.WARNING)
    logging.getLogger("sentence_transformers").setLevel(logging.WARNING)
    logging.info(f"Started to execute the pipeline")
    RUN_ID = str(uuid.uuid4()) 
    # Use a configuration file to store sensitive information
    config = configparser.ConfigParser()
    # Determine the file path of the config script
    config.read(os.path.join(os.path.dirname(os.path.realpath(__file__)), 'config.ini'))
    # Initialize an instance of Pipeline class
    pipeline = pl.Pipeline(config, RUN_ID)
    # Execute the pipeline job
    pipeline.run()


if __name__ == "__main__":
    try:
        start = time.time()
        # Call the main method to execute the pipeline
        main()
        end = time.time()
        total = end - start
        logging.info("Pipeline has been execute successfully (" + "{:.2f}".format(total) + "s)\n")
    except Exception as e:
        logging.info(f"Pipeline has been aborted while executing due to errors: {e}\n")
    sys.exit(1)
