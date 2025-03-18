# PCRB Capstone

# Introduction
1. Phase 1:
    - Intro: This phase will involve developing and deploying machine learning models such as NLP parsers as well as implementing data scrubbing and matching techniques such as fuzzy matching,  which will help us to accurately match the address between PCRB’ data and the D&B data and correct various data limitations and gaps to improve confidence levels of the matches.
    - Goal: Achieve at least 90% accuracy in data matching across the internal PCRB dataset and external D&B dataset. 
    - Measure of success: The accuracy will be measured as the percentage of successfully matched data rows from all the data in both datasets.
    - Deliverables
        - A combined dataset stored in SQL Server sandbox schema
        - A complete pipeline to clean, process, and match two datasets


# Deployment Instrction
1. Clone the project phase 1 from the main branch of Azure DevOps repository 'CMU Capstone':
```
git clone https://devops.pcrb.com/DefaultCollection/CMU%20Capstone/_git/CMU%20Capstone
```

2. Go to the Phase directory:
```
cd Phase1
```

3. Create a Python virtual environment based on Python 12.2.4 (either conda or venv is fine, in this examle we are using venv):
```
python -m venv env # Use 'python3 ...' for Python3 on Linux and MacOS
```

4. Activate the virtual environment:
```
env\Scripts\activate # use 'source env/bin/activate' on Linux and MacOS
```

5. Install all the packages that are required for this project (note that we did not specify the version of each package, but as the number of dependencies grows, we recommend to use version control to manage packages and dependencies more carefully to avoid potential conflicts):
```
pip install -r requirements.txt
```

6. Execute the pipeline by running Pipeline_Main.py directly.

7. Monitor the execution status by inspecting the console and the log file.

8. After the pipelin has been executed successfully (about 10 hours), the matched dataframe will be automatically uploaded to Snowflake database. You can inspect all the model evalution metrics in Dashboard.xlsx and the excel file in the folder 'evalution_metrics' (this folder will be automatically created if it does not exist).

9. Notes:
    - There will be some csv files generated in the folder 'Phase1' along the execution of the pipeline, they are intermediate files that cached using the self-defined decorator @cache_to_csv. They are mostly for testing purpose in case we want to make some minor changes and test the pipeline again , we don't need to wait another 10 hours to see the result. Instead, the previous steps are cached so that they will not be actually executed and therefore do not consume running time.


# General Workflow
The high level workflow is intuively reflected in the run() method in Pipeline.py:

1. Data Collection

2. Data Cleaing & Feature Enginnering

3. Model Building (matching solution)
