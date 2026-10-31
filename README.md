# Skincare skin-profile mismatch: data science project

Online shoppers cannot try skincare, and shops rank by overall rating, which
hides that a product can suit some skin types and not others. This project
cleans Sephora review data, explores it, and tests whether a poor fit can be
predicted from shopper skin profile and product information.

Data: Sephora Products and Skincare Reviews (Kaggle). Download
product_info.csv and reviews_1250-end.csv into this folder.

## Run
pip install -r requirements.txt
python clean_pipeline.py product_info.csv reviews_1250-end.csv out
python model_run.py
python model_checks.py

## Files
- clean_pipeline.py: cleaning and preprocessing (Week 3), writes out/ and cleaning_log.json
- eda.ipynb: exploratory analysis (Week 4)
- model_run.py: four models compared with a date split (Week 5)
- model_checks.py: bootstrap interval and scores by skin group
- model_results.csv: results table from model_run.py

## Requirements
pandas, numpy, scipy, scikit-learn, seaborn, matplotlib