"""
Refactoring Regression Pipeline with GenAI
==========================================

This project refactors the original pipeline from 
https://github.com/blaQPablo88/First_Pipeline_Python using GenAI engineering principles to 
improve modularity, readability, and performance.

Acknowledgements
----------------
This work is inspired by the Kaggle notebook by Yash Sahu:
https://www.kaggle.com/code/yashsahu02/drw-crypto-market-prediction

The initial exploration, modeling, and structure were contributed by the original author.
This refactor improves modularity, maintainability, and scalability using GenAI best practices.

Installation Guide
------------------
Install required Python libraries via `pip`:

    pip install numpy pandas matplotlib seaborn scikit-learn xgboost lightgbm catboost pyarrow kaggle

System-Level Installation
^^^^^^^^^^^^^^^^^^^^^^^^^
To install NumPy system-wide, follow:

- Windows:
    choco install numpy

- Linux:
    sudo apt install python3-numpy

Test the installation:

    >>> import numpy as np
    >>> print(np.__version__)

Parquet File Overview
---------------------
Parquet is a highly efficient columnar file format used in big data analytics. It is:
    
- Columnar and optimized for analytical queries.
- Schema-aware (retains column metadata).
- Efficient with compression and reading performance.
- Widely compatible with Pandas, PyArrow, Spark, etc.

Project Structure
-----------------
crypto-regression/
├── data/
│   └── kaggle/                
├── models/
│   └── saved_predictions/     
├── src/
│   ├── data_loader.py         
│   ├── preprocessing.py       
│   ├── feature_selection.py   
│   ├── train_models.py        
│   └── utils.py               
├── main.py                    
└── README.md                  

Data Access
-----------
Download the dataset using Kaggle CLI:

    pip install kaggle
    kaggle datasets download -d yashsahu02/drw-crypto-market-prediction

Make sure your Kaggle token is authenticated and the data is placed in the correct directory.

Running the Project
-------------------
After installing dependencies and downloading the data:

    python main.py

The main script performs:
    
- Data loading
- Cleaning & preprocessing
- Feature selection
- Model training (e.g., Linear Regression, XGBoost)
- Saving predictions

Outputs
-------
- Printed model evaluation metrics
- Per-model prediction CSV files (in `models/saved_predictions/`)
- Summary performance CSV (`model_performance_summary.csv`)

Powered by GenAI Refactor Principles
------------------------------------
This project showcases how GenAI-assisted refactoring transforms a raw machine learning pipeline
into a clean, reusable, and production-friendly framework.

Created by: Kagiso (aka blaQPablo88)
"""