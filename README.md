# Refactoring Regression Pipeline with GenAI

Refactoring my [First Pipeline](https://github.com/blaQPablo88/First_Pipeline_Python) project using GenAI engineering principles to improve modularity, readability, and performance.

---

---

## 🙏 Acknowledgements

This project is inspired by and based on the work by [Yash Sahu](https://www.kaggle.com/code/yashsahu02/drw-crypto-market-prediction) on Kaggle.

Credit to the original author for the initial data exploration, model experimentation, and notebook structure.

This refactor focuses on improving modularity, maintainability, and scalability using GenAI best practices.

## 📦 Installation Guide

### ✅ Required Libraries

Install the following modules using `pip` (run in terminal or inside your virtual environment):

```bash
pip install numpy pandas matplotlib seaborn scikit-learn xgboost lightgbm catboost pyarrow kaggle
```

> ⚠️ Make sure you install all dependencies **before running the scripts**.

---

### ✅ System-Level Installation

**Numpy (System Install)**  
Refer to [numpy.org/install](https://numpy.org/install)

- **Windows:**
  ```bash
  choco install numpy
  ```

- **Linux:**
  ```bash
  sudo apt install python3-numpy
  ```

> Test numpy install:
```python
import numpy as np
print(np.__version__)
```

---

## 🧾 What’s a `.parquet` File?

A `.parquet` file is a highly efficient, column-based storage format used in big data environments.

### 🔍 Why Use Parquet?
- **Columnar Format:** Optimized for analytics and queries
- **Efficient Compression:** Smaller file sizes than CSV
- **Schema-aware:** Retains data types and structure
- **Performance:** Faster reading when you only need some columns

Used heavily with tools like Pandas, PyArrow, Spark, and in ML workflows.

---

## 📁 Project Structure

```
crypto-regression/
│
├── data/
│   └── kaggle/                # Raw input data from Kaggle
│
├── models/
│   └── saved_predictions/     # Output predictions by model
│
├── src/
│   ├── data_loader.py         # Handles data import and preprocessing
│   ├── preprocessing.py       # Data cleaning and transformation
│   ├── feature_selection.py   # SelectKBest or other techniques
│   ├── train_models.py        # Train & evaluate models
│   └── utils.py               # Metrics, evaluation, helper functions
│
├── main.py                    # Main orchestration script
└── README.md                  # You are here 📘
```

---

## 📥 Data Access

Make sure you have Kaggle CLI configured and authenticated, then download the dataset:

```bash
pip install kaggle
kaggle datasets download -d yashsahu02/drw-crypto-market-prediction
```

---

## 🚀 Running the Project

Once dependencies are installed and data is in place:

```bash
python main.py
```

This will:
- Load the dataset
- Clean and preprocess the data
- Select features
- Train models (e.g., Linear Regression, XGBoost)
- Save predictions for submission

---

## 📊 Outputs

- Model evaluation metrics printed in console
- `.csv` prediction files per model (saved in `models/saved_predictions/`)
- `model_performance_summary.csv` for comparison of algorithms

---

## 🧠 Powered by GenAI Refactor Principles

This project demonstrates how AI-assisted refactoring can improve a raw ML script into a clean, modular, maintainable pipeline.

---

## 📌 To-Do (Next Steps)
- [ ] Add more regression models
- [ ] Integrate logging instead of print statements
- [ ] Unit tests for data loading and evaluation
- [ ] Deploy via Streamlit or FastAPI

---

Created with ❤️ by Kagiso using GenAI.