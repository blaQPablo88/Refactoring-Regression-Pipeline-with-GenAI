
# Regression-Pipeline-with-GenAI

A comprehensive end-to-end regression modeling pipeline for a Kaggle competition, refactored using GenAI engineering principles.  
The pipeline extracts data, transforms it through preprocessing and feature selection, and trains multiple regression models to predict a continuous target.

---

## Key Features

- Modular pipeline using reusable scripts
- Load `.parquet` and `.csv` data formats
- Cleans and preprocesses raw data
- Feature selection using `SelectKBest`
- Model training and evaluation using multiple regressors
- Saves predictions for submission
- Summary of model performance written to `.csv`

---

## Technologies Used

- [Python 3.10+](https://www.python.org/)
- [pandas](https://pandas.pydata.org/)
- [numpy](https://numpy.org/)
- [scikit-learn](https://scikit-learn.org/stable/)

> This project is designed for tabular regression tasks using open Kaggle datasets.

---

## Installation Instructions

1. **Clone the repository:**

   ```bash
   git clone https://github.com/blaQPablo88/Refactoring-Regression-Pipeline-with-GenAI
   cd Refactoring-Regression-Pipeline-with-GenAI
   ```
2. **Install all required packages (at once):**
   ```bash
   pip install -r requirements.txt
   ```

   **OR YOU COULD DO IT INDIVISUALLY**

3. **Install required packages:**

   ```bash
   pip install numpy pandas scikit-learn pyarrow
   ```

4. **Ensure Kaggle CLI is set up** *(for downloading competition datasets)*

   ```bash
   pip install kaggle
   kaggle datasets download -d yashsahu02/drw-crypto-market-prediction
   ```

5. **Extract and place the data** under:

   ```
   /kaggle/input/drw-crypto-market-prediction/
   ```

---

## Project Structure

```
Regression-Pipeline-with-GenAI/
│
├── kaggle/
│   └── input/
│       └── drw-crypto-market-prediction/     # Place dataset files here
│
├── models/
│   └── saved_predictions/                    # Saved model outputs
│
├── src/
│   ├── data_loader.py                        # Loads .parquet and CSV files
│   ├── preprocessing.py                      # Cleans and scales data
│   ├── feature_selection.py                  # Feature selection using SelectKBest
│   ├── train_models.py                       # Training and evaluating models
│   └── utils.py                              # Evaluation metrics
│
├── main.py                                   # Orchestrates full pipeline
└── README.md
```

---

## How to Run

```bash
python main.py
```

This will:

1. Load the Kaggle dataset
2. Clean and scale the data
3. Select the top 200 features
4. Train and evaluate two linear regression models
5. Save predictions and a summary file

---

## Configuration Options

Edit `main.py` to:

- Change dataset paths
- Select more/less features (`k` in `SelectKBest`)
- Add/remove regression models from the `models` dictionary
- Switch scaling methods

---

## Troubleshooting

| Issue | Fix |
|------|------|
| `FileNotFoundError` when loading `.parquet` | Ensure dataset is placed under `/kaggle/input/drw-crypto-market-prediction/` |
| `pyarrow` not found | Run `pip install pyarrow` |
| Feature mismatch errors during prediction | Ensure `test_df` only contains selected columns |

---

## Contributing

Contributions are welcome!

1. Fork the repo
2. Create a new branch (`feature/my-feature`)
3. Make your changes
4. Submit a pull request

---

## License

MIT License. See [LICENSE](LICENSE) for details.

---

## Acknowledgements

This project is inspired by and based on [Yash Sahu's notebook on Kaggle](https://www.kaggle.com/code/yashsahu02/drw-crypto-market-prediction).  
Credit to the original author for data exploration, modeling flow, and base structure.

This refactor emphasizes **modularity, scalability, and maintainability** through GenAI principles.

---

Built by Kagiso ([@blaQPablo88](https://github.com/blaQPablo88)) using GenAI assistance.
