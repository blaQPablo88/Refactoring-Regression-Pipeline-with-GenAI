
# Regression Pipeline API Documentation

## Overview
This API provides a simple interface to use a trained machine learning regression model for making predictions. It is part of a larger project that builds a full regression modeling pipeline using scikit-learn, FastAPI, and joblib.

---

## 🔧 Base URL
```
http://localhost:8000
```

You can run the server using:
```bash
uvicorn app:app --reload
```

---

## 📬 Endpoints

### 1. **`POST /predict`**
Make a prediction using the trained regression model.

#### Request
- **URL**: `/predict`
- **Method**: `POST`
- **Payload**: A JSON object containing feature values
- **Content-Type**: `application/json`

#### Example Payload
```json
{
  "feature1": 0.532,
  "feature2": 134.0,
  "feature3": 0.002,
  "feature4": 18.3,
  ...
}
```
⚠️ Note: You must include **all features** used in the model's training and preprocessing.

#### Response
```json
{
  "prediction": [137.42]
}
```
- A list containing the predicted label(s) from the model.

---

## ⚙️ Internal Workflow
When the `/predict` endpoint is called:

1. JSON is received and converted into a pandas DataFrame.
2. Infinite values are cleaned.
3. Only selected features (`selected_columns.joblib`) are kept.
4. Data is scaled using the trained transformer (`scaler_transformer.joblib`).
5. The model (`LinearRegression.joblib`) predicts the output.

---

## 💾 Model Artifacts
The following serialized files are used:

- `models/LinearRegression.joblib`: The trained regression model
- `models/scaler_transformer.joblib`: StandardScaler/ColumnTransformer fitted on training data
- `models/selected_columns.joblib`: List of selected feature columns

---

## 🧪 Testing the API
You can test the API using:
- **curl**
```bash
curl -X POST http://localhost:8000/predict      -H "Content-Type: application/json"      -d '{"feature1": 0.5, "feature2": 134.0, ... }'
```
- **Postman** or **Thunder Client** in VSCode
- **FastAPI's Swagger UI**:
  Visit `http://localhost:8000/docs` in your browser

---

## 🐞 Troubleshooting
| Issue | Possible Cause | Fix |
|-------|----------------|-----|
| `KeyError` for a column | Missing or misspelled feature in payload | Ensure payload matches expected features |
| `ValueError` during transform | Feature types mismatch | Match data types (float/int) and ensure all columns are present |
| `FileNotFoundError` | joblib model files not found | Re-train model and ensure files exist in `models/` |

---

## 📌 Notes
- Input features must match those used during training exactly
- Model expects pre-cleaned numeric input
- Only `POST` is supported for prediction

---

## 📃 License
This project is licensed under the MIT License.

---

## 🙋 Author
Kagiso Mogotsi (blaQPablo88)

Inspired by Kaggle project by Yash Sahu: https://www.kaggle.com/code/yashsahu02/drw-crypto-market-prediction
