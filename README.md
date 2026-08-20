# 🩺 Medical Disease Predictor

A machine learning-powered disease prediction system with a **FastAPI** backend and **Streamlit** frontend.

## Project Structure

| File | Description |
|---|---|
| `main.py` | FastAPI backend — loads the ML model and serves prediction & symptom endpoints |
| `app.py` | Streamlit frontend — UI that connects to the FastAPI backend |
| `train_model.py` | Script to train the Random Forest model from the dataset |
| `dataset.csv` | Training dataset with symptoms and diseases |

## Setup

### 1. Install Dependencies

```bash
pip install streamlit fastapi uvicorn pandas joblib scikit-learn requests
```

### 2. Train the Model

```bash
python train_model.py
```

This generates `disease_model.pkl` and `symptoms_list.pkl`.

### 3. Run the Backend

```bash
uvicorn main:app --reload
```

The API will be available at `http://localhost:8000`.

### 4. Run the Frontend

```bash
streamlit run app.py
```

The UI will open at `http://localhost:8501`.

## API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/symptoms` | Returns the list of available symptoms |
| `POST` | `/predict` | Accepts symptoms and returns predicted disease |

## Disclaimer

This tool is for **educational purposes only** and is not a substitute for professional medical advice.
