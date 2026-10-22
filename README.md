# 🏠 California House Price Predictor

> A production-ready machine learning API that predicts **median house values** across California districts — powered by **FastAPI**, trained with **scikit-learn**, and deployed via **Docker**.

---

## 🐳 Docker Hub

[![Docker Image](https://img.shields.io/badge/Docker-karanbyte%2Fcalifornia--house--predictor-2496ED?logo=docker&logoColor=white)](https://hub.docker.com/repository/docker/karanbyte/california-house-predictor/general)

```bash
docker pull karanbyte/california-house-predictor:1.0
```

---

## 🚀 Quick Start

### ▶ Run with Docker *(Recommended)*

```bash
docker run -d -p 8000:8000 --name california-app karanbyte/california-house-predictor:1.0
```

| URL | Description |
|---|---|
| http://localhost:8000 | REST API |
| http://localhost:8000/docs | Interactive Swagger UI |
| http://localhost:8000/redoc | ReDoc Documentation |

---

### 💻 Run Locally

```bash
# Clone the repository
git clone https://github.com/Karanbyte07/california-house-predictor.git
cd california-house-predictor

# Set up virtual environment
python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # macOS / Linux

# Install dependencies
pip install -r requirements.txt

# Start the API server
uvicorn app:app --host 0.0.0.0 --port 8000 --reload

# (Optional) Launch the Streamlit UI in a separate terminal
streamlit run streamlit_app.py
```

---

## 📁 Project Structure

```
california-house-predictor/
│
├── app.py                    # FastAPI application entry point
├── streamlit_app.py          # Streamlit frontend UI
├── requirements.txt          # Python dependencies
├── Dockerfile                # Container build configuration
│
├── models/
│   ├── predict.py            # Inference logic
│   ├── housing_model.pkl     # Trained ML model (~144 MB)
│   └── housing_pipeline.pkl  # Preprocessing pipeline
│
├── schemas/
│   └── user_input.py         # Pydantic request / response schemas
│
├── dataset/
│   └── housing.csv           # California housing dataset
│
├── train model/
│   ├── train_model.py        # Model training script
│   └── housing_main.py       # Training entry point
│
└── test_cases/
    ├── input_data.csv         # Sample test inputs
    └── output_data.csv        # Expected outputs
```

---

## 🏗️ Architecture

```
Streamlit UI  ──POST /predict──►  FastAPI (app.py)
                                        │
                                  Pydantic Validation
                                        │
                                  models/predict.py
                                        │
                              ┌─────────┴──────────┐
                              ▼                    ▼
                   housing_pipeline.pkl    housing_model.pkl
                     (preprocessing)        (ML inference)
                              └─────────┬──────────┘
                                        ▼
                          predicted_median_house_value
```

---

## 📊 Model Input Parameters

The prediction model expects the following 9 geographical and socio-economic features:

| Parameter | Type | Description | Example / Allowed Values |
|---|---|---|---|
| `longitude` | `float` | East/West geographic coordinate | `-122.23` |
| `latitude` | `float` | North/South geographic coordinate | `37.88` |
| `housing_median_age` | `float` | Median age of houses in block group | `41.0` |
| `total_rooms` | `float` | Total number of rooms in block group | `880.0` |
| `total_bedrooms` | `float` | Total number of bedrooms in block group | `129.0` |
| `population` | `float` | Total population residing in block group | `322.0` |
| `households` | `float` | Total households (group residing in a home) | `126.0` |
| `median_income` | `float` | Median income in tens of thousands of USD | `8.3252` (~$83,252) |
| `ocean_proximity` | `string` | Location category relative to ocean/coast | `<1H OCEAN`, `INLAND`, `ISLAND`, `NEAR BAY`, `NEAR OCEAN` |

<details>
<summary><b>View Sample JSON Payload</b></summary>

```json
{
  "longitude": -122.23,
  "latitude": 37.88,
  "housing_median_age": 41.0,
  "total_rooms": 880.0,
  "total_bedrooms": 129.0,
  "population": 322.0,
  "households": 126.0,
  "median_income": 8.3252,
  "ocean_proximity": "NEAR BAY"
}
```
</details>

---

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| API Framework | FastAPI |
| ASGI Server | Uvicorn |
| Machine Learning | scikit-learn |
| Data Processing | pandas, numpy |
| Model Serialization | joblib |
| Input Validation | Pydantic v2 |
| Frontend UI | Streamlit |
| Containerization | Docker · python:3.11-slim |

---

## 👤 Author

**Karanjeet Kumar**
- 🐳 Docker Hub: [karanbyte](https://hub.docker.com/u/karanbyte)

---

## 📄 License

This project is open source and available under the [MIT License](LICENSE).
