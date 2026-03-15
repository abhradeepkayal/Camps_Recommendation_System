# CAMPS Recommendation System

This repository contains the backend recommendation engine for the **CAMPS platform**.  
It recommends relevant college events to users based on their interests, past interactions, and event metadata.

---


## Architecture

The recommendation pipeline works as follows:

User
  │
  ▼
User Embeddings
  │
  ▼
FAISS Candidate Retrieval
  │
  ▼
Feature Engineering
  │
  ▼
LightGBM Ranking Model
  │
  ▼
MMR Diversification
  │
  ▼
Top-K Event Recommendations
  │
  ▼
FastAPI Endpoint
---

## Technologies Used

- Python
- FastAPI
- LightGBM
- FAISS
- NumPy
- Pandas
- Scikit-learn

---

## API Endpoints

### Health Check

GET /health

Example:

http://127.0.0.1:8000/health

Response:

{
  "status": "ok"
}

---

### Get Recommendations

GET /recommend/{user_id}

Example:

http://127.0.0.1:8000/recommend/100

Returns top recommended events for the user.

---

## Running the API Locally

Install dependencies:

pip install -r requirements.txt

Run the server:

uvicorn api.main:app --reload

Open API docs:

http://127.0.0.1:8000/docs

---

## Project Structure

api/
main.py

scripts/
recommend.py

models/
trained model files (not included in repo)

data/
datasets used for training (not included in repo)
