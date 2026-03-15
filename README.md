# CAMPS Event Recommendation System

This repository contains the backend recommendation system for the CAMPS platform.

The system recommends relevant college events to users using a hybrid recommendation architecture.

---

## Architecture

User → Embeddings → FAISS Candidate Retrieval → Feature Engineering → LightGBM Ranking → MMR Diversity → Top K Events

---

## Technologies Used

- Python
- FastAPI
- LightGBM
- FAISS
- Pandas
- NumPy
- Scikit-learn

---

## API Endpoints

### Health Check

GET /health

Returns:

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

## Running the API

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
trained recommendation models

data/
event and user datasets