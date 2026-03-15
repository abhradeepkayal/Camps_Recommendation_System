from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# create API
app = FastAPI()

# allow frontend access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# import recommender
from scripts.recommend import recommend


@app.get("/")
def home():
    return {"message": "CAMPS Recommender API Running"}

@app.get("/health")
def health():
    return {"status": "ok"}
@app.get("/recommend/{user_id}")
def recommend_events(user_id: int, top_k: int = 10):

    result = recommend(user_id, top_k)

    return result.to_dict(orient="records")