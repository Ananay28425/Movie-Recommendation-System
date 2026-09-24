"""FastAPI adapter and in-memory feedback re-ranking for the hybrid model."""
from contextlib import asynccontextmanager
from collections import Counter
from pathlib import Path
from threading import Lock
from typing import Literal

import numpy as np
from fastapi import FastAPI, HTTPException, Path as PathParam, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from hybrid import HybridRecommender, download_movielens_data, load_movielens_data

DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "movielens_100k.csv"
user_feedback: dict[int, dict[str, set[int]]] = {}
feedback_lock = Lock()


@asynccontextmanager
async def lifespan(app: FastAPI):
    download_movielens_data(DATA_PATH)
    movies, ratings = load_movielens_data(DATA_PATH)
    app.state.model = HybridRecommender(movies, ratings)
    features = app.state.model.item_features.astype(float)
    lengths = np.linalg.norm(features, axis=1, keepdims=True)
    app.state.unit_features = np.divide(features, lengths, out=np.zeros_like(features), where=lengths > 0)
    yield


app = FastAPI(title="Adaptive Movies API", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type"],
)


class Feedback(BaseModel):
    user_id: int = Field(gt=0, strict=True)
    movie_id: int = Field(gt=0, strict=True)
    signal: Literal["like", "reject"]


def taste_profile(model: HybridRecommender, likes: set[int], dislikes: set[int]) -> dict:
    scores: Counter[str] = Counter()
    for movie_id, weight in [(mid, 1) for mid in likes] + [(mid, -1) for mid in dislikes]:
        for genre in str(model.movie_lookup.loc[movie_id, "Genres"]).split("|"):
            if genre and genre != "Unknown":
                scores[genre] += weight
    top = sorted((genre for genre, score in scores.items() if score > 0), key=lambda genre: (-scores[genre], genre))[:3]
    return {"likes": len(likes), "dislikes": len(dislikes), "top_genres": top}


@app.get("/api/health")
def health():
    return {"status": "ok"}


@app.get("/api/recommendations/{user_id}")
def recommendations(
    user_id: int = PathParam(gt=0),
    limit: int = Query(default=6, ge=1, le=24),
):
    model: HybridRecommender = app.state.model
    if user_id not in model.user_index:
        raise HTTPException(status_code=404, detail=f"User {user_id} is not in MovieLens 100K (valid IDs: 1–943).")

    with feedback_lock:
        entry = user_feedback.get(user_id, {"likes": set(), "dislikes": set()})
        likes, dislikes = entry["likes"].copy(), entry["dislikes"].copy()

    vectors = app.state.unit_features
    preference = np.zeros(vectors.shape[1], dtype=float)
    if likes:
        preference += vectors[[model.movie_index[mid] for mid in sorted(likes)]].mean(axis=0)
    if dislikes:
        preference -= 0.5 * vectors[[model.movie_index[mid] for mid in sorted(dislikes)]].mean(axis=0)
    has_feedback = bool(likes or dislikes)
    # Keep the preference's magnitude: normalizing it after every click could
    # make a rejection increase scores even for similar movies.
    similarity = np.clip(vectors @ preference, -1.0, 1.0) if has_feedback else np.zeros(len(model.movie_ids))

    # Cosine similarity is in [-1, 1]. A 0.8-point maximum adjustment is
    # 20% of the four-point MovieLens rating span. The trained model is unchanged.
    candidates = []
    for item in model.predict_for_user(user_id, top_n=len(model.movie_ids)):
        movie_id = item["MovieID"]
        if movie_id in likes or movie_id in dislikes:
            continue
        base = float(item["FinalScore"])
        personal = float(similarity[model.movie_index[movie_id]])
        final = float(np.clip(base + 0.8 * personal, 1.0, 5.0))
        contribution = final - base
        explanation = item["Explanation"].replace("RF", "Random Forest")
        if has_feedback:
            explanation += "; feedback favors similar movies" if contribution > 0.005 else (
                "; feedback pushes down similar movies" if contribution < -0.005 else "; feedback has little effect"
            )
        candidates.append({
            "movie_id": movie_id,
            "title": item["Title"],
            "genres": str(model.movie_lookup.loc[movie_id, "Genres"]),
            "base_score": round(base, 3),
            "personalization_score": round(personal, 3),
            "personalization_contribution": round(contribution, 3),
            "final_score": round(final, 3),
            "explanation": explanation,
        })

    candidates.sort(key=lambda item: (-item["final_score"], item["movie_id"]))
    return {
        "user_id": user_id,
        "recommendations": candidates[:limit],
        "profile": taste_profile(model, likes, dislikes),
    }


@app.post("/api/feedback")
def feedback(payload: Feedback):
    model: HybridRecommender = app.state.model
    if payload.user_id not in model.user_index:
        raise HTTPException(status_code=404, detail="Unknown MovieLens user ID.")
    if payload.movie_id not in model.movie_index:
        raise HTTPException(status_code=404, detail="Unknown MovieLens movie ID.")
    with feedback_lock:
        entry = user_feedback.setdefault(payload.user_id, {"likes": set(), "dislikes": set()})
        target = "likes" if payload.signal == "like" else "dislikes"
        other = "dislikes" if target == "likes" else "likes"
        entry[other].discard(payload.movie_id)
        entry[target].add(payload.movie_id)
        profile = taste_profile(model, entry["likes"], entry["dislikes"])
    return {"status": "recorded", "title": str(model.movie_lookup.loc[payload.movie_id, "Title"]), "profile": profile}
