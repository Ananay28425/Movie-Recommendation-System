# ADAPTIVE//MOVIES

A small full-stack MovieLens 100K recommender. A hybrid model ranks movies a user has not rated. **Like** or **Reject** a suggestion and its ranking changes immediately, without retraining the model. The interface shows the original model score, feedback adjustment, and final score for every recommendation.

## How it works

```text
MovieLens ratings + movie titles/genres
                 │
     ┌───────────┼───────────────┐
     ▼           ▼               ▼
TF-IDF + SVD   User KNN    RandomForestRegressor
     └───────────┼───────────────┘
                 ▼
        Weighted hybrid base score
                 │
        Feedback feature alignment
                 ▼
     Re-ranked unseen recommendations → React UI
```

- **Content:** TF-IDF on title and genres, reduced with TruncatedSVD; a user's ratings form a weighted content profile.
- **Collaborative:** cosine KNN finds neighboring users in the sparse rating matrix. Predictions use observed neighbor ratings and per-item normalization.
- **Regression:** a `RandomForestRegressor` estimates ratings from movie content features. It trains once when the API starts.
- **Hybrid:** weights depend on how many ratings the user has; the Random Forest carries the largest base weight. This implementation does not claim experimentally optimized weights or measured production accuracy.
- **Adaptation:** normalize the same movie feature vectors to unit length, then compute `preference = mean(liked vectors) − 0.5 × mean(rejected vectors)` (empty sides contribute zero). For a candidate, `alignment = clip(candidate · preference, −1, 1)` and `final = clip(base + 0.8 × alignment, 1, 5)`. Keeping the preference magnitude means a relevant rejection subtracts from the candidate score even after a related like. The feedback adjustment is at most 0.8 points, 20% of the four-point rating span. The API returns signed alignment and actual score adjustment; clipping can make the adjustment smaller. Movies already rated or explicitly liked/rejected are excluded. Genre counts add one for likes and subtract one for rejects, with deterministic ties.

## Stack and layout

Python, FastAPI, pandas, NumPy, SciPy, scikit-learn; React, Vite, and plain CSS. All API logic lives in `backend/main.py`; the existing ML engine lives in `backend/hybrid.py`. `frontend/src/App.jsx` holds the one-page UI and `frontend/src/styles.css` its styling. `data/movielens_100k.csv` is created locally when the server first starts and is gitignored.

## Run locally

Python 3.10+ and Node 20.19+ (or 22.12+) are recommended. From the repository root:

```bash
python -m venv .venv
```

Activate `.venv` with `.venv\Scripts\Activate.ps1` in PowerShell or `source .venv/bin/activate` on macOS/Linux. Then start the API:

```bash
python -m pip install -r backend/requirements.txt
cd backend
python -m uvicorn main:app --host 127.0.0.1 --port 8000
```

The **first start** downloads the official [MovieLens 100K archive](https://files.grouplens.org/datasets/movielens/ml-100k.zip), converts `u.data` and `u.item` into the local CSV, and trains the model. This takes longer than subsequent UI interactions. If the automatic download fails, download that archive manually into `data/ml-100k.zip` and restart the server. GroupLens [terms prohibit redistributing this dataset without separate permission](https://files.grouplens.org/datasets/movielens/ml-100k/README), so the archive and generated CSV are deliberately excluded from git. Keep your local copy under the dataset's terms. The app needs internet access once unless you supply the archive. At startup it validates that the CSV contains the official 100,000 ratings, 943 users, and 1,682 movies. If an existing CSV fails this check, remove it and restart to fetch the official archive.

The current tree omits the dataset. An older commit included a CSV at the repository root, so that file remains in Git history until the history is rewritten.

In a second terminal:

```bash
cd frontend
npm ci
npm run dev
```

Open <http://localhost:5173>. Vite proxies `/api` to port 8000. API documentation is at <http://127.0.0.1:8000/docs>. Use a valid MovieLens user ID, such as `54` (IDs 1–943).

## API

| Method | Path | Purpose |
| --- | --- | --- |
| GET | `/api/health` | `{"status":"ok"}` |
| GET | `/api/recommendations/54?limit=6` | Unseen ranked movies, base/personal/final scores and genre profile; `limit` is 1–24 |
| POST | `/api/feedback` | Record `{"user_id":54,"movie_id":123,"signal":"like"}` or `"reject"` |

Unknown users and movies return 404; malformed feedback or out-of-range limits return 422. Re-request recommendations after posting feedback to see the changed order and scores. Feedback is **in memory per server process**: it survives a browser refresh but resets when the API restarts. Keep one API worker for the demo. Recent feedback listed in the UI is browser state and resets on refresh; the server's taste counts remain until restart.

## Scope and next steps

MovieLens 100K is small and dated. The Random Forest uses movie content features only (not user features), and base weights are heuristic. Feedback is not persisted or shared between workers. Explanations describe the scoring signals rather than proving why a person will like a film. A sensible next iteration would evaluate against held-out ratings with per-user splits, measure ranking metrics, tune the weights on validation data, and persist feedback if the demo becomes a multi-user service.
