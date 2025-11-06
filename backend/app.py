# backend/app.py
from flask import Flask, send_from_directory, request, jsonify, session
import pandas as pd
from hybrid import HybridRecommender

app = Flask(
    __name__,
    static_folder="E:/PROGRAMMING-2/Movie-Recommendation-System/frontend",
    static_url_path="E:/PROGRAMMING-2/Movie-Recommendation-System/frontend/assets"
)
app.secret_key = "change-this-to-a-strong-secret"  # set a real secret in prod

# ---- load data & build model once ----
DATA_PATH = "E:/PROGRAMMING-2/Movie-Recommendation-System/movielens_100k.csv"
df = pd.read_csv(DATA_PATH, encoding="latin-1")[['UserID','MovieID','Rating','Title','Genres']]

movies_df  = df[['MovieID','Title','Genres']].drop_duplicates('MovieID').reset_index(drop=True)
ratings_df = df[['UserID','MovieID','Rating']].copy()

recommender = HybridRecommender(movies_df, ratings_df)

def _get_feedback():
    fb = session.get("feedback", {"likes": [], "dislikes": []})
    fb["likes"]    = list({int(x) for x in fb.get("likes", [])})
    fb["dislikes"] = list({int(x) for x in fb.get("dislikes", []) if x not in fb["likes"]})
    return fb

@app.route("/")
def index():
    return send_from_directory(app.static_folder, "index.html")

# (Optional: still available if you want to inspect IDs)
@app.route("/users")
def users():
    users = sorted(ratings_df['UserID'].dropna().astype(int).unique().tolist())
    return jsonify(users)

# ---- feedback: store in session (no DB) ----
@app.route("/feedback", methods=["POST"])
def feedback():
    payload = request.get_json(silent=True) or {}
    movie_id = int(payload.get("movie_id"))
    like     = int(payload.get("like", 1))  # 1=like, 0=dislike
    fb = _get_feedback()
    if like:
        fb["likes"].append(movie_id)
        fb["dislikes"] = [m for m in fb["dislikes"] if m != movie_id]
    else:
        fb["dislikes"].append(movie_id)
        fb["likes"] = [m for m in fb["likes"] if m != movie_id]
    session["feedback"] = fb
    return jsonify({"ok": True, "feedback": fb})

# ---- recommend with light personalization from session feedback ----
@app.route("/recommend", methods=["POST"])
def recommend():
    try:
        payload = request.form if request.form else (request.get_json(silent=True) or {})
        user_id = int(payload.get("user_id"))
        top_n   = int(payload.get("top_n", 5))

        fb = _get_feedback()
        results = recommender.predict_for_user(
            user_id,
            top_n=top_n,
            session_likes=fb["likes"],
            session_dislikes=fb["dislikes"]
        )

        # enrich each item with a human-readable breakdown
        enriched = []
        for r in results:
            expl = recommender.explain(user_id, r["MovieID"], fb["likes"])
            r.update({
                "Prob": round(expl["prob"], 3),
                "CFScore": round(expl["cf_score"], 3),
                "top_genres": expl["top_genres"],
                "similar_title": expl["similar_title"],
                "similarity": round(expl["similarity"], 3),
            })
            # ready-to-show string if you prefer a single field:
            r["Why"] = (
                f"High on {', '.join(expl['top_genres'])}; "
                f"similar to '{expl['similar_title']}' (sim {r['similarity']}); "
                f"RF {r['Prob']}"
            )
            enriched.append(r)

        return jsonify(enriched)
    except Exception as e:
        return jsonify({"error": str(e)}), 400

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
