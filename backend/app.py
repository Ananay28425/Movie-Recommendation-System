# backend/app.py
from flask import Flask, send_from_directory, request, jsonify
import pandas as pd
from hybrid import HybridRecommender

app = Flask(
    __name__,
    static_folder="E:/PROGRAMMING-2/Movie-Recommendation-System/frontend",
    static_url_path="/"
)

# ---- load data & build model once on startup ----
DATA_PATH = "E:/PROGRAMMING-2/Movie-Recommendation-System/movielens_100k.csv"
df = pd.read_csv(DATA_PATH, encoding="latin-1")[['UserID','MovieID','Rating','Title','Genres']]

movies_df  = df[['MovieID','Title','Genres']].drop_duplicates('MovieID').reset_index(drop=True)
ratings_df = df[['UserID','MovieID','Rating']].copy()

recommender = HybridRecommender(movies_df, ratings_df)  # <-- pass TWO dataframes

# ---- Serve the SPA ----
@app.route("/")
def index():
    return send_from_directory(app.static_folder, "index.html")

# ---- API: Get users list (for dropdown) ----
@app.route("/users")
def users():
    users = sorted(ratings_df['UserID'].dropna().astype(int).unique().tolist())
    return jsonify(users)

# ---- API: Recommend top-N for a user ----
@app.route("/recommend", methods=["POST"])
def recommend():
    try:
        payload = request.form if request.form else (request.get_json(silent=True) or {})
        user_id = int(payload.get("user_id"))
        top_n   = int(payload.get("top_n", 5))
        results = recommender.predict_for_user(user_id, top_n=top_n)  # <-- correct method name
        return jsonify(results)
    except Exception as e:
        return jsonify({"error": str(e)}), 400

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
