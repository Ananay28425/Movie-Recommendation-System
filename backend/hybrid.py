from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.decomposition import TruncatedSVD
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.neighbors import NearestNeighbors


class HybridRecommender:
    """
    Hybrid recommender for MovieLens-style data.

    Compatible with:
    - Flask backend import usage
    - Direct CLI execution for evaluation output

    Expected columns:
    movies_df: MovieID, Title, Genres
    ratings_df: UserID, MovieID, Rating
    """

    def __init__(self, movies_df: pd.DataFrame, ratings_df: pd.DataFrame, random_state: int = 42):
        self.random_state = random_state

        self.movies = movies_df.copy()
        self.ratings = ratings_df.copy()

        self._validate_and_prepare_data()

        self.user_ids = sorted(self.ratings["UserID"].unique().tolist())
        self.movie_ids = sorted(self.movies["MovieID"].unique().tolist())

        self.user_index = {uid: i for i, uid in enumerate(self.user_ids)}
        self.movie_index = {mid: i for i, mid in enumerate(self.movie_ids)}

        self.movie_lookup = self.movies.set_index("MovieID", drop=True)

        self.global_mean_rating = float(self.ratings["Rating"].mean()) if len(self.ratings) else 3.0
        self.movie_mean_rating = self.ratings.groupby("MovieID")["Rating"].mean().to_dict()
        self.user_mean_rating = self.ratings.groupby("UserID")["Rating"].mean().to_dict()

        # Build the matrix first. Everything else depends on this.
        self.user_item_matrix = self._create_user_item_matrix()
        self.user_item_sparse = csr_matrix(self.user_item_matrix.values.astype(np.float32))

        # Item features and models
        self.item_features = self._build_tfidf_features()
        self.knn = self._train_knn()
        self.rf = self._train_random_forest()

        # Precompute model-wide item predictions once
        self.rf_scores = np.asarray(self.rf.predict(self.item_features), dtype=float)
        self.rf_scores = np.clip(self.rf_scores, 1.0, 5.0)

        # Per-user caches so evaluation and recommendations do not recompute the same thing repeatedly
        self._content_cache: Dict[int, np.ndarray] = {}
        self._cf_cache: Dict[int, np.ndarray] = {}
        self._hybrid_cache: Dict[int, np.ndarray] = {}

    def _validate_and_prepare_data(self) -> None:
        required_movie_cols = {"MovieID", "Title", "Genres"}
        required_rating_cols = {"UserID", "MovieID", "Rating"}

        missing_movie_cols = required_movie_cols - set(self.movies.columns)
        missing_rating_cols = required_rating_cols - set(self.ratings.columns)

        if missing_movie_cols:
            raise ValueError(f"movies_df is missing columns: {sorted(missing_movie_cols)}")
        if missing_rating_cols:
            raise ValueError(f"ratings_df is missing columns: {sorted(missing_rating_cols)}")

        self.movies = self.movies.copy()
        self.ratings = self.ratings.copy()

        self.movies["MovieID"] = pd.to_numeric(self.movies["MovieID"], errors="coerce")
        self.ratings["UserID"] = pd.to_numeric(self.ratings["UserID"], errors="coerce")
        self.ratings["MovieID"] = pd.to_numeric(self.ratings["MovieID"], errors="coerce")
        self.ratings["Rating"] = pd.to_numeric(self.ratings["Rating"], errors="coerce")

        self.movies = self.movies.dropna(subset=["MovieID"])
        self.ratings = self.ratings.dropna(subset=["UserID", "MovieID", "Rating"])

        self.movies["MovieID"] = self.movies["MovieID"].astype(int)
        self.ratings["UserID"] = self.ratings["UserID"].astype(int)
        self.ratings["MovieID"] = self.ratings["MovieID"].astype(int)
        self.ratings["Rating"] = self.ratings["Rating"].astype(float)

        self.movies = self.movies.drop_duplicates(subset="MovieID").copy()
        self.movies = self.movies.sort_values("MovieID").reset_index(drop=True)

        self.ratings = self.ratings.drop_duplicates(subset=["UserID", "MovieID"]).reset_index(drop=True)

        self.movies["Title"] = self.movies["Title"].fillna("").astype(str)
        self.movies["Genres"] = self.movies["Genres"].fillna("").astype(str)

    def _create_user_item_matrix(self) -> pd.DataFrame:
        matrix = self.ratings.pivot_table(
            index="UserID",
            columns="MovieID",
            values="Rating",
            aggfunc="mean",
        ).fillna(0.0)

        matrix = matrix.reindex(index=self.user_ids, columns=self.movie_ids, fill_value=0.0)
        return matrix

    def _build_tfidf_features(self) -> np.ndarray:
        content_series = (
            self.movies["Title"].fillna("").astype(str)
            + " "
            + self.movies["Genres"].fillna("").astype(str).str.replace("|", " ", regex=False)
        )

        tfidf = TfidfVectorizer(
            stop_words="english",
            max_features=5000,
            ngram_range=(1, 2),
        )
        tfidf_matrix = tfidf.fit_transform(content_series)

        n_samples, n_features = tfidf_matrix.shape
        max_possible = min(n_samples, n_features) - 1

        if max_possible >= 2:
            n_components = min(100, max_possible)
            svd = TruncatedSVD(n_components=n_components, random_state=self.random_state)
            reduced = svd.fit_transform(tfidf_matrix)
        else:
            reduced = tfidf_matrix.toarray()

        return np.asarray(reduced, dtype=np.float32)

    def _train_knn(self) -> NearestNeighbors:
        n_neighbors = min(20, len(self.user_ids))
        n_neighbors = max(1, n_neighbors)

        model = NearestNeighbors(metric="cosine", algorithm="brute", n_neighbors=n_neighbors)
        model.fit(self.user_item_sparse)
        return model

    def _train_random_forest(self) -> RandomForestRegressor:
        X = []
        y = []

        for _, row in self.ratings.iterrows():
            mid = int(row["MovieID"])
            if mid in self.movie_index:
                X.append(self.item_features[self.movie_index[mid]])
                y.append(float(row["Rating"]))

        if not X:
            raise ValueError("RandomForest training failed: no aligned movie features were found.")

        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y, dtype=np.float32)

        model = RandomForestRegressor(
            n_estimators=100,
            max_depth=8,
            min_samples_leaf=5,
            min_samples_split=2,
            random_state=self.random_state,
            n_jobs=-1,
        )
        model.fit(X, y)
        return model

    def _clip_rating(self, value: float) -> float:
        return float(np.clip(value, 1.0, 5.0))

    def _fallback_rating(self, user_id: int | None = None, movie_id: int | None = None) -> float:
        if user_id is not None and user_id in self.user_mean_rating:
            return self._clip_rating(self.user_mean_rating[user_id])
        if movie_id is not None and movie_id in self.movie_mean_rating:
            return self._clip_rating(self.movie_mean_rating[movie_id])
        return self._clip_rating(self.global_mean_rating)

    def _scale_to_rating_scale(self, arr: np.ndarray, fallback: float | None = None) -> np.ndarray:
        arr = np.asarray(arr, dtype=float)

        if arr.size == 0 or not np.isfinite(arr).any():
            fill = self._clip_rating(fallback if fallback is not None else self.global_mean_rating)
            return np.full(arr.shape, fill, dtype=float)

        arr_min = np.nanmin(arr)
        arr_max = np.nanmax(arr)

        if not np.isfinite(arr_min) or not np.isfinite(arr_max) or abs(arr_max - arr_min) < 1e-12:
            fill = self._clip_rating(fallback if fallback is not None else self.global_mean_rating)
            return np.full(arr.shape, fill, dtype=float)

        scaled = 1.0 + (arr - arr_min) * 4.0 / (arr_max - arr_min)
        return np.clip(scaled, 1.0, 5.0)

    def _get_user_ratings_vector(self, user_id: int) -> np.ndarray:
        if user_id in self.user_item_matrix.index:
            return self.user_item_matrix.loc[user_id].values.astype(float)
        return np.zeros(len(self.movie_ids), dtype=float)

    def _get_weights_for_user(self, user_id: int) -> Tuple[float, float, float]:
        if user_id not in self.user_item_matrix.index:
            return 0.10, 0.10, 0.80

        num_rated = int((self.user_item_matrix.loc[user_id] > 0).sum())

        # RF is the strongest signal in your own results, so do not let content/CF dominate.
        if num_rated < 5:
            return 0.10, 0.10, 0.80
        if num_rated < 20:
            return 0.12, 0.18, 0.70
        return 0.10, 0.25, 0.65

    def _predict_content_vector(self, user_id: int) -> np.ndarray:
        if user_id in self._content_cache:
            return self._content_cache[user_id]

        user_ratings = self._get_user_ratings_vector(user_id)
        rated_mask = user_ratings > 0

        if not rated_mask.any():
            vec = np.full(len(self.movie_ids), self._fallback_rating(user_id=user_id), dtype=float)
            self._content_cache[user_id] = vec
            return vec

        rated_features = self.item_features[rated_mask]
        ratings = user_ratings[rated_mask]

        denom = float(ratings.sum())
        if denom <= 1e-12:
            vec = np.full(len(self.movie_ids), self._fallback_rating(user_id=user_id), dtype=float)
            self._content_cache[user_id] = vec
            return vec

        user_profile = np.average(rated_features, axis=0, weights=ratings)
        raw_scores = np.dot(self.item_features, user_profile)

        vec = self._scale_to_rating_scale(
            raw_scores,
            fallback=self.user_mean_rating.get(user_id, self.global_mean_rating),
        )
        self._content_cache[user_id] = vec
        return vec

    def _positive_mean(self, row: np.ndarray) -> float:
        positive = row[row > 0]
        if positive.size == 0:
            return self.global_mean_rating
        return float(positive.mean())

    def _predict_cf_vector(self, user_id: int) -> np.ndarray:
        if user_id in self._cf_cache:
            return self._cf_cache[user_id]

        if user_id not in self.user_item_matrix.index:
            vec = np.full(len(self.movie_ids), self._fallback_rating(user_id=user_id), dtype=float)
            self._cf_cache[user_id] = vec
            return vec

        user_ratings = self.user_item_matrix.loc[user_id].values.astype(float)
        user_mean = self._positive_mean(user_ratings)

        centered_user = (user_ratings - user_mean).reshape(1, -1)

        n_neighbors = min(self.knn.n_neighbors, self.user_item_matrix.shape[0])
        n_neighbors = max(1, n_neighbors)

        distances, indices = self.knn.kneighbors(centered_user, n_neighbors=n_neighbors)
        sims = 1.0 - distances.flatten()
        sims = np.maximum(sims, 0.0)

        neighbors = self.user_item_matrix.iloc[indices.flatten()].values.astype(float)
        neighbor_means = np.array([self._positive_mean(row) for row in neighbors], dtype=float)
        centered_neighbors = neighbors - neighbor_means[:, None]

        weighted = np.dot(sims, centered_neighbors)
        denom = float(sims.sum()) + 1e-8

        if denom <= 1e-12:
            vec = np.full(len(self.movie_ids), self._fallback_rating(user_id=user_id), dtype=float)
            self._cf_cache[user_id] = vec
            return vec

        preds = user_mean + (weighted / denom)
        preds = np.clip(preds, 1.0, 5.0)

        self._cf_cache[user_id] = preds
        return preds

    def _predict_hybrid_vector(self, user_id: int) -> np.ndarray:
        if user_id in self._hybrid_cache:
            return self._hybrid_cache[user_id]

        content_vec = self._predict_content_vector(user_id)
        cf_vec = self._predict_cf_vector(user_id)
        rf_vec = self.rf_scores

        w_content, w_cf, w_rf = self._get_weights_for_user(user_id)
        final_vec = (w_content * content_vec) + (w_cf * cf_vec) + (w_rf * rf_vec)
        final_vec = np.clip(final_vec, 1.0, 5.0)

        self._hybrid_cache[user_id] = final_vec
        return final_vec

    def predict_components(self, user_id: int, movie_id: int) -> Dict[str, float]:
        if movie_id not in self.movie_index:
            fallback = self._fallback_rating(user_id=user_id, movie_id=movie_id)
            return {
                "ContentScore": fallback,
                "CollaborativeScore": fallback,
                "ClassifierScore": fallback,
                "HybridScore": fallback,
                "WeightContent": 0.0,
                "WeightCollaborative": 0.0,
                "WeightClassifier": 1.0,
            }

        idx = self.movie_index[movie_id]

        content_vec = self._predict_content_vector(user_id)
        cf_vec = self._predict_cf_vector(user_id)
        rf_vec = self.rf_scores

        w_content, w_cf, w_rf = self._get_weights_for_user(user_id)

        return {
            "ContentScore": float(round(content_vec[idx], 4)),
            "CollaborativeScore": float(round(cf_vec[idx], 4)),
            "ClassifierScore": float(round(rf_vec[idx], 4)),
            "HybridScore": float(round(
                (w_content * content_vec[idx]) +
                (w_cf * cf_vec[idx]) +
                (w_rf * rf_vec[idx]),
                4,
            )),
            "WeightContent": float(round(w_content, 4)),
            "WeightCollaborative": float(round(w_cf, 4)),
            "WeightClassifier": float(round(w_rf, 4)),
        }

    def predict_rating(self, user_id: int, movie_id: int) -> float:
        comp = self.predict_components(user_id, movie_id)
        return self._clip_rating(comp["HybridScore"])

    def _build_explanation(self, content_score: float, cf_score: float, rf_score: float, w_content: float, w_cf: float, w_rf: float) -> str:
        strongest = max(
            [("content", content_score), ("collaborative", cf_score), ("classifier", rf_score)],
            key=lambda x: x[1],
        )[0]

        parts = []
        if strongest == "content":
            parts.append("content signal is strongest")
        elif strongest == "collaborative":
            parts.append("collaborative signal is strongest")
        else:
            parts.append("classifier signal is strongest")

        if w_rf >= max(w_content, w_cf):
            parts.append("RF is weighted highest for stability")
        elif w_cf >= max(w_content, w_rf):
            parts.append("collaborative signal dominates")
        else:
            parts.append("content signal dominates")

        return "; ".join(parts)

    def predict_for_user(self, user_id: int, top_n: int = 5) -> List[Dict[str, float | int | str]]:
        if user_id not in self.user_item_matrix.index and user_id not in self.user_mean_rating:
            return []

        user_ratings = self._get_user_ratings_vector(user_id)
        already_rated = set(np.array(self.movie_ids)[user_ratings > 0].tolist())

        content_vec = self._predict_content_vector(user_id)
        cf_vec = self._predict_cf_vector(user_id)
        rf_vec = self.rf_scores
        w_content, w_cf, w_rf = self._get_weights_for_user(user_id)

        final_scores = (w_content * content_vec) + (w_cf * cf_vec) + (w_rf * rf_vec)
        final_scores = np.clip(final_scores, 1.0, 5.0)

        recommendations = []
        for movie_id in self.movie_ids:
            if movie_id in already_rated:
                continue

            idx = self.movie_index[movie_id]
            movie_row = self.movie_lookup.loc[movie_id]

            recommendations.append(
                {
                    "MovieID": int(movie_id),
                    "Title": str(movie_row["Title"]),
                    "FinalScore": float(round(final_scores[idx], 4)),
                    "ContentScore": float(round(content_vec[idx], 4)),
                    "CollaborativeScore": float(round(cf_vec[idx], 4)),
                    "ClassifierScore": float(round(rf_vec[idx], 4)),
                    "Explanation": self._build_explanation(
                        content_vec[idx],
                        cf_vec[idx],
                        rf_vec[idx],
                        w_content,
                        w_cf,
                        w_rf,
                    ),
                }
            )

        recommendations.sort(key=lambda x: x["FinalScore"], reverse=True)
        return recommendations[:top_n]

    def evaluate(self, test_ratings: pd.DataFrame) -> Dict[str, float]:
        if test_ratings.empty:
            return {
                "Coverage": 0.0,
                "Content_RMSE": np.nan,
                "Content_MAE": np.nan,
                "Collaborative_RMSE": np.nan,
                "Collaborative_MAE": np.nan,
                "Classifier_RMSE": np.nan,
                "Classifier_MAE": np.nan,
                "Hybrid_RMSE": np.nan,
                "Hybrid_MAE": np.nan,
                "TestRows": 0,
                "EvaluatedRows": 0,
            }

        actuals = []
        content_preds = []
        cf_preds = []
        rf_preds = []
        hybrid_preds = []

        evaluated_rows = 0

        # Compute each user's vectors once, then score only the test rows for that user.
        for user_id, user_test in test_ratings.groupby("UserID", sort=False):
            user_id = int(user_id)

            content_vec = self._predict_content_vector(user_id)
            cf_vec = self._predict_cf_vector(user_id)
            rf_vec = self.rf_scores
            w_content, w_cf, w_rf = self._get_weights_for_user(user_id)
            final_vec = (w_content * content_vec) + (w_cf * cf_vec) + (w_rf * rf_vec)

            for _, row in user_test.iterrows():
                movie_id = int(row["MovieID"])
                if movie_id not in self.movie_index:
                    continue

                idx = self.movie_index[movie_id]

                actuals.append(float(row["Rating"]))
                content_preds.append(float(content_vec[idx]))
                cf_preds.append(float(cf_vec[idx]))
                rf_preds.append(float(rf_vec[idx]))
                hybrid_preds.append(float(final_vec[idx]))
                evaluated_rows += 1

        if evaluated_rows == 0:
            return {
                "Coverage": 0.0,
                "Content_RMSE": np.nan,
                "Content_MAE": np.nan,
                "Collaborative_RMSE": np.nan,
                "Collaborative_MAE": np.nan,
                "Classifier_RMSE": np.nan,
                "Classifier_MAE": np.nan,
                "Hybrid_RMSE": np.nan,
                "Hybrid_MAE": np.nan,
                "TestRows": int(len(test_ratings)),
                "EvaluatedRows": 0,
            }

        actuals_arr = np.asarray(actuals, dtype=float)

        metrics = {
            "Coverage": float(evaluated_rows / len(test_ratings)),
            "Content_RMSE": float(np.sqrt(mean_squared_error(actuals_arr, np.asarray(content_preds)))),
            "Content_MAE": float(mean_absolute_error(actuals_arr, np.asarray(content_preds))),
            "Collaborative_RMSE": float(np.sqrt(mean_squared_error(actuals_arr, np.asarray(cf_preds)))),
            "Collaborative_MAE": float(mean_absolute_error(actuals_arr, np.asarray(cf_preds))),
            "Classifier_RMSE": float(np.sqrt(mean_squared_error(actuals_arr, np.asarray(rf_preds)))),
            "Classifier_MAE": float(mean_absolute_error(actuals_arr, np.asarray(rf_preds))),
            "Hybrid_RMSE": float(np.sqrt(mean_squared_error(actuals_arr, np.asarray(hybrid_preds)))),
            "Hybrid_MAE": float(mean_absolute_error(actuals_arr, np.asarray(hybrid_preds))),
            "TestRows": int(len(test_ratings)),
            "EvaluatedRows": int(evaluated_rows),
        }
        return metrics


def load_movielens_data(csv_path: str | Path) -> Tuple[pd.DataFrame, pd.DataFrame]:
    csv_path = Path(csv_path)

    if not csv_path.exists():
        raise FileNotFoundError(f"Could not find dataset at: {csv_path}")

    data = pd.read_csv(csv_path)

    required = {"UserID", "MovieID", "Rating", "Title", "Genres"}
    missing = required - set(data.columns)
    if missing:
        raise ValueError(f"Dataset is missing required columns: {sorted(missing)}")

    movies = data[["MovieID", "Title", "Genres"]].drop_duplicates(subset="MovieID").copy()
    ratings = data[["UserID", "MovieID", "Rating"]].copy()

    movies["MovieID"] = pd.to_numeric(movies["MovieID"], errors="coerce")
    ratings["UserID"] = pd.to_numeric(ratings["UserID"], errors="coerce")
    ratings["MovieID"] = pd.to_numeric(ratings["MovieID"], errors="coerce")
    ratings["Rating"] = pd.to_numeric(ratings["Rating"], errors="coerce")

    movies = movies.dropna(subset=["MovieID"])
    ratings = ratings.dropna(subset=["UserID", "MovieID", "Rating"])

    movies["MovieID"] = movies["MovieID"].astype(int)
    ratings["UserID"] = ratings["UserID"].astype(int)
    ratings["MovieID"] = ratings["MovieID"].astype(int)
    ratings["Rating"] = ratings["Rating"].astype(float)

    movies = movies.drop_duplicates(subset="MovieID").sort_values("MovieID").reset_index(drop=True)
    ratings = ratings.drop_duplicates(subset=["UserID", "MovieID"]).reset_index(drop=True)

    return movies, ratings


def train_test_split_ratings(
    ratings: pd.DataFrame,
    test_size: float = 0.2,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if not 0.0 < test_size < 1.0:
        raise ValueError("test_size must be between 0 and 1")

    rng = np.random.default_rng(random_state)
    mask = rng.random(len(ratings)) < test_size

    test_ratings = ratings.loc[mask].copy().reset_index(drop=True)
    train_ratings = ratings.loc[~mask].copy().reset_index(drop=True)

    return train_ratings, test_ratings


def print_metrics(metrics: Dict[str, float]) -> None:
    print("\n" + "=" * 78)
    print("HYBRID RECOMMENDER EVALUATION")
    print("=" * 78)
    print(f"Test Rows      : {metrics['TestRows']}")
    print(f"Evaluated Rows : {metrics['EvaluatedRows']}")
    print(f"Coverage       : {metrics['Coverage']:.4f}")
    print("-" * 78)
    print(f"Content Based   -> RMSE: {metrics['Content_RMSE']:.4f} | MAE: {metrics['Content_MAE']:.4f}")
    print(f"Collaborative   -> RMSE: {metrics['Collaborative_RMSE']:.4f} | MAE: {metrics['Collaborative_MAE']:.4f}")
    print(f"Random Forest   -> RMSE: {metrics['Classifier_RMSE']:.4f} | MAE: {metrics['Classifier_MAE']:.4f}")
    print(f"Hybrid          -> RMSE: {metrics['Hybrid_RMSE']:.4f} | MAE: {metrics['Hybrid_MAE']:.4f}")
    print("=" * 78 + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train and evaluate the hybrid movie recommender.")
    parser.add_argument(
        "--csv",
        type=str,
        default=None,
        help="Path to movielens_100k.csv. Default is project_root/movielens_100k.csv",
    )
    parser.add_argument("--test-size", type=float, default=0.2, help="Test split ratio.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--top-n", type=int, default=5, help="Number of sample recommendations to print.")
    parser.add_argument("--sample-user", type=int, default=None, help="UserID to preview recommendations for.")
    args = parser.parse_args()

    script_dir = Path(__file__).resolve().parent
    default_csv = script_dir.parent / "movielens_100k.csv"
    csv_path = Path(args.csv) if args.csv else default_csv

    print("Loading dataset...")
    movies, ratings = load_movielens_data(csv_path)

    train_ratings, test_ratings = train_test_split_ratings(
        ratings=ratings,
        test_size=args.test_size,
        random_state=args.seed,
    )

    print(f"Loaded dataset from: {csv_path}")
    print(f"Unique users : {ratings['UserID'].nunique()}")
    print(f"Unique movies: {ratings['MovieID'].nunique()}")
    print(f"Train rows   : {len(train_ratings)}")
    print(f"Test rows    : {len(test_ratings)}")

    print("Training model...")
    model = HybridRecommender(movies_df=movies, ratings_df=train_ratings, random_state=args.seed)

    print("Evaluating...")
    metrics = model.evaluate(test_ratings)
    print_metrics(metrics)

    sample_user = args.sample_user
    if sample_user is None:
        sample_user = int(train_ratings["UserID"].iloc[0])

    print(f"Sample recommendations for UserID = {sample_user}")
    recs = model.predict_for_user(sample_user, top_n=args.top_n)

    if not recs:
        print("No recommendations available for this user.")
        return

    for i, rec in enumerate(recs, start=1):
        print(
            f"{i}. {rec['Title']} | "
            f"Final: {rec['FinalScore']:.4f} | "
            f"CBF: {rec['ContentScore']:.4f} | "
            f"CF: {rec['CollaborativeScore']:.4f} | "
            f"RF: {rec['ClassifierScore']:.4f} | "
            f"{rec['Explanation']}"
        )


if __name__ == "__main__":
    main()
