# backend/hybrid.py  — memory-safe (sparse) hybrid
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import TruncatedSVD
from sklearn.neighbors import NearestNeighbors
from sklearn.ensemble import RandomForestClassifier
from sklearn.multiclass import OneVsRestClassifier
from scipy.sparse import csr_matrix, issparse

class HybridRecommender:
    def __init__(self, movies_df, ratings_df):
        """
        movies_df : ['MovieID','Title','Genres'] (unique rows)
        ratings_df: ['UserID','MovieID','Rating']
        """
        # Keep minimal copies
        self.movies  = movies_df[['MovieID','Title','Genres']].drop_duplicates('MovieID').copy()
        self.ratings = ratings_df[['UserID','MovieID','Rating']].copy()

        # Build sparse user-item and id<->index maps
        self.user_item_csr, self.user_ids, self.movie_ids, self.uid2idx, self.mid2idx = \
            self._build_user_item_sparse(self.ratings)

        # Content features (dense but compact: n_movies x 200 float32)
        self.item_features = self._compute_item_features(self.movies, self.movie_ids)

        # Optional (dense) content similarity in float32 (n_movies x n_movies).
        # For large n_movies this can be big; set BUILD_CONTENT_SIM=False to compute on-demand instead.
        self.BUILD_CONTENT_SIM = True
        self.content_sim = self._compute_content_similarity(self.item_features) if self.BUILD_CONTENT_SIM else None

        # KNN on USERS (CSR works) for user-based CF
        self.knn_model = self._train_knn_user_based(self.user_item_csr)

        # Multi-label RF over genres (probabilities averaged to a single score)
        self.rf_model = self._train_multilabel_rf(self.item_features, self.movies.loc[
            self.movies['MovieID'].isin(self.movie_ids)
        ])
        self.movies_idx = self.movies.set_index('MovieID')  # for title lookup

    # Sparse user-item 
    def _build_user_item_sparse(self, ratings):
        # Categorical encode to compact integer indices
        ucat = pd.Categorical(ratings['UserID'])
        mcat = pd.Categorical(ratings['MovieID'])
        rows = ucat.codes.astype(np.int32)
        cols = mcat.codes.astype(np.int32)
        vals = ratings['Rating'].astype(np.float32).values

        n_users = len(ucat.categories)
        n_items = len(mcat.categories)

        csr = csr_matrix((vals, (rows, cols)), shape=(n_users, n_items), dtype=np.float32)

        user_ids = pd.Index(ucat.categories.astype(int))
        movie_ids = pd.Index(mcat.categories.astype(int))
        uid2idx = {int(u): i for i, u in enumerate(user_ids)}
        mid2idx = {int(m): i for i, m in enumerate(movie_ids)}
        return csr, user_ids, movie_ids, uid2idx, mid2idx

    # Content (TF-IDF + SVD) 
    def _compute_item_features(self, movies, movie_ids):
        m = movies[movies['MovieID'].isin(movie_ids)].copy()
        m['Title']  = m['Title'].fillna('')
        m['Genres'] = m['Genres'].fillna('')
        m['content'] = m['Title'] + ' ' + m['Genres'].str.replace('|',' ', regex=False)

        tfidf = TfidfVectorizer(stop_words='english', max_features=5000)
        X = tfidf.fit_transform(m['content'])                   # sparse
        svd = TruncatedSVD(n_components=200, random_state=42)
        R = svd.fit_transform(X).astype(np.float32)             # dense but compact
        feats = pd.DataFrame(R, index=m['MovieID'])
        # Reindex to full movie_ids to keep strict alignment
        return feats.reindex(movie_ids).astype(np.float32)

    def _compute_content_similarity(self, item_features_df):
        # cosine on reduced features; cast to float32 to halve memory
        sims = cosine_similarity(item_features_df.values, item_features_df.values).astype(np.float32)
        return sims  # numpy array [n_items, n_items]

    # KNN (user-based) 
    def _train_knn_user_based(self, user_item_csr):
        model = NearestNeighbors(metric='cosine', algorithm='brute', n_neighbors=5)
        model.fit(user_item_csr)  # CSR supported
        return model

    # Multi-label RF over genres 
    def _train_multilabel_rf(self, item_features_df, movies_sub):
        all_genres = sorted(set('|'.join(movies_sub['Genres'].fillna('')).split('|')) - {''})
        Y = pd.DataFrame(0, index=movies_sub['MovieID'], columns=all_genres)
        for _, row in movies_sub.iterrows():
            for g in row['Genres'].split('|'):
                if g: Y.at[row['MovieID'], g] = 1

        X = item_features_df.reindex(Y.index).values
        clf = OneVsRestClassifier(RandomForestClassifier(
            n_estimators=120, max_depth=14, n_jobs=-1, random_state=42
        ))
        clf.fit(X, Y.values.astype(np.int32))
        return clf

    # Public API 
    def predict_for_user(self, user_id, top_n=5, w_content=0.4, w_collab=0.4, w_rf=0.2):
        if user_id not in self.uid2idx:
            return []

        uidx = self.uid2idx[user_id]

        # 1) Content score = sim * ratings
        # user ratings vector (1 x n_items), dense float32
        u_vec = self.user_item_csr.getrow(uidx).toarray().astype(np.float32).ravel()

        if self.content_sim is not None:
            content_scores = self.content_sim @ u_vec  # (n_items,)
        else:
            # On-demand content score: sum over rated items of sim(i, j)*r_ij
            rated_idx = np.where(u_vec > 0)[0]
            content_scores = np.zeros(len(self.movie_ids), dtype=np.float32)
            if rated_idx.size:
                sims = cosine_similarity(
                    self.item_features.values,               # all items (n x d)
                    self.item_features.values[rated_idx]     # only rated items (k x d)
                ).astype(np.float32)                          # (n x k)
                content_scores = sims @ u_vec[rated_idx]      # (n,)

        # 2) Collaborative score = mean neighbor ratings
        distances, nbr_users = self.knn_model.kneighbors(self.user_item_csr.getrow(uidx), n_neighbors=5)
        nbr_block = self.user_item_csr[nbr_users[0]]              # (k x n_items) CSR
        collab_scores = np.asarray(nbr_block.mean(axis=0)).ravel().astype(np.float32)

        # 3) RF genre probability score (avg across genres)
        X_items = self.item_features.values
        proba_list = self.rf_model.predict_proba(X_items)
        if isinstance(proba_list, list):
            pos_cols = [p[:, 1] for p in proba_list]              # take positive class
            proba = np.stack(pos_cols, axis=1)                    # (n_items, n_genres)
        else:
            proba = proba_list                                    # already 2D
        rf_scores = proba.mean(axis=1).astype(np.float32)

        # 4) Weighted fusion
        fused = (w_content * content_scores) + (w_collab * collab_scores) + (w_rf * rf_scores)

        # Exclude already-rated items
        rated_mask = u_vec > 0
        fused[rated_mask] = -np.inf

        # Top-N
        top_idx = np.argpartition(fused, -top_n)[-top_n:]
        top_idx = top_idx[np.argsort(fused[top_idx])[::-1]]

        results = []
        for mi in top_idx:
            mid = int(self.movie_ids[mi])
            title = self.movies_idx.at[mid, 'Title'] if mid in self.movies_idx.index else str(mid)
            results.append({
                "MovieID": mid,
                "Title": title,
                "Score": float(round(fused[mi], 3))
            })
        return results
