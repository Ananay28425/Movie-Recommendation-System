# backend/hybrid.py
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import TruncatedSVD
from sklearn.neighbors import NearestNeighbors
from sklearn.ensemble import RandomForestClassifier
from sklearn.multiclass import OneVsRestClassifier
from scipy.sparse import csr_matrix

class HybridRecommender:
    def __init__(self, movies_df, ratings_df):
        """
        movies_df : DataFrame ['MovieID','Title','Genres']
        ratings_df: DataFrame ['UserID','MovieID','Rating']
        """
        self.movies  = movies_df[['MovieID','Title','Genres']].drop_duplicates('MovieID').copy()
        self.ratings = ratings_df[['UserID','MovieID','Rating']].copy()

        self.user_item_csr, self.user_ids, self.movie_ids, self.uid2idx, self.mid2idx = self._build_user_item(self.ratings)
        self.item_features = self._build_item_features(self.movies, self.movie_ids)
        self.content_sim   = self._build_content_sim(self.item_features)
        self.knn_user      = self._build_knn_user(self.user_item_csr)
        self.rf_ovr        = self._build_multilabel_rf(self.item_features, self.movies[self.movies['MovieID'].isin(self.movie_ids)])
        self.movies_idx    = self.movies.set_index('MovieID')

    # Build user-item matrix
    def _build_user_item(self, ratings):
        ucat = pd.Categorical(ratings['UserID'])
        mcat = pd.Categorical(ratings['MovieID'])
        rows = ucat.codes.astype(np.int32)
        cols = mcat.codes.astype(np.int32)
        vals = ratings['Rating'].astype(np.float32).values
        csr = csr_matrix((vals, (rows, cols)), shape=(len(ucat.categories), len(mcat.categories)), dtype=np.float32)
        user_ids = pd.Index(ucat.categories.astype(int))
        movie_ids = pd.Index(mcat.categories.astype(int))
        uid2idx = {int(u): i for i, u in enumerate(user_ids)}
        mid2idx = {int(m): i for i, m in enumerate(movie_ids)}
        return csr, user_ids, movie_ids, uid2idx, mid2idx

    # Building Item features(content features + similarity)
    def _build_item_features(self, movies, movie_ids):
        m = movies[movies['MovieID'].isin(movie_ids)].copy()
        m['Title'] = m['Title'].fillna('')
        m['Genres'] = m['Genres'].fillna('')
        m['content'] = m['Title'] + ' ' + m['Genres'].str.replace('|', ' ', regex=False)
        tfidf = TfidfVectorizer(stop_words='english', max_features=4000)
        X = tfidf.fit_transform(m['content'])
        svd = TruncatedSVD(n_components=200, random_state=42)
        R = svd.fit_transform(X).astype(np.float32)
        feats = pd.DataFrame(R, index=m['MovieID'])
        return feats.reindex(movie_ids).astype(np.float32)

    # Build content similarity matrix
    def _build_content_sim(self, item_feats_df):
        sims = cosine_similarity(item_feats_df.values)
        return sims.astype(np.float32)

    # user-based KNN 
    def _build_knn_user(self, csr_mat):
        knn = NearestNeighbors(metric='cosine', algorithm='brute', n_neighbors=5)
        knn.fit(csr_mat)
        return knn

    # multilabel RF on genres (content confidence) 
    def _build_multilabel_rf(self, item_feats_df, movies_sub):
        all_genres = sorted(set('|'.join(movies_sub['Genres'].fillna('')).split('|')) - {''})
        Y = pd.DataFrame(0, index=movies_sub['MovieID'], columns=all_genres)
        for _, row in movies_sub.iterrows():
            for g in str(row['Genres']).split('|'):
                if g:
                    Y.at[row['MovieID'], g] = 1
        X = item_feats_df.reindex(Y.index).values
        rf = RandomForestClassifier(n_estimators=120, max_depth=14, n_jobs=-1, random_state=42) 
        # rf = RandomForestClassifier(n_estimators=120, max_depth=14, n_jobs=-1, random_state=42)
        ovr = OneVsRestClassifier(rf)
        ovr.fit(X, Y.values.astype(np.int32))
        self._genre_columns = list(Y.columns)
        return ovr

    # RF Probablity
    def _rf_prob(self, movie_idx):
        proba_list = self.rf_ovr.predict_proba(self.item_features.values[[movie_idx]])
        if isinstance(proba_list, list):
            pos_cols = [p[:, 1] for p in proba_list]
            arr = np.stack(pos_cols, axis=1)
        else:
            arr = proba_list
        return float(arr.mean())

    # Collaborative Filtering Score
    def _cf_score(self, user_idx, movie_idx):
        _, nbr_idx = self.knn_user.kneighbors(self.user_item_csr[user_idx], n_neighbors=5)
        nbr_block = self.user_item_csr[nbr_idx[0]]
        return float(np.asarray(nbr_block.mean(axis=0))[0, movie_idx])
    
    
    # Similarity to top liked movies
    def _top_similar_liked(self, movie_idx, liked_indices):
        if len(liked_indices) == 0:
            return None, 0.0
        sims = self.content_sim[movie_idx, liked_indices]
        j = int(np.argmax(sims))
        return int(liked_indices[j]), float(sims[j])

    # Top genres of a movie
    def _genres(self, movie_id, top_k=2):
        gs = str(self.movies_idx.at[movie_id, 'Genres']).split('|')
        return [g for g in gs if g][:top_k] or ['General']

    # Prediction for a user
    def predict_for_user(self, user_id, top_n=5, w_content=0.4, w_collab=0.4, w_rf=0.2,
                         session_likes=None, session_dislikes=None):
        if user_id not in self.uid2idx:
            return []

        session_likes    = session_likes or []
        session_dislikes = session_dislikes or []

        uidx = self.uid2idx[user_id]
        u_vec = self.user_item_csr.getrow(uidx).toarray().astype(np.float32).ravel()

        # 1) content score
        content_scores = self.content_sim @ u_vec

        # 2) CF score (user-based neighbors)
        _, nbr_idx = self.knn_user.kneighbors(self.user_item_csr[uidx], n_neighbors=5)
        nbr_block = self.user_item_csr[nbr_idx[0]]
        collab_scores = np.asarray(nbr_block.mean(axis=0)).ravel().astype(np.float32)

        # 3) RF score
        proba_list = self.rf_ovr.predict_proba(self.item_features.values)
        if isinstance(proba_list, list):
            pos_cols = [p[:, 1] for p in proba_list]
            proba = np.stack(pos_cols, axis=1)
        else:
            proba = proba_list
        rf_scores = proba.mean(axis=1).astype(np.float32)

        fused = (w_content*content_scores + w_collab*collab_scores + w_rf*rf_scores)

        # session nudge
        like_idxs    = np.array([self.mid2idx[m] for m in session_likes if m in self.mid2idx], dtype=np.int32)
        dislike_idxs = np.array([self.mid2idx[m] for m in session_dislikes if m in self.mid2idx], dtype=np.int32)

        if like_idxs.size:
            bonus = self.content_sim[:, like_idxs].max(axis=1)
            fused += 0.4 * bonus
        if dislike_idxs.size:
            penalty = self.content_sim[:, dislike_idxs].max(axis=1)
            fused -= 0.4 * penalty

        rated_mask = u_vec > 0
        fused[rated_mask] = -np.inf

        top_idx = np.argpartition(fused, -top_n)[-top_n:]
        top_idx = top_idx[np.argsort(fused[top_idx])[::-1]]

        out = []
        for mi in top_idx:
            mid = int(self.movie_ids[mi])
            title = self.movies_idx.at[mid, 'Title'] if mid in self.movies_idx.index else str(mid)
            out.append({"MovieID": mid, "Title": title, "Score": float(round(fused[mi], 3))})
        return out
    
    # Explanation for a recommendation
    def explain(self, user_id, movie_id, session_likes=None):
        session_likes = session_likes or []
        if user_id not in self.uid2idx or movie_id not in self.mid2idx:
            return {"prob": 0.0, "cf_score": 0.0, "top_genres": [], "similar_title": "N/A", "similarity": 0.0}

        uidx = self.uid2idx[user_id]
        midx = self.mid2idx[movie_id]
        prob = self._rf_prob(midx)
        cf   = self._cf_score(uidx, midx)

        liked_train = self.ratings[(self.ratings['UserID']==user_id) & (self.ratings['Rating']>=4)]['MovieID'].tolist()
        liked_all = list({*liked_train, *(session_likes or [])})
        liked_idx = [self.mid2idx[m] for m in liked_all if m in self.mid2idx]
        sim_idx, sim_val = self._top_similar_liked(midx, np.array(liked_idx, dtype=np.int32))
        sim_title = self.movies_idx.at[int(self.movie_ids[sim_idx]), 'Title'] if sim_idx is not None else "your liked titles"

        return {
            "prob": prob,
            "cf_score": cf,
            "top_genres": self._genres(movie_id, top_k=2),
            "similar_title": sim_title,
            "similarity": sim_val
        }
