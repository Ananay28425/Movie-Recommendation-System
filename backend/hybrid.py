import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import TruncatedSVD
from sklearn.neighbors import NearestNeighbors
from sklearn.ensemble import RandomForestClassifier
from scipy.sparse import csr_matrix

class HybridRecommender:
    def __init__(self, movies_df, ratings_df):
        """
        movies_df : pd.DataFrame with ['MovieID', 'Title', 'Genres']
        ratings_df : pd.DataFrame with ['UserID', 'MovieID', 'Rating']
        """
        self.movies = movies_df
        self.ratings = ratings_df
        self.user_ids = ratings_df['UserID'].unique()
        self.movie_ids = movies_df['MovieID'].unique()
        self.user_item_matrix = self._create_user_item_matrix()
        self.item_features = self._compute_item_features()
        self.content_sim_matrix = self._compute_content_similarity()
        self.knn_model = self._train_knn()
        self.rf_model, self.rf_features = self._train_random_forest()

    def _create_user_item_matrix(self):
        return self.ratings.pivot_table(index='UserID', columns='MovieID', values='Rating').fillna(0)

    def _compute_item_features(self):
        # TF-IDF encoding of "Title + Genres"
        self.movies['content'] = self.movies['Title'] + ' ' + self.movies['Genres'].str.replace('|', ' ')
        tfidf = TfidfVectorizer(stop_words='english', max_features=5000)
        tfidf_matrix = tfidf.fit_transform(self.movies['content'])
        # Dimensionality reduction
        svd = TruncatedSVD(n_components=200, random_state=42)
        reduced_matrix = svd.fit_transform(tfidf_matrix)
        return pd.DataFrame(reduced_matrix, index=self.movies['MovieID'])

    def _compute_content_similarity(self):
        return pd.DataFrame(cosine_similarity(self.item_features), 
                            index=self.item_features.index, 
                            columns=self.item_features.index)

    def _train_knn(self):
        model = NearestNeighbors(metric='cosine', algorithm='brute', n_neighbors=5)
        model.fit(self.user_item_matrix.values)
        return model

    def _train_random_forest(self):
        # Multi-label genre prediction
        # One-hot encode genres
        genre_cols = list(set('|'.join(self.movies['Genres']).split('|')))
        genre_df = pd.DataFrame(0, index=self.movies['MovieID'], columns=genre_cols)
        for idx, row in self.movies.iterrows():
            for g in row['Genres'].split('|'):
                genre_df.loc[row['MovieID'], g] = 1
        
        rf = RandomForestClassifier(n_estimators=150, max_depth=15, random_state=42)
        rf.fit(self.item_features.values, genre_df.values)
        return rf, genre_df.columns.tolist()

    def predict_for_user(self, user_id, top_n=5, w_content=0.4, w_collab=0.4, w_rf=0.2):
        if user_id not in self.user_ids:
            return []

        # Content-based scores
        user_ratings = self.user_item_matrix.loc[user_id]
        content_scores = self.content_sim_matrix.dot(user_ratings)
        
        # Collaborative KNN scores
        user_vec = self.user_item_matrix.loc[user_id].values.reshape(1, -1)
        distances, neighbors_idx = self.knn_model.kneighbors(user_vec)
        neighbor_ratings = self.user_item_matrix.iloc[neighbors_idx[0]]
        collab_scores = neighbor_ratings.mean(axis=0)

        # RandomForest scores
        rf_probs = self.rf_model.predict_proba(self.item_features.values)
        classifier_scores = np.array([prob.mean() for prob in rf_probs])
        classifier_scores = pd.Series(classifier_scores, index=self.item_features.index)

        # Weighted fusion
        final_scores = w_content*content_scores + w_collab*collab_scores + w_rf*classifier_scores
        final_scores = final_scores.drop(user_ratings[user_ratings>0].index)  # exclude already rated movies

        top_movies = final_scores.sort_values(ascending=False).head(top_n)
        recommendations = []
        for mid, score in top_movies.items():
            title = self.movies.loc[self.movies['MovieID']==mid, 'Title'].values[0]
            recommendations.append({'MovieID': mid, 'Title': title, 'Score': round(score, 3)})
        return recommendations
