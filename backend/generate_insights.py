# backend/generate_insights.py
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset
df = pd.read_csv(r"E:/PROGRAMMING-2/Movie-Recommendation-System/movielens_100k.csv", encoding="latin-1")
df = df[['UserID', 'MovieID', 'Rating', 'Title', 'Genres']]

# Create assets directory if not exists
import os
assets_path = r"E:/PROGRAMMING-2/Movie-Recommendation-System/frontend/assets"
os.makedirs(assets_path, exist_ok=True)

# ---- Rating distribution ----
plt.figure(figsize=(6,4))
sns.histplot(data=df, x='Rating', bins=10, color='goldenrod', edgecolor='black')
plt.title('Rating Distribution', fontsize=13, fontweight='bold')
plt.xlabel('Rating')
plt.ylabel('Count')
plt.tight_layout()
plt.savefig(os.path.join(assets_path, "rating_dist.png"), dpi=150)
plt.close()

# ---- Genre popularity ----
g = df.assign(Genres=df['Genres'].fillna('').str.split('|')).explode('Genres')
counts = g[g['Genres'] != '']['Genres'].value_counts().head(10)
plt.figure(figsize=(7,5))
counts.sort_values().plot(kind='barh', color='#219EBC', edgecolor='black')
plt.title('Top 10 Genres (by frequency)', fontsize=13, fontweight='bold')
plt.xlabel('Count')
plt.ylabel('Genre')
plt.tight_layout()
plt.savefig(os.path.join(assets_path, "genre_popularity.png"), dpi=150)
plt.close()

print("Insight images generated successfully in /frontend/assets/")
