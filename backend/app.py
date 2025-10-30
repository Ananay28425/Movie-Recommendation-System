from flask import Flask, render_template, request, jsonify
import pandas as pd
from hybrid import HybridRecommender

# Load data
data = pd.read_csv('../movielens_100k.csv')  # adjust path if needed
ratings = data[['UserID','MovieID','Rating']]
movies = data[['MovieID','Title','Genres']].drop_duplicates(subset='MovieID')

# Initialize hybrid model
hybrid_model = HybridRecommender(movies, ratings)

app = Flask(__name__)

@app.route('/')
def home():
    users = sorted(ratings['UserID'].unique())
    return render_template('index.html', users=users)

@app.route('/recommend', methods=['POST'])
def recommend():
    try:
        user_id = int(request.form.get('user_id'))
        recommendations = hybrid_model.predict_for_user(user_id)
        return jsonify(recommendations)
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)
