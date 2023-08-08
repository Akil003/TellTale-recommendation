import pandas as pd
from flask import Flask, request
from sklearn.metrics.pairwise import cosine_similarity
from flask_cors import CORS, cross_origin

ids = pd.read_json('./ids.json')
feature_matrix = pd.read_json('./feature_matrix.json')

app = Flask(__name__)
cors = CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'

@app.route('/')
def hello():
    given_movie_id = int(request.args.get('id', '57'))
    # Find the index of the given movie ID in the 'movies_data' DataFrame
    given_movie_index = ids[ids["_id"] == given_movie_id].index[0]

    # Extract the feature vector for the given movie from the 'feature_matrix'
    feature_vector_for_given_movie = feature_matrix.iloc[given_movie_index]
    # Calculate the similarity between the feature vector of the given movie and all other movies in the dataset
    movie_similarity_scores = cosine_similarity([feature_vector_for_given_movie], feature_matrix)[0]

    # Sort movies based on similarity scores (higher similarity means more similar)
    # Exclude the given movie itself from the similar movies
    num_similar_movies = 40
    top_similar_indices = movie_similarity_scores.argsort()[:-num_similar_movies-1:-1]

    # Get the IDs of the similar movies
    similar_movie_ids = ids.iloc[top_similar_indices]["_id"][1:].tolist()
    return similar_movie_ids

if __name__ == '__main__':
    app.run()
