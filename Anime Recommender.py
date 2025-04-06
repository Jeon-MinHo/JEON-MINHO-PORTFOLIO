#this is a simple anime recommender that I have trialed by using cosine similarity to find an anime recommendation based on the genres of the three previous anime being inputted.



import pandas as pd
import numpy as np
from numpy.linalg import norm

def load_anime_data(path="/Users/jeonminho/Code/anime-dataset-2023.csv"):
    df = pd.read_csv(path)
    all_genres = sorted(set('|'.join(df['Genres']).split('|')))
    
    def encode_genres(genres_str):
        genres = genres_str.split('|')
        return np.array([1 if g in genres else 0 for g in all_genres])
    
    df['vector'] = df['Genres'].apply(encode_genres)
    return df

def cosine_similarity(a, b):
    return np.dot(a, b) / (norm(a) * norm(b))

def recommend_anime(user_titles, top_k=5):
    df = load_anime_data()

    selected_vectors = []
    for title in user_titles:
        match = df[df['English name'].str.lower() == title.lower()]
        if not match.empty:
            selected_vectors.append(match.iloc[0]['vector'])

    if not selected_vectors:
        return ["No matches found. Check your input?"]

    user_vector = np.mean(selected_vectors, axis=0)

    scores = []
    for _, row in df.iterrows():
        if row['English name'] not in user_titles:
            score = cosine_similarity(user_vector, row['vector'])
            scores.append((row['English name'], score))

    scores.sort(key=lambda x: x[1], reverse=True)
    return [title for title, _ in scores[:top_k]]
