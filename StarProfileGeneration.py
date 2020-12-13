import pandas as pd
import numpy as np
import json
import nltk
import re
import csv
import matplotlib.pyplot as plt 
import seaborn as sns
from tqdm import tqdm
import sklearn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split

genre_indices = {}
with open("GenreIndices.txt", "r") as reader:
    line = reader.readline()
    line = line.split(";")
    for genre in line:
        genre_split = genre.split("_")
        if (len(genre_split) > 1):
            genre_indices[genre_split[0]] = int(genre_split[1])


# REPEATED LOGIC TO STORE MOVIE INFO (some unnecessary attributes, such as plots)
pd.set_option('display.max_colwidth', 300)
meta = pd.read_csv("movie.metadata.tsv", sep = '\t', header = None)
# Added for actor name and movie correlation
char_meta = pd.read_csv("character.metadata.tsv", sep='\t', header=None)
meta.columns = ["movie_id", 1, "movie_name", 3, 4, 5, 6, 7, "genre"]
char_meta.columns = ["movie_id", 1, 2, 3, 4, 5, 6, 7, "actor_name", 9, 10, 11, 12]

plots = []
with open("plot_summaries.txt", 'r') as f:
    reader = csv.reader(f, dialect='excel-tab')
    for row in tqdm(reader):
        plots.append(row)

movie_id = []
plot = []

for i in tqdm(plots):
    movie_id.append(i[0])
    plot.append(i[1])

movies = pd.DataFrame({'movie_id': movie_id, 'plot': plot})
meta['movie_id'] = meta['movie_id'].astype(str)
char_meta['movie_id'] = char_meta['movie_id'].astype(str)

movies = pd.merge(movies, meta[['movie_id', 'movie_name', 'genre']], on='movie_id')

genres = []
for i in movies['genre']:
    genres.append(list(json.loads(i).values()))

movies['genre_new'] = genres
movies_new = movies[~(movies['genre_new'].str.len() == 0)]
# END REPEATED LOGIC


# Dictionary Prep for Acessing Movie Genres / Names
id_genres = {}
id_names = {}
new_movie_ids = []
new_movie_genres = []
new_movie_names = []
for s in movies_new['movie_id']:
    new_movie_ids.append(s)

for a in movies_new['movie_name']:
    new_movie_names.append(a)

for g in movies_new['genre_new']:
    new_movie_genres.append(g)

count = 0
for x in new_movie_ids:
    id_genres[x] = new_movie_genres[count]
    id_names[x] = new_movie_names[count]
    count += 1


# The star's profile vector
target_vector = [0] * 363

count = 0
movie_count = 0
counted = {}
for y in char_meta['actor_name']:
    if (y == "Bruce Willis"):
        movie_ref = char_meta['movie_id'][count]
        # Some movies, like animated, have stars playing multiple roles, so check in counted
        if (movie_ref in id_genres and movie_ref not in counted):
            movie_genres = id_genres[movie_ref]
            counted[movie_ref] = True
            for movie_g in movie_genres:
                vector_index = genre_indices[movie_g]
                target_vector[vector_index] += 1
            movie_count += 1
    count += 1


for i in range(len(target_vector)):
    target_vector[i] /= movie_count

# Movie vector for comparison
target_movie_vector = [0] * 363

for m in movies_new['movie_id']:
    if (id_names[m] == "Demolition Man"):
        for g in id_genres[m]:
            v_index = genre_indices[g]
            target_movie_vector[v_index] += 1
        break

print("Bruce Willis vs. Demolition Man")
print(sklearn.metrics.pairwise.cosine_similarity([target_movie_vector], [target_vector]))
print(target_movie_vector)


