import pandas as pd
import numpy as np
import json
import nltk
import re
import csv
import matplotlib.pyplot as plt 
import seaborn as sns
from tqdm import tqdm
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split

# THIS FILE WAS SOLEY FOR GENERATING GENRE INDICES (STORED IN FILE) FOR CONSISTENCY AMONG VECTORS

pd.set_option('display.max_colwidth', 300)
meta = pd.read_csv("movie.metadata.tsv", sep = '\t', header = None)
meta.columns = ["movie_id", 1, "movie_name", 3, 4, 5, 6, 7, "genre"]
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
movies = pd.merge(movies, meta[['movie_id', 'movie_name', 'genre']], on='movie_id')

genres = []
for i in movies['genre']:
    genres.append(list(json.loads(i).values()))

movies['genre_new'] = genres
movies_new = movies[~(movies['genre_new'].str.len() == 0)]
all_genres = sum(genres, [])
set_genres = set(all_genres)

with open("GenreIndices.txt", "w") as writer:
    count = 0
    for genre in set_genres:
        input_string = genre + "_" + str(count) + ";"
        writer.write(input_string)
        count += 1
