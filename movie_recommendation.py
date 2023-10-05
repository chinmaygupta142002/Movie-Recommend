import ast
import pandas as pd
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def convert_genres_keywords(obj):
    arr = []
    for i in ast.literal_eval(obj):
        arr.append(i["name"].replace(" ", "").lower())
    return " ".join(arr)


def convert_cast(obj):
    arr = []
    counter = 0
    for i in ast.literal_eval(obj):
        arr.append(i["name"].replace(" ", "").lower())
        counter += 1
        if counter == 3:
            break
    return " ".join(arr)


def convert_crew(obj):
    arr = []
    for i in ast.literal_eval(obj):
        if i["job"] == "Director":
            arr.append(i["name"].replace(" ", "").lower())
    return " ".join(arr)


pd.set_option("display.max_columns", None)
movies = pd.read_csv("tmdb_5000_movies.csv")
cast = pd.read_csv("tmdb_5000_credits.csv")
df = pd.merge(movies, cast)
df = df[["movie_id", "title", "cast", "crew", "keywords", "overview", "genres"]]
df.dropna(inplace=True)
df["genres"] = df["genres"].apply(convert_genres_keywords)
df["keywords"] = df["keywords"].apply(convert_genres_keywords)
df["overview"] = df["overview"].apply(lambda x: x.lower())
df["cast"] = df["cast"].apply(convert_cast)
df["crew"] = df["crew"].apply(convert_crew)
df["keywords"] = df["genres"]+df["keywords"]+df["overview"]+df["cast"]+df["crew"]
ps = PorterStemmer()


def stem_words(text):
    arr = []
    for i in text.split(" "):
        arr.append(ps.stem(i))
    return " ".join(arr)


df["keywords"] = df["keywords"].apply(stem_words)
cv = CountVectorizer(stop_words="english", max_features=5000)
vectors = cv.fit_transform(df["keywords"])
similarity = cosine_similarity(vectors)


def recommend_movie():
    movie = input("Enter Movie: ")
    movie_index = df[df["title"] == movie].index[0]
    distances = similarity[movie_index]
    movies_list = sorted(list(enumerate(distances)), reverse=True, key=lambda x: x[1])[1: 6]

    for i in movies_list:
        print(df.iloc[i[0]].title)


recommend_movie()










