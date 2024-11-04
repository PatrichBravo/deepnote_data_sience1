import pandas

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

movies = pandas.read_csv("movies_small.csv", sep=';')

tfidf = TfidfVectorizer(stop_words="english")
tfidf_matrix = tfidf.fit_transform(movies['overview'])
pandas.DataFrame(tfidf_matrix.toarray(), columns=tfidf.get_feature_names())



movies['overview'] = movies['overview'].fillna('')


similarity_matrix = linear_kernel(tfidf_matrix, tfidf_matrix)



def similar_movies(movie_title, nr_movies):
    idx = movies.loc[movies["title"] == movie_title].index[0]
    scores = list(enumerate(similarity_matrix[idx]))
    scores = sorted(scores, key=lambda x: x[1], reverse=True)
    movies_index = [tlp[0] for tlp in scores[1:nr_movies+1]]
    similar_titles = list(movies["title"].iloc[movies_index])
    return similar_titles