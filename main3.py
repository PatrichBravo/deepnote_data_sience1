import pandas
from surprise import Dataset, Reader
from surprise import SVD

ratings = pandas.read_csv("ratings.csv")[["userId","movieId","rating"]]
ra = ratings.head()

print(ra)
reader = Reader(rating_scale=(1, 5))
dataset = Dataset.load_from_df(ratings, reader)
print(dataset)

trainset = dataset.build_full_trainset()
trainset1 = list(trainset.all_ratings())
                 
                 
print(trainset1)



svd = SVD()
svd.fit(trainset)

print(svd.predict(15, 1956))


from surprise import model_selection
model_validation = model_selection.cross_validate(svd, dataset, measures=['RMSE', 'MAE'])

print(model_validation)