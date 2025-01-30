# -*- coding: utf-8 -*-
# Neural Collaborative Filtering on MIND Dataset.

# Commented out IPython magic to ensure Python compatibility.
# %load_ext autoreload
# %autoreload 2
import warnings
warnings.filterwarnings('ignore')

import sys
import pandas as pd
import tensorflow as tf
tf.get_logger().setLevel('ERROR') # only show error messages

from recommenders.utils.timer import Timer
from recommenders.models.ncf.ncf_singlenode import NCF
from recommenders.models.ncf.dataset import Dataset as NCFDataset
from recommenders.datasets import movielens
# from recommenders.datasets.python_splitters import python_chrono_split
from recommenders.evaluation.python_evaluation import (
    map, ndcg_at_k, precision_at_k, recall_at_k
)
from recommenders.utils.notebook_utils import store_metadata

print("System version: {}".format(sys.version))
print("Pandas version: {}".format(pd.__version__))
print("Tensorflow version: {}".format(tf.__version__))

"""Set the default parameters."""

# Model parameters
EPOCHS = 50
BATCH_SIZE = 256

SEED = 42

"""### 1. Load the MIND dataset"""

behaviors_df = pd.read_csv("data/MIND/MINDsmall_train/behaviors.tsv", sep="\t",
                           names=["impression_id", "user_id", "timestamp", "history", "impressions"])

# little preprocessing to get the interactions
def process_impressions(row):
    impressions = row["impressions"].split(" ")
    return [(row["user_id"], imp.split("-")[0], int(imp.split("-")[1]), row["timestamp"])
            for imp in impressions]

interactions = []
for _, row in behaviors_df.iterrows():
    interactions.extend(process_impressions(row))

interactions_df = pd.DataFrame(interactions, columns=["userID", "itemID", "rating", "timestamp"])

news_df = pd.read_csv("data/MIND/MINDsmall_train/news.tsv", sep="\t",
                      names=["news_id", "category", "subcategory", "title", "abstract", "url", "entity_list", "relation_list"])

interactions_df = interactions_df[interactions_df["itemID"].isin(news_df["news_id"])]

# convert to csv
interactions_df.to_csv("data/MIND/mind_interactions_train.csv", index=False)

"""### 2. Split the data using the Spark chronological splitter provided in utilities"""

## -*- not required in our dataset -*
# train, test = python_chrono_split(df, 0.75)

"""Write datasets to csv files."""

train_file = "data/MIND/mind_interactions_train.csv"
# train.to_csv(train_file, index=False)
# test_file = "./test.csv"
# test.to_csv(test_file, index=False)

"""Generate an NCF dataset object from the data subsets."""

data = NCFDataset(train_file=train_file, seed=SEED)

"""### 3. Train the NCF model on the training data, and get the top-k recommendations for our testing data

NCF accepts implicit feedback and generates prospensity of items to be recommended to users in the scale of 0 to 1. A recommended item list can then be generated based on the scores. Note that this quickstart notebook is using a smaller number of epochs to reduce time for training. As a consequence, the model performance will be slighlty deteriorated.
"""

model = NCF (
    n_users=data.n_users,
    n_items=data.n_items,
    model_type="NeuMF",
    n_factors=4,
    layer_sizes=[16,8,4],
    n_epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    learning_rate=1e-3,
    verbose=10,
    seed=SEED
)

with Timer() as train_time:
    model.fit(data)

print("Took {} seconds for training.".format(train_time))

"""In the movie recommendation use case scenario, seen movies are not recommended to the users."""

with Timer() as test_time:
    users, items, preds = [], [], []
    item = list(train.itemID.unique())
    for user in train.userID.unique():
        user = [user] * len(item)
        users.extend(user)
        items.extend(item)
        preds.extend(list(model.predict(user, item, is_list=True)))

    all_predictions = pd.DataFrame(data={"userID": users, "itemID":items, "prediction":preds})

    merged = pd.merge(train, all_predictions, on=["userID", "itemID"], how="outer")
    all_predictions = merged[merged.rating.isnull()].drop('rating', axis=1)

print("Took {} seconds for prediction.".format(test_time))

"""### 4. Evaluate how well NCF performs

The ranking metrics are used for evaluation.
"""

eval_map = map(test, all_predictions, col_prediction='prediction', k=TOP_K)
eval_ndcg = ndcg_at_k(test, all_predictions, col_prediction='prediction', k=TOP_K)
eval_precision = precision_at_k(test, all_predictions, col_prediction='prediction', k=TOP_K)
eval_recall = recall_at_k(test, all_predictions, col_prediction='prediction', k=TOP_K)

print("MAP:\t%f" % eval_map,
      "NDCG:\t%f" % eval_ndcg,
      "Precision@K:\t%f" % eval_precision,
      "Recall@K:\t%f" % eval_recall, sep='\n')

"""NDCG - Normalised Discounted Cumulative Gain  
MAP - Mean Average Precision
"""

# Record results for tests - ignore this cell
store_metadata("map", eval_map)
store_metadata("ndcg", eval_ndcg)
store_metadata("precision", eval_precision)
store_metadata("recall", eval_recall)
store_metadata("train_time", train_time.interval)
store_metadata("test_time", test_time.interval)

