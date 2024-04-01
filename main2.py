
from typing import Dict, Text
import pprintpp

import numpy as np
import pandas as pd

import tensorflow as tf
import tensorflow_datasets as tfds
import tensorflow_recommenders as tfrs


interaction_data = pd.read_csv("/kaggle/input/food-com-recipes-and-user-interactions/RAW_interactions.csv")
recipe_data = pd.read_csv("/kaggle/input/food-com-recipes-and-user-interactions/RAW_recipes.csv")

interaction_train = pd.read_csv("/kaggle/input/food-com-recipes-and-user-interactions/interactions_train.csv")
interaction_test = pd.read_csv("/kaggle/input/food-com-recipes-and-user-interactions/interactions_test.csv")

interaction_data = interaction_data.astype({'user_id': 'string', 'recipe_id':' string'})
interaction_train = interaction_train.astype({'user_id': 'string', 'recipe_id':' string'})
interaction_test = interaction_test.astype({'user_id': 'string', 'recipe_id':' string'})


class RankingModel(tf.keras.Model):

    def __init__(self):
        super().__init__()
        embedding_dimension = 32

        self.user_embeddings = tf.keras.Sequential([
            tf.keras.layers.experimental.preprocessing.StringLookup(
                vocabulary=uniqueUserIds, mask_token=None),
            # add addional embedding to account for unknow tokens
            tf.keras.layers.Embedding(len(uniqueUserIds)+ 1 , embedding_dimension)
        ])

        self.product_embeddings = tf.keras.Sequential([
            tf.keras.layers.experimental.preprocessing.StringLookup(
                vocabulary=uniqueFoodIds, mask_token=None),
            # add addional embedding to account for unknow tokens
            tf.keras.layers.Embedding(len(uniqueFoodIds)+ 1 , embedding_dimension)
        ])
        # Set up a retrieval task and evaluation metrics over the
        # entire dataset of candidates.
        self.ratings = tf.keras.Sequential([
            tf.keras.layers.Dense(256, activation="relu"),
            tf.keras.layers.Dense(64, activation="relu"),
            tf.keras.layers.Dense(1)
        ])

    def call(self, userId, foodId):
        user_embeddings  = self.user_embeddings (userId)
        food_embeddings = self.product_embeddings(foodId)
        return self.ratings(tf.concat([user_embeddings, food_embeddings], axis=1))


# Build a model.
class FoodModel(tfrs.models.Model):

    def __init__(self):
        super().__init__()
        self.ranking_model: tf.keras.Model = RankingModel()
        self.task: tf.keras.layers.Layer = tfrs.tasks.Ranking(
            loss=tf.keras.losses.MeanSquaredError(),
            metrics=[tf.keras.metrics.RootMeanSquaredError()])

    def compute_loss(self, features, training=False):
        rating_predictions = self.ranking_model(features["userID"], features["foodID"])

        return self.task(labels=features["rating"], predictions=rating_predictions)


uniqueUserIds = interaction_data.user_id.unique()
uniqueFoodIds = interaction_data.recipe_id.unique()

train_data = tf.data.Dataset.from_tensor_slices(
    {
        "userID": tf.cast(interaction_train.user_id.values, tf.string),
        "foodID": tf.cast(interaction_train.recipe_id.values, tf.string),
        "rating": tf.cast(interaction_train.rating.values, tf.float32)
    })

test_data = tf.data.Dataset.from_tensor_slices(
    {
        "userID": tf.cast(interaction_test.user_id.values, tf.string),
        "foodID": tf.cast(interaction_test.recipe_id.values, tf.string),
        "rating": tf.cast(interaction_test.rating.values, tf.float32)
    })

tf.random.set_seed(42)

train_data = train_data.shuffle(100_000, seed=42, reshuffle_each_iteration=False)

model = FoodModel()
model.compile(optimizer=tf.keras.optimizers.Adagrad(learning_rate=0.001))
cached_train = train_data.shuffle(100_000).batch(8192).cache()
cached_test = test_data.batch(4096).cache()
model.fit(cached_train, epochs=10)




user_rand = uniqueUserIds[200]
test_rating = {}
for m in test_data.take(10):
    test_rating[m["foodID"].numpy()] = RankingModel(tf.convert_to_tensor([user_rand]),
                                                      tf.convert_to_tensor([m["foodID"]]))

print("Top 10 recommended products for User {}: ".format(user_rand))
for m in sorted(test_rating, key=test_rating.get, reverse=True):
    print(recipe_data.loc[recipe_data['id'] == int(m.decode())]['name'].item())

