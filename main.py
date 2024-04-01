from typing import Dict, Text
import numpy as np
import pandas as pd

import tensorflow as tf
import tensorflow_datasets as tfds
import tensorflow_recommenders as tfrs

# Load the preprocessed data
interaction_data = pd.read_csv("RAW_interactions.csv")
recipe_data = pd.read_csv("RAW_recipes.csv")

interaction_train = pd.read_csv("interactions_train.csv")
interaction_test = pd.read_csv("interactions_test.csv")

# Convert IDs to strings to be used in embeddings
interaction_data = interaction_data.astype({'user_id': 'string', 'recipe_id': 'string'})
interaction_train = interaction_train.astype({'user_id': 'string', 'recipe_id': 'string'})
interaction_test = interaction_test.astype({'user_id': 'string', 'recipe_id': 'string'})

# Unique IDs
uniqueUserIds = interaction_data.user_id.unique()
uniqueFoodIds = interaction_data.recipe_id.unique()

# TensorFlow datasets
train_data = tf.data.Dataset.from_tensor_slices({
    "userID": tf.cast(interaction_train.user_id.values, tf.string),
    "foodID": tf.cast(interaction_train.recipe_id.values, tf.string),
    "rating": tf.cast(interaction_train.rating.values, tf.float32)
})

test_data = tf.data.Dataset.from_tensor_slices({
    "userID": tf.cast(interaction_test.user_id.values, tf.string),
    "foodID": tf.cast(interaction_test.recipe_id.values, tf.string),
    "rating": tf.cast(interaction_test.rating.values, tf.float32)
})

tf.random.set_seed(42)

train_data = train_data.shuffle(100_000, seed=42, reshuffle_each_iteration=False)


class RankingModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        embedding_dimension = 32

        # User and product embeddings
        self.user_embeddings = tf.keras.Sequential([
            tf.keras.layers.StringLookup(vocabulary=uniqueUserIds, mask_token=None),
            tf.keras.layers.Embedding(len(uniqueUserIds) + 1, embedding_dimension)
        ])

        self.product_embeddings = tf.keras.Sequential([
            tf.keras.layers.StringLookup(vocabulary=uniqueFoodIds, mask_token=None),
            tf.keras.layers.Embedding(len(uniqueFoodIds) + 1, embedding_dimension)
        ])

        # Rating prediction layers
        self.ratings = tf.keras.Sequential([
            tf.keras.layers.Dense(256, activation="relu"),
            tf.keras.layers.Dense(64, activation="relu"),
            tf.keras.layers.Dense(1)
        ])

    def call(self, inputs):
        userId, foodId = inputs
        user_embeddings = self.user_embeddings(userId)
        food_embeddings = self.product_embeddings(foodId)
        return self.ratings(tf.concat([user_embeddings, food_embeddings], axis=1))


class FoodModel(tfrs.models.Model):
    def __init__(self):
        super().__init__()
        self.ranking_model: tf.keras.Model = RankingModel()
        self.task: tf.keras.layers.Layer = tfrs.tasks.Ranking(
            loss=tf.keras.losses.MeanSquaredError(),
            metrics=[tf.keras.metrics.RootMeanSquaredError()]
        )

    def compute_loss(self, features, training=False):
        rating_predictions = self.ranking_model((features["userID"], features["foodID"]))
        return self.task(labels=features["rating"], predictions=rating_predictions)


model = FoodModel()
model.compile(optimizer=tf.keras.optimizers.Adagrad(learning_rate=0.001))

# Preparing the data
cached_train = train_data.shuffle(100_000).batch(8192).cache()
cached_test = test_data.batch(4096).cache()

# Train the model
model.fit(cached_train, epochs=10)

# Evaluate the model
model.evaluate(cached_test, return_dict=True)


# Function for predicting ratings
def predict_rating(user_id, food_id):
    inputs = {"userID": tf.convert_to_tensor([user_id]), "foodID": tf.convert_to_tensor([food_id])}
    predicted_rating = model.ranking_model(inputs)
    return predicted_rating.numpy()[0][0]


# Generating predictions for a random user
user_rand = uniqueUserIds[200]
test_rating = {}

for m in test_data.take(10):
    food_id = m["foodID"].numpy().decode()
    predicted_rating = predict_rating(user_rand, food_id)
    test_rating[food_id] = predicted_rating

# Displaying the recommendations
print(f"Top 10 recommended products for User {user_rand}:")
for food_id, rating in sorted(test_rating.items(), key=lambda x: x[1], reverse=True):
    recipe_name = recipe_data.loc[recipe_data['id'] == int(food_id)]['name'].item()
    print(f"{recipe_name}: {rating}")



