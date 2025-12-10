import pandas as pd
import numpy as np
from svd_matchify import compute_svd, predict_ratings
import matplotlib.pyplot as plt

# Helper function to get predictions only for movies the user hasn't rated
def get_unrated_predictions(user_ratings, predicted_row, movie_list):
    unrated = []
    for i, rating in enumerate(user_ratings):
        if pd.isna(rating):  # Check for NaN (unrated)
            unrated.append((movie_list[i], predicted_row[i]))
    return unrated

# Step 1: Load the data
ratings_file = "data/ratings_small.csv"
ratings_df = pd.read_csv(ratings_file)

# Fix: replace blank strings or spaces with NaN
ratings_df = ratings_df.replace(r"^\s*$", np.nan, regex=True)

print("Original Ratings Table:")
print(ratings_df)

# Step 2: Add new user ratings interactively
user_id = input("\nEnter your user ID (or create a new one): ")

movies = ratings_df.columns[1:]  # skip user_id column
new_ratings = []

print("Enter your ratings for the following movies (leave blank if you haven't seen it):")
for movie in movies:
    while True:
        rating = input(f"{movie}: ")
        if rating == "":
            new_ratings.append(np.nan)
            break
        try:
            r = float(rating)
            if 0 <= r <= 5:  # allowed range
                new_ratings.append(r)
                break
            else:
                print("Please enter a number between 0 and 5, or leave blank.")
        except ValueError:
            print("Invalid input. Please enter a number between 0 and 5, or leave blank.")


# Add new user to DataFrame

ratings_df.loc[len(ratings_df)] = [user_id] + new_ratings

# Step 3: Convert the ratings table to a numeric matrix

ratings_matrix = ratings_df.iloc[:, 1:].values.astype(float)

# Fill NaN (unrated) values with the average rating for each movie
# This prevents SVD from failing and gives more realistic predictions

movie_means = np.nanmean(ratings_matrix, axis=0)
indices_nan = np.where(np.isnan(ratings_matrix))
ratings_matrix[indices_nan] = np.take(movie_means, indices_nan[1])

# Step 4: Compute SVD

U, Sigma, VT = compute_svd(ratings_matrix)

# Step 5: Predict ratings for all users

predicted_matrix = predict_ratings(U, Sigma, VT)

# Clip predictions to stay within 0-5

predicted_matrix = np.clip(predicted_matrix, 0, 5)

print("\nPredicted Ratings Table:")
print(np.round(predicted_matrix, 1))

# Step 6: Recommend top unrated movies for the current user

def get_unrated_predictions(user_ratings, predicted_row, movie_list):
    unrated = []
    for i, rating in enumerate(user_ratings):
        if np.isnan(rating):  # user did NOT rate this movie
            unrated.append((movie_list[i], predicted_row[i]))
    return unrated

print(f"\nTop recommendations for {user_id}:")
user_index = list(ratings_df.iloc[:, 0].values).index(user_id)
predicted_row = predicted_matrix[user_index]
user_input_ratings = ratings_df.iloc[user_index, 1:].values

unrated_predictions = get_unrated_predictions(user_input_ratings, predicted_row, movies)

if len(unrated_predictions) == 0:
    print("You have already rated all movies — nothing to recommend!")
else:
    # Sort by predicted rating (highest first)
    unrated_predictions.sort(key=lambda x: x[1], reverse=True)
    # Show top 2 recommendations
    for movie, score in unrated_predictions[:2]:
        print(f"  {movie} (predicted rating: {score:.2f})")


# Step 7: Visual heatmap
plt.imshow(predicted_matrix, cmap='coolwarm', aspect='auto')
plt.colorbar(label='Predicted Rating')
plt.xlabel('Movies')
plt.ylabel('Users')
plt.title('Predicted Movie Ratings')
plt.show()

