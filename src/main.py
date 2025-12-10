import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# -----------------------------
# Helper functions
# -----------------------------
def compute_svd(ratings_matrix):
    """
    Compute the Singular Value Decomposition of the ratings matrix.
    """
    U, S, VT = np.linalg.svd(ratings_matrix, full_matrices=False)
    Sigma = np.diag(S)
    return U, Sigma, VT

def predict_ratings(U, Sigma, VT):
    """
    Compute the predicted ratings by reconstructing the matrix.
    """
    predicted = np.dot(np.dot(U, Sigma), VT)
    predicted = np.clip(predicted, 0, 5)  # ratings between 0-5
    predicted = np.round(predicted).astype(int)  # round to nearest integer
    return predicted

def get_unrated_predictions(user_ratings, predicted_row, movie_list):
    """
    Return a list of movies the user hasn't rated, along with predicted scores.
    """
    unrated = []
    for i, rating in enumerate(user_ratings):
        if np.isnan(rating):  # user did not rate
            unrated.append((movie_list[i], predicted_row[i]))
    return unrated

# -----------------------------
# Load existing ratings
# -----------------------------
ratings_file = "data/ratings_small.csv"
ratings_df = pd.read_csv(ratings_file)

# Replace blank strings with NaN
ratings_df = ratings_df.replace(r"^\s*$", np.nan, regex=True)

print("Original Ratings Table:")
print(ratings_df)

# -----------------------------
# Add new user ratings
# -----------------------------
user_id = input("\nEnter your user ID (or create a new one): ")

movies = ratings_df.columns[1:]  # skip user_id column
new_ratings = []

print("Enter your ratings for the following movies (0-5). Leave blank if you haven't seen it:")
for movie in movies:
    rating = input(f"{movie}: ")
    if rating == "":
        new_ratings.append(np.nan)
    else:
        value = float(rating)
        value = max(0, min(5, value))  # clamp between 0-5
        new_ratings.append(value)

# Add new user to DataFrame
ratings_df.loc[len(ratings_df)] = [user_id] + new_ratings

# -----------------------------
# Convert ratings to numeric matrix for SVD
# -----------------------------
ratings_matrix = ratings_df.iloc[:, 1:].values.astype(float)

# Keep a copy of original ratings with NaN to know what the user rated
original_ratings = ratings_matrix.copy()

# Replace NaN with column means for SVD
col_means = np.nanmean(ratings_matrix, axis=0)
inds = np.where(np.isnan(ratings_matrix))
ratings_matrix[inds] = np.take(col_means, inds[1])

# -----------------------------
# Compute SVD and predicted ratings
# -----------------------------
U, Sigma, VT = compute_svd(ratings_matrix)
predicted_matrix = predict_ratings(U, Sigma, VT)

print("\nPredicted Ratings Table:")
print(predicted_matrix)

# -----------------------------
# Recommend movies
# -----------------------------
user_index = list(ratings_df["user_id"].values).index(user_id)
predicted_row = predicted_matrix[user_index].astype(float)
user_input_ratings = original_ratings[user_index]

# -----------------------------
# Adjust predicted ratings for new user
# -----------------------------
user_mean = np.nanmean(user_input_ratings)  # mean of user's input ratings
if np.isnan(user_mean):
    user_mean = 3  # default if user left everything blank
# Shift predicted row toward the user's mean rating
predicted_row += (user_mean - np.mean(predicted_row))
predicted_row = np.clip(predicted_row, 0, 5)
predicted_row = np.round(predicted_row).astype(int)

unrated_predictions = get_unrated_predictions(user_input_ratings, predicted_row, movies)

if len(unrated_predictions) == 0:
    print(f"\nYou have already rated all movies, nothing to recommend!")
else:
    # Sort by predicted rating descending
    unrated_predictions.sort(key=lambda x: x[1], reverse=True)
    print(f"\nTop recommendations for {user_id}:")
    for movie, score in unrated_predictions[:2]:
        print(f"  {movie} (predicted rating: {score})")

# -----------------------------
# Visual heatmap
# -----------------------------
plt.imshow(predicted_matrix, cmap='coolwarm', aspect='auto')
plt.colorbar(label='Predicted Rating')
plt.xticks(ticks=np.arange(len(movies)), labels=movies)
plt.yticks(ticks=np.arange(len(ratings_df)), labels=ratings_df['user_id'])
plt.xlabel('Movies')
plt.ylabel('Users')
plt.title('Predicted Movie Ratings')
plt.show()

