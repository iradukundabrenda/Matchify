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
    Reconstruct the ratings matrix from SVD components.
    """
    predicted = np.dot(np.dot(U, Sigma), VT)
    predicted = np.clip(predicted, 0, 5)  # clamp ratings to 0-5
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

# Keep a copy of original ratings with NaN
original_ratings = ratings_matrix.copy()

# -----------------------------
# Normalize: subtract movie means, leave NaNs as 0
# -----------------------------
col_means = np.nanmean(ratings_matrix, axis=0)
ratings_norm = np.where(np.isnan(ratings_matrix), 0, ratings_matrix - col_means)

# -----------------------------
# Compute SVD and predicted ratings
# -----------------------------
U, Sigma, VT = compute_svd(ratings_norm)
predicted_matrix = predict_ratings(U, Sigma, VT)

# Add back the movie means to predicted ratings
predicted_matrix += col_means
predicted_matrix = np.clip(predicted_matrix, 0, 5)  # ensure 0-5
predicted_matrix_rounded = np.round(predicted_matrix, 1)  # for display

print("\nPredicted Ratings Table:")
print(predicted_matrix_rounded)

# -----------------------------
# Recommend movies
# -----------------------------
user_index = list(ratings_df["user_id"].values).index(user_id)
predicted_row = predicted_matrix[user_index]
user_input_ratings = original_ratings[user_index]

unrated_predictions = get_unrated_predictions(user_input_ratings, predicted_row, movies)

if len(unrated_predictions) == 0:
    print(f"\nYou have already rated all movies, nothing to recommend!")
else:
    # Sort by predicted rating descending
    unrated_predictions.sort(key=lambda x: x[1], reverse=True)
    print(f"\nTop recommendations for {user_id}:")
    for movie, score in unrated_predictions[:2]:
        print(f"  {movie} (predicted rating: {round(score, 2)})")

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

