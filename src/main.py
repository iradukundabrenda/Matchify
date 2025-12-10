import pandas as pd
import numpy as np
from svd_matchify import compute_svd, predict_ratings
import matplotlib.pyplot as plt

def get_unrated_predictions(user_ratings, predicted_row, movie_list):
    unrated = []
    for i, rating in enumerate(user_ratings):
        if rating == "" or rating is None:  # user did NOT rate this movie
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
    rating = input(f"{movie}: ").strip()  # .strip() removes accidental spaces
    if rating == "":
        new_ratings.append(np.nan)  # safe blank handling
    else:
        new_ratings.append(float(rating))

# Add new user to DataFrame
ratings_df.loc[len(ratings_df)] = [user_id] + new_ratings

# Step 3: Convert the ratings table to a matrix
ratings_matrix = ratings_df.iloc[:, 1:].values.astype(float)

# Step 4: Compute SVD
U, Sigma, VT = compute_svd(ratings_matrix)

# Step 5: Predict missing ratings
predicted_matrix = predict_ratings(U, Sigma, VT)

print("\nPredicted Ratings Table:")
print(np.round(predicted_matrix, 2))

# Step 6: Predict ratings for unseen movies

print(f"\nTop recommendations for {user_id}:")

# Find this user’s row in the dataframe
user_index = list(ratings_df["user_id"].values).index(user_id)

# Get predicted ratings for this user
predicted_row = predicted_matrix[user_index]

# Extract the user’s original input ratings (strings or numbers)
user_input_ratings = [ratings_df.iloc[user_index, i + 1] for i in range(len(movies))]

# Get predicted ratings only for movies the user did NOT rate
unrated_predictions = get_unrated_predictions(user_input_ratings, predicted_row, movies)

# If the user already rated everything
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

