import pandas as pd
import numpy as np
from svd_matchify import compute_svd, predict_ratings

# Step 1: Load the data

ratings_file = "data/ratings_small.csv"

# Read CSV and treat empty strings as NaN
ratings_df = pd.read_csv(ratings_file, na_values=["", " "])

print("Original Ratings Table:")
print(ratings_df)

# Convert the ratings table to a numeric matrix (skip the first column with user IDs)
ratings_matrix = ratings_df.iloc[:, 1:].values
ratings_matrix = ratings_matrix.astype(float)  # Ensure all values are floats

# Step 2: Compute SVD

U, Sigma, VT = compute_svd(ratings_matrix)

# Step 3: Predict missing ratings

predicted_matrix = predict_ratings(U, Sigma, VT)

# Replace tiny numerical errors with 0
predicted_matrix[np.abs(predicted_matrix) < 1e-10] = 0


print("\nPredicted Ratings Table:")
print(predicted_matrix)

# Step 4: Recommend top movies

top_n = 2
for i, user_ratings in enumerate(predicted_matrix):
    top_indices = user_ratings.argsort()[::-1][:top_n]
    print(f"\nTop {top_n} recommendations for user {ratings_df.iloc[i,0]}:")
    for idx in top_indices:
        print(f"  Movie {idx + 1} (predicted rating: {user_ratings[idx]:.2f})")

