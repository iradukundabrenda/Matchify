import pandas as pd
import numpy as np

# This file will compute a very simple movie recommender using SVD


# Step 1: Load the data
# CSV file with user ratings
# Rows = users, Columns = movies

ratings_file = "../data/ratings_small.csv"
ratings_df = pd.read_csv(ratings_file)

print("Original Ratings Table:")
print(ratings_df)

# Convert the ratings table to a numeric matrix
# Skip the first column which has user IDs

ratings_matrix = ratings_df.iloc[:, 1:].values

# Step 2: Compute SVD
# Simple SVD using numpy
# U, Sigma, VT = decomposition of the ratings matrix

U, Sigma, VT = np.linalg.svd(ratings_matrix, full_matrices=False)

# Step 3: Reconstruct the matrix
# Multiply the SVD components to predict missing ratings

Sigma_matrix = np.diag(Sigma)
predicted_matrix = np.dot(np.dot(U, Sigma_matrix), VT)

print("\nPredicted Ratings Table:")
print(predicted_matrix)

# Step 4: Recommend top movies
# Recommend the top 2 movies for each user

top_n = 2
for i, user_ratings in enumerate(predicted_matrix):

    # Get the indices of the top ratings

    top_indices = user_ratings.argsort()[::-1][:top_n]
    print(f"\nTop {top_n} recommendations for user {ratings_df.iloc[i,0]}:")
    for idx in top_indices:
        print(f"  Movie {idx + 1} (predicted rating: {user_ratings[idx]:.2f})")

