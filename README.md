# Matchify — Simple Movie Recommender System

## Purpose

Matchify is a small project that recommends movies to users based on their past ratings. The goal is to predict which movies a user might enjoy even if they haven’t rated them yet. This project demonstrates how linear algebra concepts, particularly **Singular Value Decomposition (SVD)**, can be applied in a real-world scenario.

## How It Works

1. **Data Input**:
   A CSV file (`ratings_small.csv`) contains user ratings for different movies. Blank or empty cells indicate movies that the user has not rated.

2. **Matrix Conversion**:
   The ratings table is converted into a numeric matrix, where rows represent users and columns represent movies. Missing ratings are temporarily replaced with **movie average ratings** for computation purposes.

3. **Normalization**:
   Each movie column is centered by subtracting its mean rating. This allows the SVD to capture **patterns in user preferences relative to average ratings**.

4. **SVD (Singular Value Decomposition)**:
   The normalized ratings matrix is decomposed into three matrices:
   * **U**: captures user patterns  
   * **Σ (Sigma)**: contains singular values showing importance of latent features  
   * **V^T**: captures movie patterns  

   This decomposition finds **latent relationships** between users and movies that are not directly visible in the raw data.

5. **Reconstruction & Prediction**:
   Multiplying the SVD components reconstructs an approximation of the original ratings matrix. Adding back the movie means produces **predicted ratings** for all user-movie combinations, including the ones users haven’t rated.

6. **Top Recommendations**:
   For each user, movies they haven’t rated are sorted by predicted ratings, and the top N movies are suggested as recommendations.

## Connection to Linear Algebra

* **Matrix Representation**: Users and movies are represented in a matrix, emphasizing concepts like **rows, columns, subspaces, and dimensions**.  
* **SVD**: Demonstrates **decomposition of a matrix into orthogonal components (U, V) and singular values (Σ)**, showing how linear transformations can reveal hidden structure.  
* **Prediction**: Reconstructing the matrix illustrates **matrix multiplication, rank, and linear combinations** in a practical application.  
* **Understanding Concepts**: Course topics including **vector spaces, bases, subspaces, null space, rank, eigenvalues/eigenvectors, and matrix norms** are applied in a real-world context.

## Logic of the Algorithm

1. Start with a **ratings matrix** with users as rows and movies as columns.  
2. Replace missing ratings with **the average rating for that movie** to allow matrix calculations.  
3. Normalize the matrix by subtracting the movie mean, so SVD can detect patterns **relative to average ratings**.  
4. Apply **SVD** to extract latent features of users and movies.  
5. Reconstruct the matrix to predict missing ratings. Add back movie means to produce final predictions.  
6. Recommend the **highest predicted movies** for each user that they haven’t rated yet.

This ensures that even if a user has rated only a few movies, the algorithm can suggest movies they are likely to enjoy based on patterns from all users’ ratings.

## How to Run

1. Make sure you have Python 3 installed with the `numpy`, `pandas`, and `matplotlib` libraries.  
2. From the project root, run:

```bash
python3 src/main.py

