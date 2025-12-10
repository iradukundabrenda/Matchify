# Matchify — Simple Movie Recommender System

## Purpose

Matchify is a small project that recommends movies to users based on their past ratings. The goal is to predict which movies a user might like even if they haven’t rated them yet. This project demonstrates how Linear Algebra concepts can be applied to real-world problems like recommendation systems.

## How It Works

1. **Data Input**: A CSV file (`ratings_small.csv`) contains user ratings for different movies. Empty cells indicate movies that the user has not rated.  
2. **Matrix Conversion**: The ratings table is converted into a numeric matrix, where rows represent users and columns represent movies. Missing ratings are temporarily filled with 0 for computation purposes.  
3. **SVD (Singular Value Decomposition)**: The ratings matrix is decomposed into three matrices U, Σ (Sigma), and V^T using SVD. This factorization captures latent patterns in user preferences and movie features.  
4. **Reconstruction & Prediction**: Multiplying the SVD components reconstructs the original matrix and predicts missing ratings.  
5. **Top Recommendations**: For each user, the top N movies with the highest predicted ratings are selected and displayed as recommendations.

## Connection to Linear Algebra

- **Matrix Representation**: The ratings data is treated as a matrix, highlighting concepts like **rows/columns, subspaces, and dimensions**.  
- **SVD**: This decomposition uses Linear Algebra to break the matrix into **orthogonal components** (U, V) and a **diagonal matrix of singular values** (Σ), showing how linear transformations can capture patterns.  
- **Prediction**: Reconstructing the matrix from its SVD components demonstrates **matrix multiplication, rank, and linear combinations** in a practical context.  
- **Understanding Concepts**: This project touches on key linear algebra topics from the course, including **vector spaces, bases, subspaces, null space, rank, eigenvalues/eigenvectors**, and **matrix norms** (implicitly via SVD).

## How to Run

1. Make sure you have Python 3 installed with the `numpy` and `pandas` libraries.  
2. From the project root, run:

```bash
python3 src/main.py

