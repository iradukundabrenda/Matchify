import numpy as np

def compute_svd(matrix):
    """Compute SVD (Singular Value Decomposition) of the matrix."""
    matrix_filled = np.nan_to_num(matrix, nan=0.0)
    U, Sigma, VT = np.linalg.svd(matrix_filled, full_matrices=False)
    return U, Sigma, VT

def predict_ratings(U, Sigma, VT):
    """Reconstruct the matrix using SVD to predict missing ratings."""
    Sigma_matrix = np.diag(Sigma)
    predicted_matrix = np.dot(np.dot(U, Sigma_matrix), VT)
    return predicted_matrix

