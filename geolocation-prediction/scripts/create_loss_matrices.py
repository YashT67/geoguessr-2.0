import os
import torch
import pandas as pd
import numpy as np

def haversine_distance_matrix(lats, lons):
    """
    Computes a pairwise Haversine distance matrix between all points.
    lats, lons: 1D numpy arrays
    Returns: 2D numpy array of shape (N, N)
    """
    lats = np.radians(lats)
    lons = np.radians(lons)
    
    # Broadcast to N x N matrices
    lat1 = lats[:, np.newaxis]
    lon1 = lons[:, np.newaxis]
    lat2 = lats[np.newaxis, :]
    lon2 = lons[np.newaxis, :]
    
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    
    a = np.sin(dlat / 2.0)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0)**2
    c = 2 * np.arcsin(np.sqrt(np.clip(a, 0, 1)))
    
    R = 6371.0 # Earth radius in km
    return R * c

def create_target_matrix(centroids_csv, sigma, output_pt):
    df = pd.read_csv(centroids_csv)
    
    # Ensure they are sorted by cluster integer index to match the PyTorch embedding layers
    df['cluster_idx'] = df['cluster_id'].apply(lambda x: int(x.split('_')[1]))
    df = df.sort_values('cluster_idx')
    
    lats = df['latitude'].values
    lons = df['longitude'].values
    
    # 1. Compute pairwise Haversine distances (N x N)
    D = haversine_distance_matrix(lats, lons)
    
    # 2. Apply Gaussian Radial Basis Function
    # Formula: exp(- D^2 / (2 * sigma^2))
    similarity = np.exp(- (D ** 2) / (2 * (sigma ** 2)))
    
    # 3. Row-normalize so each row sums to 1 (making them valid probability distributions)
    row_sums = similarity.sum(axis=1, keepdims=True)
    target_distributions = similarity / row_sums
    
    # 4. Convert to PyTorch tensor and save
    tensor_matrix = torch.tensor(target_distributions, dtype=torch.float32)
    torch.save(tensor_matrix, output_pt)
    print(f"Saved {tensor_matrix.shape} target matrix (Sigma={sigma}km) to {os.path.basename(output_pt)}")

def main():
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    CLUSTERS_DIR = os.path.join(BASE_DIR, "clustering_data")
    
    # Head 1: K=16, Sigma = 1225 km
    create_target_matrix(
        centroids_csv=os.path.join(CLUSTERS_DIR, "centroids_16.csv"),
        sigma=1225.0,
        output_pt=os.path.join(CLUSTERS_DIR, "loss_matrix_16.pt")
    )
    
    # Head 2: K=160, Sigma = 225 km
    create_target_matrix(
        centroids_csv=os.path.join(CLUSTERS_DIR, "centroids_160.csv"),
        sigma=225.0,
        output_pt=os.path.join(CLUSTERS_DIR, "loss_matrix_160.pt")
    )
    
if __name__ == "__main__":
    main()
