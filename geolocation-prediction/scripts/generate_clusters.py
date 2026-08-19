import os
import numpy as pd
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans

def latlon_to_cartesian(lat, lon):
    """
    Convert latitude and longitude to 3D Cartesian coordinates on a unit sphere.
    """
    lat_rad = np.radians(lat)
    lon_rad = np.radians(lon)
    x = np.cos(lat_rad) * np.cos(lon_rad)
    y = np.cos(lat_rad) * np.sin(lon_rad)
    z = np.sin(lat_rad)
    return x, y, z

def cartesian_to_latlon(x, y, z):
    """
    Convert 3D Cartesian coordinates back to latitude and longitude.
    """
    lon_rad = np.arctan2(y, x)
    hyp = np.sqrt(x * x + y * y)
    lat_rad = np.arctan2(z, hyp)
    lat = np.degrees(lat_rad)
    lon = np.degrees(lon_rad)
    return lat, lon

def main():
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    ORIGINAL_CSV = os.path.join(BASE_DIR, r"original_training_dataset\noised_dataset\ground_truth_coordinates.csv")
    EXTRA_CSV = os.path.join(BASE_DIR, r"extra_training_dataset\coordinates.csv")
    
    OUTPUT_DIR = os.path.join(BASE_DIR, "clustering_data")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    print("Loading Original Dataset...")
    orig_df = pd.read_csv(ORIGINAL_CSV)
    
    # 1. Transform original coords to Cartesian
    orig_x, orig_y, orig_z = latlon_to_cartesian(orig_df['latitude'].values, orig_df['longitude'].values)
    orig_cartesian = np.column_stack((orig_x, orig_y, orig_z))
    
    # 2. Train K-Means strictly on Original Dataset
    print("Training KMeans (K=16) on original 19K dataset...")
    kmeans_16 = KMeans(n_clusters=16, random_state=42, n_init=10)
    kmeans_16.fit(orig_cartesian)
    
    print("Training KMeans (K=160) on original 19K dataset...")
    kmeans_160 = KMeans(n_clusters=160, random_state=42, n_init=10)
    kmeans_160.fit(orig_cartesian)
    
    # 3. Save Centroids
    print("Saving Centroids...")
    def save_centroids(kmeans_model, filename):
        centers = kmeans_model.cluster_centers_
        lats, lons = cartesian_to_latlon(centers[:, 0], centers[:, 1], centers[:, 2])
        df = pd.DataFrame({
            'cluster_id': [f"CLUSTER_{i}" for i in range(len(centers))],
            'x': centers[:, 0],
            'y': centers[:, 1],
            'z': centers[:, 2],
            'latitude': lats,
            'longitude': lons
        })
        df.to_csv(os.path.join(OUTPUT_DIR, filename), index=False)
        
    save_centroids(kmeans_16, "centroids_16.csv")
    save_centroids(kmeans_160, "centroids_160.csv")
    
    # 4. Process all data and assign clusters
    print("Assigning clusters to Original Dataset...")
    orig_df['cluster_16'] = [f"CLUSTER_{i}" for i in kmeans_16.labels_]
    orig_df['cluster_160'] = [f"CLUSTER_{i}" for i in kmeans_160.labels_]
    orig_df['dataset_source'] = 'original'
    orig_df = orig_df[['image_id', 'latitude', 'longitude', 'cluster_16', 'cluster_160', 'dataset_source']]
    
    print("Loading Extra Dataset...")
    extra_df = pd.read_csv(EXTRA_CSV)
    
    print("Assigning clusters to Extra Dataset...")
    extra_x, extra_y, extra_z = latlon_to_cartesian(extra_df['latitude'].values, extra_df['longitude'].values)
    extra_cartesian = np.column_stack((extra_x, extra_y, extra_z))
    
    extra_labels_16 = kmeans_16.predict(extra_cartesian)
    extra_labels_160 = kmeans_160.predict(extra_cartesian)
    
    # Format extra dataframe to match original
    extra_df['image_id'] = extra_df['id'].astype(str)
    extra_df['cluster_16'] = [f"CLUSTER_{i}" for i in extra_labels_16]
    extra_df['cluster_160'] = [f"CLUSTER_{i}" for i in extra_labels_160]
    extra_df['dataset_source'] = 'extra'
    
    extra_df_clean = extra_df[['image_id', 'latitude', 'longitude', 'cluster_16', 'cluster_160', 'dataset_source']]
    
    # 5. Concatenate and save
    print("Concatenating and saving unified clustering dataset...")
    all_data_df = pd.concat([orig_df, extra_df_clean], ignore_index=True)
    all_data_df.to_csv(os.path.join(OUTPUT_DIR, "all_image_clusters.csv"), index=False)
    
    print("Done! Check the 'clustering_data' directory.")

if __name__ == "__main__":
    main()
