import os
import glob
import pandas as pd
import argparse
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from tqdm import tqdm
from pathlib import Path
import warnings

# Suppress some noisy warnings from torchvision
warnings.filterwarnings("ignore")

class GeoImageDataset(Dataset):
    def __init__(self, data_list, transform=None):
        """
        data_list: list of dicts with keys 'image_path', 'latitude', 'longitude', 'image_id'
        """
        self.data_list = data_list
        self.transform = transform

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        item = self.data_list[idx]
        image_path = item['image_path']
        
        try:
            image = Image.open(image_path).convert('RGB')
            if self.transform:
                image = self.transform(image)
        except Exception as e:
            # If an image is corrupt, return None (handled in collate_fn)
            print(f"Error loading image {image_path}: {e}")
            return None
            
        return {
            'image': image,
            'latitude': torch.tensor(item['latitude'], dtype=torch.float32),
            'longitude': torch.tensor(item['longitude'], dtype=torch.float32),
            'image_id': item['image_id']
        }

def collate_fn(batch):
    # Filter out None values from corrupt images
    batch = [b for b in batch if b is not None]
    if len(batch) == 0:
        return {}
    
    images = torch.stack([b['image'] for b in batch])
    latitudes = torch.stack([b['latitude'] for b in batch])
    longitudes = torch.stack([b['longitude'] for b in batch])
    image_ids = [b['image_id'] for b in batch]
    
    return {
        'images': images,
        'latitudes': latitudes,
        'longitudes': longitudes,
        'image_ids': image_ids
    }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dry-run', action='store_true', help='Run on a small subset of data to test the pipeline')
    args = parser.parse_args()

    # Configuration
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    ORIGINAL_CSV = os.path.join(BASE_DIR, r"original_training_dataset\noised_dataset\ground_truth_coordinates.csv")
    ORIGINAL_IMG_DIR = os.path.join(BASE_DIR, r"original_training_dataset\noised_dataset\images")
    
    EXTRA_CSV = os.path.join(BASE_DIR, r"extra_training_dataset\coordinates.csv")
    EXTRA_IMG_DIR = os.path.join(BASE_DIR, r"extra_training_dataset\images")
    
    OUTPUT_FILE = os.path.join(BASE_DIR, "extracted_features_dinov2_large.pt")
    
    BATCH_SIZE = 32
    NUM_WORKERS = 0  # Reduced slightly from 16 to ensure stability on Windows
    IMAGE_SIZE = 420
    
    print("Preparing dataset...")
    data_list = []
    
    # 1. Load Original Dataset
    print(f"Loading original dataset from {ORIGINAL_CSV}...")
    orig_df = pd.read_csv(ORIGINAL_CSV)
    for _, row in orig_df.iterrows():
        img_id = str(row['image_id'])
        img_path = os.path.join(ORIGINAL_IMG_DIR, f"{img_id}.jpg")
        if os.path.exists(img_path):
            data_list.append({
                'image_path': img_path,
                'latitude': row['latitude'],
                'longitude': row['longitude'],
                'image_id': img_id
            })
    
    # 2. Load Extra Dataset
    print(f"Loading extra dataset from {EXTRA_CSV}...")
    extra_df = pd.read_csv(EXTRA_CSV)
    
    # Pre-scan extra images to build a fast lookup map: ID -> filepath
    print(f"Scanning extra image directory {EXTRA_IMG_DIR}...")
    extra_img_paths = list(Path(EXTRA_IMG_DIR).rglob('*.jpg'))
    # Extract the filename without extension to use as ID
    extra_img_map = {p.stem: str(p) for p in extra_img_paths}
    
    for _, row in extra_df.iterrows():
        img_id = str(row['id'])
        if img_id in extra_img_map:
            data_list.append({
                'image_path': extra_img_map[img_id],
                'latitude': row['latitude'],
                'longitude': row['longitude'],
                'image_id': img_id
            })
            
    if args.dry_run:
        print("DRY RUN: Limiting to 64 images for testing.")
        data_list = data_list[:64]
        
    print(f"Total valid images found: {len(data_list)}")
    
    # DINOv2 transforms (Resize, CenterCrop, Normalize)
    transform = transforms.Compose([
        transforms.Resize(IMAGE_SIZE, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ])
    
    dataset = GeoImageDataset(data_list, transform=transform)
    dataloader = DataLoader(
        dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=False, 
        num_workers=NUM_WORKERS,
        collate_fn=collate_fn,
        pin_memory=True
    )
    
    # Load Model
    print("Loading DINOv2 Large (dinov2_vitl14_reg) model...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitl14_reg')
    model = model.to(device)
    model.eval()
    
    # Storage
    all_features = []
    all_latitudes = []
    all_longitudes = []
    all_image_ids = []
    
    print("Starting extraction...")
    with torch.no_grad():
        for i, batch in enumerate(tqdm(dataloader, desc="Extracting Embeddings")):
            if not batch:
                continue
                
            images = batch['images'].to(device)
            # DINOv2 returns the CLS token embedding directly when called
            features = model(images)
            
            # Move to CPU immediately to save RAM
            all_features.append(features.cpu())
            all_latitudes.append(batch['latitudes'])
            all_longitudes.append(batch['longitudes'])
            all_image_ids.extend(batch['image_ids'])
            
            # Save periodic checkpoints
            if (i + 1) % 1000 == 0:
                checkpoint_data = {
                    'features': torch.cat(all_features, dim=0),
                    'latitudes': torch.cat(all_latitudes, dim=0),
                    'longitudes': torch.cat(all_longitudes, dim=0),
                    'image_ids': all_image_ids
                }
                torch.save(checkpoint_data, OUTPUT_FILE + ".tmp")
                print(f" Saved intermediate checkpoint at batch {i+1}")

    # Final concat and save
    print("Concatenating all features...")
    final_data = {
        'features': torch.cat(all_features, dim=0),
        'latitudes': torch.cat(all_latitudes, dim=0),
        'longitudes': torch.cat(all_longitudes, dim=0),
        'image_ids': all_image_ids
    }
    
    print(f"Saving final dataset to {OUTPUT_FILE}...")
    torch.save(final_data, OUTPUT_FILE)
    
    # Clean up temp file if exists
    if os.path.exists(OUTPUT_FILE + ".tmp"):
        os.remove(OUTPUT_FILE + ".tmp")
        
    print(f"Successfully extracted and saved {len(all_image_ids)} embeddings!")

if __name__ == '__main__':
    # Fix for multiprocessing on Windows
    import multiprocessing
    multiprocessing.freeze_support()
    main()
