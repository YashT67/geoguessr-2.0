import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from tqdm import tqdm
from pathlib import Path
import warnings

warnings.filterwarnings("ignore")

class TestGeoImageDataset(Dataset):
    def __init__(self, image_paths, transform=None):
        self.image_paths = image_paths
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image_id = Path(image_path).stem
        
        try:
            image = Image.open(image_path).convert('RGB')
            if self.transform:
                image = self.transform(image)
        except Exception as e:
            print(f"Error loading image {image_path}: {e}")
            return None
            
        return {
            'image': image,
            'image_id': image_id
        }

def collate_fn(batch):
    batch = [b for b in batch if b is not None]
    if len(batch) == 0:
        return {}
    
    images = torch.stack([b['image'] for b in batch])
    image_ids = [b['image_id'] for b in batch]
    
    return {
        'images': images,
        'image_ids': image_ids
    }

def main():
    # Configuration
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    TEST_IMG_DIR = os.path.join(BASE_DIR, "photo_pull")
    OUTPUT_FILE = os.path.join(BASE_DIR, "full_test_extracted_features_dinov2_large.pt")
    
    BATCH_SIZE = 32
    NUM_WORKERS = 8 
    IMAGE_SIZE = 420
    
    print(f"Scanning test image directory {TEST_IMG_DIR}...")
    image_paths = list(Path(TEST_IMG_DIR).glob('*.jpg'))
    print(f"Total test images found: {len(image_paths)}")
    
    transform = transforms.Compose([
        transforms.Resize(IMAGE_SIZE, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ])
    
    dataset = TestGeoImageDataset(image_paths, transform=transform)
    dataloader = DataLoader(
        dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=False, 
        num_workers=NUM_WORKERS,
        collate_fn=collate_fn,
        pin_memory=True
    )
    
    print("Loading DINOv2 Large (dinov2_vitl14_reg) model...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitl14_reg')
    model = model.to(device)
    model.eval()
    
    all_features = []
    all_image_ids = []
    
    print("Starting extraction...")
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Extracting Test Embeddings"):
            if not batch:
                continue
                
            images = batch['images'].to(device)
            features = model(images)
            
            all_features.append(features.cpu())
            all_image_ids.extend(batch['image_ids'])

    print("Concatenating all features...")
    final_data = {
        'features': torch.cat(all_features, dim=0),
        'image_ids': all_image_ids
    }
    
    print(f"Saving test dataset to {OUTPUT_FILE}...")
    torch.save(final_data, OUTPUT_FILE)
    print(f"Successfully extracted and saved {len(all_image_ids)} test embeddings!")

if __name__ == '__main__':
    import multiprocessing
    multiprocessing.freeze_support()
    main()
