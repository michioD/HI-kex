import os
import glob
import torch
from ultralytics import YOLO

CACHE_DIR = "cache_yolo_results"
IMAGE_DIR = "datasets/coco_images/val2017/*.jpg"

def run_precompute():
    """Generates and saves YOLO predictions for all images if they do not exist."""
    os.makedirs(CACHE_DIR, exist_ok=True)
    
    image_paths = sorted(glob.glob(IMAGE_DIR))
    
    # Initialize models only when precomputing is necessary
    device = torch.device("mps")

    s_ml = YOLO('yolov8n.pt')
    l_ml = YOLO('yolov8x.pt')
    s_ml.to(device)
    l_ml.to(device)

    for t, img_path in enumerate(image_paths):
        print(f"computing image {t}: {img_path}")
        base_name = os.path.basename(img_path)
        cache_file = os.path.join(CACHE_DIR, f"{base_name}.pt")
        
        # Skip if this specific image has already been processed
        if os.path.exists(cache_file):
            continue
            
        sml_results = s_ml.predict(img_path, verbose=False)
        lml_results = l_ml.predict(img_path, verbose=False)
        
        # Save the results list directly to disk
        torch.save({
            'yolov8n_coco': sml_results,
            'yolov8x_coco': lml_results
        }, cache_file)



def check_cache():
    image_paths = sorted(glob.glob(IMAGE_DIR))
    if not os.path.exists(CACHE_DIR) or len(os.listdir(CACHE_DIR)) < len(image_paths):
        print("Cache incomplete or missing. Initializing precomputation...")
        run_precompute()
    
def get_cached_data(img_path):
    base_name = os.path.basename(img_path)
    cache_file = os.path.join(CACHE_DIR, f"{base_name}.pt")
    if not os.path.exists(cache_file):
        raise FileNotFoundError(f"Cache file not found for {img_path}. Please run precompute first.")
    return torch.load(cache_file, weights_only=False)