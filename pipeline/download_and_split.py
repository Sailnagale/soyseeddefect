import os
import shutil
import random
from pathlib import Path
from roboflow import Roboflow

def execute_ingestion_pipeline(api_key, workspace, project, version_id):
    print("Initializing Roboflow connection...")
    rf = Roboflow(api_key=api_key)
    project_instance = rf.workspace(workspace).project(project)
    dataset = project_instance.version(version_id).download("yolov8")
    
    raw_dir = dataset.location
    output_root = "./dataset_root"
    
    class_mapping = {
        0: "Normal_Damage",        # bad seed
        1: "Testa cracking",       # crack
        2: "Normal_Damage",        # damage
        3: "pulse_beetle_damage",  # egg
        4: "Healthy",              # healthy
        6: "pulse_beetle_damage",  # hole
        7: "Normal_Damage"         # infected
    }
    
    grouped_pairs = {group: [] for group in set(class_mapping.values())}
    img_exts = {'.jpg', '.jpeg', '.png', '.bmp', '.JPG', '.JPEG', '.PNG'}
    
    for split in ['train', 'valid', 'test']:
        img_dir = os.path.join(raw_dir, split, 'images')
        lbl_dir = os.path.join(raw_dir, split, 'labels')
        if not os.path.exists(img_dir): continue
            
        for img_name in os.listdir(img_dir):
            img_path = os.path.join(img_dir, img_name)
            lbl_path = os.path.join(lbl_dir, f"{Path(img_name).stem}.txt")
            
            if Path(img_path).suffix in img_exts and os.path.exists(lbl_path):
                with open(lbl_path, 'r') as f:
                    line = f.readline().strip()
                    if line:
                        class_id = int(line.split()[0])
                        if class_id in class_mapping:
                            grouped_pairs[class_mapping[class_id]].append((img_path, lbl_path))
                            
    random.seed(42)
    for group, pairs in grouped_pairs.items():
        random.shuffle(pairs)
        n = len(pairs)
        if n == 0: continue
        
        train_idx = int(n * 0.80)
        val_idx = train_idx + int(n * 0.10)
        
        splits = {
            'train': pairs[:train_idx],
            'Validation': pairs[train_idx:val_idx],
            'test': pairs[val_idx:]
        }
        
        for split_name, current_pairs in splits.items():
            for img_p, lbl_p in current_pairs:
                dest_img = os.path.join(output_root, split_name, 'images', group, os.path.basename(img_p))
                dest_lbl = os.path.join(output_root, split_name, 'annotations', group, os.path.basename(lbl_p))
                os.makedirs(os.path.dirname(dest_img), exist_ok=True)
                os.makedirs(os.path.dirname(dest_lbl), exist_ok=True)
                shutil.copy2(img_p, dest_img)
                shutil.copy2(lbl_p, dest_lbl)
                
    if os.path.exists(raw_dir):
        shutil.rmtree(raw_dir)
    print(f"Dataset custom pipeline successfully created inside: {output_root}")

if __name__ == "__main__":
    execute_ingestion_pipeline(
        api_key="8A1mX7sGlQZwjtbfV6yL", 
        workspace="sail-bsj2k", 
        project="heisenberg-bxdzq", 
        version_id=3
    )
