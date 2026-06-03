import os
import cv2
import numpy as np
from pathlib import Path

IMAGE_FOLDER = "./dataset_root/train/images/Healthy"
SUPPORTED_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.JPG', '.JPEG', '.PNG'}

def calculate_iqa_metrics(image_path):
    img = cv2.imread(str(image_path))
    if img is None: return None
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    brightness = np.mean(gray) / 255.0
    contrast = np.std(gray)
    laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    sharpness = (laplacian_var / 1000.0)
    
    mean_signal = np.mean(gray)
    std_noise = np.std(gray)
    snr_db = 20 * np.log10(mean_signal / std_noise) if std_noise > 0 else 0.0
        
    return brightness, contrast, sharpness, snr_db

def main():
    if not os.path.exists(IMAGE_FOLDER):
        print(f"Folder path '{IMAGE_FOLDER}' does not exist.")
        return

    brightness_list, contrast_list, sharpness_list, snr_list = [], [], [], []
    for root, _, files in os.walk(IMAGE_FOLDER):
        for file in files:
            file_path = Path(root) / file
            if file_path.suffix in SUPPORTED_EXTENSIONS:
                metrics = calculate_iqa_metrics(file_path)
                if metrics:
                    b, c, s, snr = metrics
                    brightness_list.append(b)
                    contrast_list.append(c)
                    sharpness_list.append(s)
                    snr_list.append(snr)

    if len(brightness_list) == 0: return
    
    print("=" * 85)
    print(f"{'METRIC':<15} | {'RANGE':<12} | {'MEAN VALUE (μ)':<16} | {'STANDARD DEV (σ)':<18}")
    print("=" * 85)
    for name, data in [("Brightness", brightness_list), ("Contrast", contrast_list), ("Sharpness", sharpness_list), ("SNR", snr_list)]:
        range_str = "[0, 1]" if name == "Brightness" else "[0, 255]" if name == "Contrast" else "x10^3" if name == "Sharpness" else "dB"
        print(f"{name:<15} | {range_str:<12} | {np.mean(data):<16.3f} | {np.std(data):<18.3f}")

if __name__ == "__main__":
    main()
