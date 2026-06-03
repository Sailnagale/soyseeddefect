import os
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

DATASET_DIR = "./dataset_root"
OUTPUT_CHART = "dataset_distribution.png"

def verify_and_plot_distribution():
    splits = ['train', 'Validation', 'test']
    categories = ['images', 'annotations']
    counts = {split: defaultdict(lambda: {'images': 0, 'annotations': 0}) for split in splits}
    all_groups = set()

    if not os.path.exists(DATASET_DIR): return

    for split in splits:
        for cat in categories:
            cat_dir = os.path.join(DATASET_DIR, split, cat)
            if not os.path.exists(cat_dir): continue
            for group in os.listdir(cat_dir):
                if os.path.isdir(os.path.join(cat_dir, group)):
                    all_groups.add(group)
                    counts[split][group][cat] = len(os.listdir(os.path.join(cat_dir, group)))

    groups = sorted(list(all_groups))
    print("\n" + "="*75)
    print(f"{'SPLIT':<12} | {'STRICT CLASS GROUP':<26} | {'IMAGES':<10} | {'ANNOTATIONS':<12}")
    print("="*75)
    for split in splits:
        for group in groups:
            print(f"{split:<12} | {group:<26} | {counts[split][group]['images']:<10} | {counts[split][group]['annotations']:<12}")

    # Plot chart
    x = np.arange(len(groups))
    width = 0.22
    fig, ax = plt.subplots(figsize=(11, 6))
    ax.bar(x - width, [counts['train'][g]['images'] for g in groups], width, label='Train (80%)', color='#1f77b4')
    ax.bar(x, [counts['Validation'][g]['images'] for g in groups], width, label='Val (10%)', color='#ff7f0e')
    ax.bar(x + width, [counts['test'][g]['images'] for g in groups], width, label='Test (10%)', color='#2ca02c')
    ax.set_xticks(x)
    ax.set_xticklabels(groups, rotation=12, ha='right')
    ax.legend()
    plt.tight_layout()
    plt.savefig(OUTPUT_CHART, dpi=300)
    print(f"📈 Chart saved as {OUTPUT_CHART}")

if __name__ == "__main__":
    verify_and_plot_distribution()
