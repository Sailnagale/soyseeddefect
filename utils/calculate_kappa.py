from sklearn.metrics import cohen_kappa_score

def calculate_metrics():
    # Placeholder validation instance log matching your evaluation criteria
    annotator_A = ["Egg", "Egg", "Hole", "Crack", "Crack", "Discolored", "Healthy", "Healthy", "Hole", "Egg"]
    annotator_B = ["Egg", "Egg", "Hole", "Crack", "Discolored", "Discolored", "Healthy", "Healthy", "Missed", "Egg"]
    
    classes = ["Egg", "Hole", "Crack", "Discolored", "Healthy"]
    print("=" * 60)
    print(f"{'TARGET DEFECT CLASS':<25} | {'KAPPA (κ)':<12} | {'STRENGTH'}")
    print("=" * 60)
    
    for c in classes:
        binary_A = [1 if x == c else 0 for x in annotator_A]
        binary_B = [1 if x == c else 0 for x in annotator_B]
        kappa = cohen_kappa_score(binary_A, binary_B)
        strength = "Near-Perfect" if kappa >= 0.81 else "Substantial" if kappa >= 0.61 else "Moderate"
        print(f"{c:<25} | {kappa:<12.3f} | {strength}")

if __name__ == "__main__":
    calculate_metrics()
