import os
import json
import numpy as np
import matplotlib.pyplot as plt

def generate_km_data(num_curves=2):
    curves = []
    max_t = np.random.randint(50, 100)
    for _ in range(num_curves):
        t = [0]
        y = [1.0]
        curr_t = 0
        curr_y = 1.0
        while curr_t < max_t and curr_y > 0.05:
            step = np.random.randint(1, 10)
            curr_t += step
            drop = np.random.uniform(0.01, 0.1)
            curr_y = max(0.0, curr_y - drop)
            t.extend([curr_t, curr_t])
            y.extend([y[-1], curr_y])
        curves.append({"t": t, "y": y})
    return curves, max_t

def create_dataset(num_samples=50):
    os.makedirs("dataset/images", exist_ok=True)
    os.makedirs("dataset/ground_truth", exist_ok=True)
    
    for i in range(num_samples):
        plt.figure(figsize=(8, 6))
        num_curves = np.random.randint(1, 4)
        curves, max_t = generate_km_data(num_curves)
        
        gt_data = []
        for idx, curve in enumerate(curves):
            # plotting step uses step function, we already explicitly made it stepped
            plt.plot(curve["t"], curve["y"], drawstyle='steps-post', label=f'Cohort {idx+1}')
            
            # Save continuous points for ground truth metric
            curve_points = [[t_val, y_val] for t_val, y_val in zip(curve["t"], curve["y"])]
            gt_data.append({"cohort": f"Cohort {idx+1}", "points": curve_points})
        
        plt.title("Synthetic Kaplan-Meier Survival Curve")
        plt.xlabel("Time (Months)")
        plt.ylabel("Survival Probability")
        plt.xlim(0, max_t)
        plt.ylim(0, 1.05)
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        img_path = f"dataset/images/plot_{i:03d}.png"
        gt_path = f"dataset/ground_truth/plot_{i:03d}.json"
        
        plt.savefig(img_path, dpi=100)
        plt.close()
        
        with open(gt_path, 'w') as f:
            json.dump({"axes": [0, max_t, 0, 1.0], "curves": gt_data}, f)
            
    print(f"Generated {num_samples} KM plots.")

if __name__ == "__main__":
    create_dataset(50)
