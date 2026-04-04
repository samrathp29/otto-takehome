import os
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def generate_adversarial_dataset(num_samples=20):
    """
    Generate adversarial KM plots with difficult edge cases:
    - Overlapping curves (3-4 curves with similar trajectories)
    - Truncated y-axis (0.2 to 1.0 instead of 0.0 to 1.0)
    - Low DPI (50)
    - Grayscale-only palettes
    """
    os.makedirs("data/adversarial/images", exist_ok=True)
    os.makedirs("data/adversarial/ground_truth", exist_ok=True)

    for i in range(num_samples):
        fig, ax = plt.subplots(figsize=(8, 6))
        num_curves = np.random.randint(2, 5)  # 2-4 overlapping curves
        max_t = np.random.randint(50, 100)

        # Decide edge case type
        edge_type = i % 4
        truncated_y = edge_type == 0  # Every 4th: truncated y-axis
        grayscale = edge_type == 1    # Every 4th: grayscale palette
        low_dpi = edge_type == 2      # Every 4th: low DPI
        heavy_overlap = edge_type == 3  # Every 4th: tightly overlapping curves

        gt_data = {"axes": [0, max_t, 0.0, 1.0], "curves": []}

        # Generate base trajectory (all curves will be close to this if heavy_overlap)
        base_drops = np.random.uniform(0.02, 0.08, size=20)

        gray_colors = ['#333333', '#666666', '#999999', '#AAAAAA']
        color_idx = 0

        for c in range(num_curves):
            t = [0]
            y = [1.0]
            curr_t = 0
            curr_y = 1.0

            drop_idx = 0
            while curr_t < max_t and curr_y > 0.05:
                step = np.random.randint(1, 8)
                curr_t += step
                if heavy_overlap:
                    drop = base_drops[drop_idx % len(base_drops)] + np.random.uniform(-0.01, 0.01)
                else:
                    drop = np.random.uniform(0.01, 0.1)
                drop_idx += 1
                curr_y = max(0.0, curr_y - drop)
                t.extend([curr_t, curr_t])
                y.extend([y[-1], curr_y])

            color = gray_colors[color_idx % len(gray_colors)] if grayscale else None
            color_idx += 1
            if color:
                ax.plot(t, y, drawstyle='steps-post', label=f'Cohort {c+1}', color=color)
            else:
                ax.plot(t, y, drawstyle='steps-post', label=f'Cohort {c+1}')

            curve_points = [[tv, yv] for tv, yv in zip(t, y)]
            gt_data["curves"].append({"cohort": f"Cohort {c+1}", "points": curve_points})

        ax.set_title("Kaplan-Meier Survival Curve")
        ax.set_xlabel("Time (Months)")
        ax.set_ylabel("Survival Probability")
        ax.set_xlim(0, max_t)

        if truncated_y:
            ax.set_ylim(0.2, 1.05)
            gt_data["axes"][2] = 0.2
        else:
            ax.set_ylim(0, 1.05)

        ax.legend()
        ax.grid(True, alpha=0.3)
        fig.tight_layout()

        dpi = 50 if low_dpi else 100
        img_path = f"data/adversarial/images/adv_{i:03d}.png"
        gt_path = f"data/adversarial/ground_truth/adv_{i:03d}.json"

        fig.savefig(img_path, dpi=dpi)
        plt.close(fig)

        with open(gt_path, 'w') as f:
            json.dump(gt_data, f)

    print(f"Generated {num_samples} adversarial KM plots.")


if __name__ == "__main__":
    generate_adversarial_dataset(20)
