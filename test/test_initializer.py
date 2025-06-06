import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import matplotlib.pyplot as plt
import numpy as np
from evopy.initializers import init_points_with_offsets

if __name__ == "__main__":
    n_points = 5
    K = 6
    results, best_points, best_balance = init_points_with_offsets(n_points, K_offsets=K)

    fig, axes = plt.subplots(1, K, figsize=(4 * K, 4), constrained_layout=True)
    if K == 1:
        axes = [axes]

    for i, (shrunk_pts, balance, center, final_radius) in enumerate(results):
        ax = axes[i]
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_aspect('equal')
        ax.grid(True)
        ax.set_title(f"Offset {i+1}/{K}\nRadius = {balance:.4f}")

        # Draw dotted circle passing through final point positions
        dotted_circle = plt.Circle(center, final_radius, color='gray', linestyle='dotted', fill=False, alpha=0.7)
        ax.add_artist(dotted_circle)

        # Final point positions
        ax.scatter(shrunk_pts[:, 0], shrunk_pts[:, 1], color='blue')

        # Draw constraint circles
        for p in shrunk_pts:
            circle = plt.Circle(p, balance, color='red', alpha=0.3)
            ax.add_artist(circle)

    plt.show()
