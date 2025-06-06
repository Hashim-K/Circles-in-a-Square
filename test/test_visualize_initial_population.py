import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from evopy import EvoPy 
from evopy.strategy import Strategy


# Dummy fitness function for testing
def dummy_fitness(individual):
    return 0.0


def test_population_basic():
    num_circles = 10
    individual_length = 2 * num_circles
    population_size = 30
    square_bounds = (0,1)

    evopy = EvoPy(
        fitness_function=dummy_fitness,
        individual_length=individual_length,
        population_size=population_size,
        bounds=square_bounds,
        lasso_init=True,  # Set to False for original Strategy. True for Lasso Initialization
        strategy=Strategy.FULL_VARIANCE,
        random_seed=42,
    )

    population = evopy._init_population()

    cols = 6
    rows = int(np.ceil(population_size / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2.5, rows * 2.5))
    axes = axes.flatten()

    for idx, individual in enumerate(population):
        ax = axes[idx]
        coords = individual.genotype.reshape((-1, 2))
        x, y = coords[:, 0], coords[:, 1]
        ax.scatter(x, y, c='blue', s=20)

        rect = patches.Rectangle((0, 0), 1, 1, linewidth=1, edgecolor='black', facecolor='none')
        ax.add_patch(rect)

        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_title(f"Ind {idx}", fontsize=8)
        ax.set_aspect('equal')
        ax.axis('off')

    for ax in axes[len(population):]:
        ax.axis('off')

    plt.tight_layout()
    plt.suptitle("Population: Random initial individuals in square", y=1.02)
    plt.show()


if __name__ == "__main__":
    test_population_basic()
