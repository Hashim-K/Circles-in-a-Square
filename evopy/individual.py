import numpy as np

from evopy.strategy import Strategy
from evopy.utils import random_with_seed


class Individual:
    """The individual of the evolutionary strategy algorithm.

    This class handles the reproduction of the individual, using both the genotype and the specified
    strategy.

    For the full variance reproduction strategy, we adopt the implementation as described in:
    [1] Schwefel, Hans-Paul. (1995). Evolution Strategies I: Variants and their computational
        implementation. G. Winter, J. Perieaux, M. Gala, P. Cuesta (Eds.), Proceedings of Genetic
        Algorithms in Engineering and Computer Science, John Wiley & Sons.
    """
    _BETA = 0.0873
    _EPSILON = 0.01

    def __init__(self, genotype, strategy, strategy_parameters, bounds=None, random_seed=None):
        """Initialize the Individual.

        :param genotype: the genotype of the individual (1D numpy array)
        :param strategy: the strategy chosen to reproduce. See the Strategy enum for more
                         information
        :param strategy_parameters: the parameters required for the given strategy, as a list
        :param bounds: tuple (lower_bound, upper_bound) applied to each gene
        :param random_seed: seed or RandomState for reproducibility
        """
        self.genotype = np.array(genotype, dtype=float)
        self.length = len(self.genotype)
        self.random_seed = random_seed
        self.random = random_with_seed(self.random_seed)
        self.fitness = None
        self.constraint = None
        self.bounds = bounds
        self.strategy = strategy
        self.strategy_parameters = strategy_parameters
        if not isinstance(strategy, Strategy):
            raise ValueError("Provided strategy parameter was not an instance of Strategy.")
        # Choose the appropriate reproduce method based on strategy and parameter length
        if strategy == Strategy.SINGLE_VARIANCE and len(strategy_parameters) == 1:
            self.reproduce = self._reproduce_single_variance
        elif strategy == Strategy.MULTIPLE_VARIANCE and len(strategy_parameters) == self.length:
            self.reproduce = self._reproduce_multiple_variance
        elif strategy == Strategy.FULL_VARIANCE and len(strategy_parameters) == int(self.length * (self.length + 1) / 2):
            self.reproduce = self._reproduce_full_variance
        else:
            raise ValueError("The length of the strategy parameters was not correct.")

    def evaluate(self, fitness_function):
        """Evaluate the genotype of the individual using the provided fitness function."""
        self.fitness = fitness_function(self.genotype)
        return self.fitness

    def _clip_to_bounds(self, vec):
        """Helper: clip a 1D array 'vec' elementwise to [bounds[0], bounds[1]]."""
        lower, upper = self.bounds
        return np.clip(vec, lower, upper)

    def _reproduce_single_variance(self):
        """Single‐variance strategy with a simple clipping‐repair at the end."""
        # 1) Mutate genotype
        sigma = self.strategy_parameters[0]
        new_genotype = self.genotype + sigma * self.random.randn(self.length)

        # 2) Simple clip‐to‐bounds repair
        new_genotype = self._clip_to_bounds(new_genotype)

        # 3) Update sigma: one global learning step (Schwefel, 1995)
        scale_factor = self.random.randn() * np.sqrt(1 / (2 * self.length))
        new_sigma = max(sigma * np.exp(scale_factor), self._EPSILON)

        return Individual(
            genotype=new_genotype,
            strategy=self.strategy,
            strategy_parameters=[new_sigma],
            bounds=self.bounds,
            random_seed=self.random_seed  # carry over the same seed (or RandomState)
        )

    def _reproduce_multiple_variance(self):
        """Multiple‐variance strategy with a simple clipping‐repair at the end."""
        # 1) Mutate genotype using separate sigma_i for each gene
        sigmas = np.array(self.strategy_parameters, dtype=float)
        noise = self.random.randn(self.length)
        new_genotype = self.genotype + sigmas * noise

        # 2) Simple clip‐to‐bounds repair
        new_genotype = self._clip_to_bounds(new_genotype)

        # 3) Adapt each sigma_i:
        global_scale = self.random.randn() * np.sqrt(1 / (2 * self.length))
        local_scales = np.array([
            self.random.randn() * np.sqrt(1 / (2 * np.sqrt(self.length)))
            for _ in range(self.length)
        ])
        new_sigmas = np.maximum(sigmas * np.exp(global_scale + local_scales), self._EPSILON)

        return Individual(
            genotype=new_genotype,
            strategy=self.strategy,
            strategy_parameters=new_sigmas.tolist(),
            bounds=self.bounds,
            random_seed=self.random_seed
        )

    def _reproduce_full_variance(self):
        """Full‐variance strategy (including rotation angles) with simple clipping‐repair at the end."""
        # Number of variance parameters = length, number of rotation angles = length*(length-1)/2
        n = self.length
        num_variances = n
        num_rotations = int(n * (n - 1) / 2)

        # 1) Adapt variances σ_i
        current_variances = np.array(self.strategy_parameters[:num_variances], dtype=float)
        global_scale = self.random.randn() * np.sqrt(1 / (2 * n))
        local_scales = np.array([
            self.random.randn() * np.sqrt(1 / (2 * np.sqrt(n)))
            for _ in range(n)
        ])
        new_variances = np.maximum(current_variances * np.exp(global_scale + local_scales), self._EPSILON)

        # 2) Adapt rotation angles ϕ_j
        current_rotations = np.array(self.strategy_parameters[num_variances:], dtype=float)
        new_rotations = current_rotations + self.random.randn(num_rotations) * self._BETA
        # Wrap angles to [−π, π]
        new_rotations = np.where(
            np.abs(new_rotations) < np.pi,
            new_rotations,
            new_rotations - np.sign(new_rotations) * 2 * np.pi
        )

        # 3) Build the rotation matrix T from the new_rotations
        T = np.eye(n)
        idx = 0
        for p in range(n - 1):
            for q in range(p + 1, n):
                angle = new_rotations[idx]
                idx += 1
                Tp = np.eye(n)
                Tp[p, p] = Tp[q, q] = np.cos(angle)
                Tp[p, q] = -np.sin(angle)
                Tp[q, p] = np.sin(angle)
                T = T @ Tp

        # 4) Mutate genotype: sample from N(0, I), then apply T and scale by √variances
        z = self.random.randn(n)
        # Scale by sqrt of variances (diagonal matrix) before rotation
        D_sqrt = np.diag(np.sqrt(new_variances))
        new_genotype = self.genotype + T @ (D_sqrt @ z)

        # 5) Simple clip‐to‐bounds repair
        new_genotype = self._clip_to_bounds(new_genotype)

        # 6) Pack new strategy parameters back into a single list
        updated_parameters = np.concatenate([new_variances, new_rotations]).tolist()

        return Individual(
            genotype=new_genotype,
            strategy=self.strategy,
            strategy_parameters=updated_parameters,
            bounds=self.bounds,
            random_seed=self.random_seed
        )
