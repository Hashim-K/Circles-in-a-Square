"""Module used for the execution of the evolutionary algorithm."""
import time

import numpy as np

from evopy.individual import Individual
from evopy.progress_report import ProgressReport
from evopy.strategy import Strategy
from evopy.utils import random_with_seed

from evopy.initializers import init_points_with_offsets

class EvoPy:
    """Main class of the EvoPy package."""

    def __init__(self, fitness_function, individual_length, warm_start=None, generations=100,
                 population_size=100, num_children=5, mean=0, std=0.1, maximize=False,
                 strategy=Strategy.MULTIPLE_VARIANCE, random_seed=None, reporter=None,
                 target_fitness_value=None, target_tolerance=1e-2, max_run_time=None,
                 max_evaluations=None, bounds=None, lasso_init=False):
        """Initializes an EvoPy instance.

        :param fitness_function: the fitness function on which the individuals are evaluated
        :param individual_length: the length of each individual
        :param warm_start: the individual to start from
        :param generations: the number of generations to execute
        :param population_size: the population size of each generation
        :param num_children: the number of children generated per parent individual
        :param mean: the mean for sampling the random offsets of the initial population
        :param std: the standard deviation for sampling the random offsets of the initial population
        :param maximize: whether the fitness function should be maximized or minimized
        :param strategy: the strategy used to generate offspring by individuals. For more
                         information, check the Strategy enum
        :param random_seed: the seed to use for the random number generator
        :param reporter: callback to be invoked at each generation with a ProgressReport as argument
        :param target_fitness_value: target fitness value for early stopping
        :param target_tolerance: tolerance to within target fitness value is to be acquired
        :param max_run_time: maximum time allowed to run in seconds
        :param max_evaluations: maximum allowed number of fitness function evaluations
        :param bounds: bounds for the sampling the parameters of individuals
        """
        
        self.lasso_init = lasso_init
        
        self.fitness_function = fitness_function
        self.individual_length = individual_length
        self.warm_start = np.zeros(self.individual_length) if warm_start is None else warm_start
        self.generations = generations
        self.population_size = population_size
        self.num_children = num_children
        self.mean = mean
        self.std = std
        self.maximize = maximize
        self.strategy = strategy
        self.random_seed = random_seed
        self.random = random_with_seed(self.random_seed)
        self.reporter = reporter
        self.target_fitness_value = target_fitness_value
        self.target_tolerance = target_tolerance
        self.max_run_time = max_run_time
        self.max_evaluations = max_evaluations
        self.bounds = bounds
        self.evaluations = 0

    def _check_early_stop(self, start_time, best):
        """Check whether the algorithm can stop early, based on time and fitness target.

        :param start_time: the starting time to compare against
        :param best: the current best individual
        :return: whether the algorithm should be terminated early
        """

        if self.max_run_time is not None and (time.time() - start_time) > self.max_run_time:
            return 1
        elif self.target_fitness_value is not None and abs(best.fitness - self.target_fitness_value) < self.target_tolerance:
            return 2
        elif self.max_evaluations is not None and self.evaluations >= self.max_evaluations:
            return 3

    def run(self):
        """Run the evolutionary strategy algorithm.

        :return: the best genotype found
        """
        if self.individual_length == 0:
            return None

        start_time = time.time()

        decay_rate = 0.99
        annealing_on = False

        population = self._init_population()
        best = sorted(population, reverse=self.maximize,
                      key=lambda individual: individual.evaluate(self.fitness_function))[0]

        for generation in range(self.generations):
            children = [parent.reproduce() for _ in range(self.num_children)
                        for parent in population]
            
            # Apply annealing to strategy parameters
            if annealing_on:
                for child in children:
                    if self.strategy == Strategy.FULL_VARIANCE:
                        n = child.length
                        child.strategy_parameters[:n] = [
                            max(sigma * decay_rate, 0.001)
                            for sigma in child.strategy_parameters[:n]
                        ]
                    else:
                        child.strategy_parameters = [
                            max(sigma * decay_rate, 0.001)
                            for sigma in child.strategy_parameters
                        ]

            population = sorted(children, reverse=self.maximize,
                                key=lambda individual: individual.evaluate(self.fitness_function))
            self.evaluations += len(population)

            combined = population + children
            combined.sort(key=lambda ind: ind.fitness, reverse=self.maximize)
            population = combined[:self.population_size]


            # population = population[:self.population_size]
            best = population[0]

            if self.reporter is not None:
                mean = np.mean([x.fitness for x in population])
                std = np.std([x.fitness for x in population])
                self.reporter(ProgressReport(generation, self.evaluations, best.genotype, best.fitness, mean, std, False))

            stop = self._check_early_stop(start_time, best)

            if stop in [1, 2, 3]:
                if stop == 2 and self.reporter is not None:
                    mean = np.mean([x.fitness for x in population])
                    std = np.std([x.fitness for x in population])
                    self.reporter(
                        ProgressReport(generation, self.evaluations, best.genotype, best.fitness, mean, std, True))

                print("Stopped")
                break

        return best.genotype

    def _init_population(self):
        if self.strategy == Strategy.SINGLE_VARIANCE:
            strategy_parameters = self.random.randn(1)
        elif self.strategy == Strategy.MULTIPLE_VARIANCE:
            strategy_parameters = self.random.randn(self.individual_length)
        elif self.strategy == Strategy.FULL_VARIANCE:
            strategy_parameters = self.random.randn(
                int((self.individual_length + 1) * self.individual_length / 2))
        else:
            raise ValueError("Provided strategy parameter was not an instance of Strategy")
        population_parameters = np.asarray([
            self.warm_start + self.random.normal(loc=self.mean, scale=self.std, size=self.individual_length)
            for _ in range(self.population_size)
        ])
        
        if self.lasso_init:
            N = self.individual_length // 2
            lower_bound, upper_bound = self.bounds[0], self.bounds[1]

            _, best_points, _ = init_points_with_offsets(N, 6, bounds=(lower_bound, upper_bound))
            center = np.array([(lower_bound + upper_bound) / 2, (lower_bound + upper_bound) / 2])
            best_points = best_points.reshape((N, 2))

            flat_best = best_points.flatten()

            population = []

            population.append(
                Individual(
                    flat_best,
                    self.strategy,
                    strategy_parameters,
                    random_seed=self.random,
                    bounds=self.bounds
                )
            )

            max_dists = []
            std_devs = []

            for p in best_points:
                direction = p - center
                dist = np.linalg.norm(direction)
                if dist == 0:
                    max_dists.append(0)
                    std_devs.append(0)
                    continue
                direction_unit = direction / dist

                def max_dist_on_ray(center, direction_unit):
                    t_vals = []
                    for i in range(2):
                        if direction_unit[i] > 0:
                            t = (upper_bound - center[i]) / direction_unit[i]
                            t_vals.append(t)
                        elif direction_unit[i] < 0:
                            t = (lower_bound - center[i]) / direction_unit[i]
                            t_vals.append(t)
                    if not t_vals:
                        return 0
                    return min(t_vals)

                max_dist = max_dist_on_ray(center, direction_unit)
                max_dists.append(max_dist)

                std_dev = max(0, (max_dist - dist) / 3)
                std_devs.append(std_dev)

            max_dists = np.array(max_dists)
            std_devs = np.array(std_devs)

            for _ in range(1, self.population_size):
                new_points = []
                for i, p in enumerate(best_points):
                    direction = p - center
                    dist = np.linalg.norm(direction)
                    if dist == 0:
                        new_points.append(center)
                        continue
                    direction_unit = direction / dist
                    sampled_dist = self.random.normal(dist, std_devs[i])
                    sampled_dist = np.clip(sampled_dist, 0, max_dists[i])
                    new_point = center + direction_unit * sampled_dist
                    new_point = np.clip(new_point, lower_bound, upper_bound)
                    new_points.append(new_point)

                new_points = np.array(new_points)
                population.append(
                    Individual(
                        new_points.flatten(),
                        self.strategy,
                        strategy_parameters,
                        random_seed=self.random,
                        bounds=self.bounds
                    )
                )

            return population

        else:
        
        

            # Make sure parameters are within bounds
            if self.bounds is not None:
                oob_indices = (population_parameters < self.bounds[0]) | (population_parameters > self.bounds[1])
                population_parameters[oob_indices] = self.random.uniform(self.bounds[0], self.bounds[1], size=np.count_nonzero(oob_indices))

            return [
                Individual(
                    # Initialize genotype within possible bounds
                    parameters,
                    # Set strategy parameters
                    self.strategy, strategy_parameters,
                    # Set seed and bounds for reproduction
                    random_seed=self.random,
                    bounds=self.bounds
                ) for parameters in population_parameters
            ]
