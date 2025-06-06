import time
import pandas as pd
import matplotlib.pyplot as plt
from main import CirclesInASquare
import seaborn as sns

def benchmark_circles(n_circles_list, runs_per_config=2):
    results = []
    seeds = [42, 67, 45, 22, 77, 90, 122, 48, 15, 28, 65, 37]  # up to 12 total runs

    for n in n_circles_list:
        for run in range(runs_per_config):
            seed = seeds[run]

            runner = CirclesInASquare(n, output_statistics=True, random_seed=seed)

            start_time = time.time()
            target_min_dist = runner.get_target()
            best_genotype, report = runner.run_evolution_strategies()
            elapsed = time.time() - start_time


            results.append({
                "n_circles": n,
                "run": run,
                "seed": seed,
                "best_fitness": report.best_fitness,
                "evaluations": report.evaluations,
                "time_sec": elapsed,
                "optimal": report.optimal,
            })

            print(
                f"Finished: {n} circles, run {run}, seed {seed}, "
                f"fitness {report.best_fitness:.5f}, evals {report.evaluations}, time {elapsed:.2f}s"
            )

    return pd.DataFrame(results)



def plot_benchmark(df):
    # plt.figure(figsize=(12, 6))
    # sns.lineplot(data=df, x="n_circles", y="best_fitness", errorbar="sd", marker="o")
    # plt.title("Best Fitness vs. Number of Circles")
    # plt.ylabel("Best Fitness (avg ± std dev)")
    # plt.xlabel("Number of Circles")
    # plt.grid(True)
    # plt.show()
    #plt.savefig("Fitness.png")

    # plt.figure(figsize=(12, 6))
    # sns.lineplot(data=df, x="n_circles", y="time_sec", errorbar="sd", marker="o")
    # plt.title("Execution Time vs. Number of Circles")
    # plt.ylabel("Time (seconds)")
    # plt.xlabel("Number of Circles")
    # plt.grid(True)
    # plt.show()
    return

def print_summary_table(df):
    summary = df.groupby("n_circles").agg({
        "best_fitness": ["mean", "std"],
        "evaluations": ["mean", "std"],
        "time_sec": ["mean", "std"],
        "optimal": ["mean", "std"]
    })
    print(summary.round(4))


def run_benchmarks():
    n_circles_list = list(range(2, 11))  # Test from 2 to 10 circles
    results_df = benchmark_circles(n_circles_list)
    results_df.to_csv("benchmark_results.csv", index=False)

    # plot_benchmark(results_df)
    print_summary_table(results_df)


run_benchmarks()