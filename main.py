import numpy as np
import torch
import random

from src.gen_algo import Population

if __name__ == "__main__":
    seed = 123
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    else:
        torch.manual_seed(seed)
    np.random.seed(seed)
    old_population = None
    elite_count = 3
    crossover_mode = "two_point"
    # "mean"
    selection_mode = "ranking" # tournament
    pop_size = 96
    n_workers = 6
    games_per_evaluation = 3

    generation_count = 10000

    population = Population(old_population,
                            elite_count,
                            crossover_mode,
                            selection_mode,
                            0,
                            pop_size,
                            seed,
                            n_workers,
                            games_per_evaluation)

    for generation_id in range(1, generation_count):
        population = Population(population,
                                elite_count,
                                crossover_mode,
                                selection_mode,
                                generation_id,
                                pop_size,
                                seed,
                                n_workers,
                                games_per_evaluation,
                                )
