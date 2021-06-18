from src.gen_algo import Population

if __name__ == "__main__":
    old_population = None
    elite_count = 3
    crossover_mode = "mean"
    selection_mode = "ranking"
    pop_size = 96
    n_workers = 6

    generation_count = 10000

    population = Population(old_population, elite_count, crossover_mode, selection_mode, 0, pop_size, n_workers)

    for generation_id in range(1, generation_count):
        population = Population(population, elite_count, crossover_mode, selection_mode, generation_id, pop_size, n_workers)
