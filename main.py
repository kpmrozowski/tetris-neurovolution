from src.gen_algo import Population

if __name__ == "__main__":
    old_population = None
    crossover_mode = "mean"
    selection_mode = "ranking"
    generation_id = 0
    pop_size = 9
    n_workers = 8

    generation_count = 10000

    population = Population(old_population,  crossover_mode, selection_mode, generation_id, pop_size, n_workers)

    for generation_id in range(1, generation_count):
        population = Population(population, crossover_mode, selection_mode, generation_id, pop_size, n_workers)
