
from src.gen_algo import Population
import numpy as np
import torch
from torch.multiprocessing import Pool, Process, set_start_method





if __name__ == "__main__":
    generation_count = 10000

    population = Population(None,  crossover_mode="mean", selection_mode="ranking", generation_id=0)

    for generation in range(1, generation_count):
        population = Population(population,  crossover_mode="mean", selection_mode="ranking", generation_id=generation)

    # try:
    #     #set_start_method('spawn')
    #     score = Test(nn1, nn2, nn3)
    #     print('score = ', score)
    # except RuntimeError:
    #     pass