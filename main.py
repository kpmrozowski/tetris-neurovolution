
from src.gen_algo import Population
from test import Test


if __name__ == "__main__":
    generation_count = 10
    population = Population(12, None,  crossover_mode="mean", selection_mode="ranking")

    for generation in range(generation_count):
        population = Population(12, population,  crossover_mode="mean", selection_mode="ranking")
    # try:
    #     #set_start_method('spawn')
    #     score = Test(nn1, nn2, nn3)
    #     print('score = ', score)
    # except RuntimeError:
    #     pass