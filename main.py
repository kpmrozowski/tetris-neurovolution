
from src.gen_algo import Population
from test import Test

if __name__ == "__main__":
    
    population = Population(50, None,  crossover_mode="mean", selection_mode="ranking")
    
    try:
        #set_start_method('spawn')  
        score = Test(nn1, nn2, nn3)
        print('score = ', score)
    except RuntimeError:
        pass