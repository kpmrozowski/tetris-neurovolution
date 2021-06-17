from multiprocessing import Process
from typing import List

import numpy as np
import torch
from torch.multiprocessing import Pool, Process, set_start_method

from test_fit import one_thread_workout, crossover_prepare
import pandas as pd

elitism_pct = 0.2
mutation_prob = 0.9
weights_mutate_power = 0.01
mutation_decrement = 0.96
device = 'cuda'

#Genetic algorithm

#1. Population
#2. Fitness function
#3. Parents selection
#4. Crossover
#5. Mutation



class Population:
    def __init__(
            self,
            old_population=None,
            crossover_mode="mean",
            selection_mode="ranking",
            generation_id=0,
            size=9,
            n_workers=8):
        self.size = size
        self.n_workers = n_workers
        self.generation_id = generation_id
        self.fitnesses = torch.zeros(self.size)

        self.in_queue = [np.floor_divide(self.size, self.n_workers) for _ in range(self.n_workers)]
        for i in range(np.remainder(self.size, self.n_workers)):
            self.in_queue[i] += 1

        if old_population is None:
            self.old_models = [torch.load("trained_models/tetris") for i in range(size)]
            self.models = [torch.load("trained_models/tetris") for i in range(size)]
            self.mutate()
            self.evaluate()
            # self.pool_test(size)
            # self.fitnesses = np.array([Test(self.models[i], i) for i in range(size)])
            self.backup()
        else:
            #1. Population
            self.old_models = old_population.models
            self.models = [torch.load("trained_models/tetris") for i in range(size)]
            self.old_fitnesses = old_population.fitnesses

            self.sort_ids = np.zeros(self.size)
            self.selected_ids = np.zeros(self.size)
            self.selection(selection_mode)
            self.crossover(crossover_mode)
            self.mutate()
            self.evaluate()
            # self.pool_test(size)
            # self.fitnesses = np.array([Test(self.models[i], i) for i in range(size)])
            self.succession()
            self.backup()

    def evaluate(self):
        print("Evaluating {} tetris neural-nets.......".format(self.size))
        set_start_method('spawn', force=True)
        processes: List[Process] = []
        self.fitnesses.share_memory_()
        for i in range(self.n_workers):
            p = Process(target=one_thread_workout, args=(self.old_models, i, self.in_queue, self.fitnesses, ))
            p.start()
            processes.append(p)
        for p in processes:
            p.join()
        pass

    def selection(self, selection_mode="ranking"):
        print("Selection")
        #selekcja rankingowa
        if selection_mode == "ranking":
            old_fitnesses = self.old_fitnesses.to().numpy()
            sum_fitnesses = np.sum(np.power(old_fitnesses, 2))
            probs = [np.power(old_fitnesses[i], 2) / sum_fitnesses for i in range(self.size)]

            self.sort_ids = np.argsort(probs)[::-1]
            print('\nall fitnesses: ', [old_fitnesses[i] for i in self.sort_ids])
            pd.DataFrame([old_fitnesses[i] for i in self.sort_ids]).to_csv('best_models/fitness_history{}.csv'.format(self.generation_id))
            best_model = self.old_models[self.sort_ids[0]]
            print('best model fitness: {}'.format(self.old_fitnesses[self.sort_ids[0]]))
            torch.save(best_model, "best_models/tetris_{}".format(self.generation_id))
            print("Crossver: done:", end=" ")

            for i in range(self.size):
                rand = np.random.rand()
                selected_id = -1
                while rand > 0:
                    selected_id += 1
                    rand -= probs[selected_id]
                self.selected_ids[i] = selected_id

        self.selected_ids = self.selected_ids.astype(int)



    def crossover(self, crossover_mode="mean"):
        for i in range(3):
            model_c = self.old_models[self.sort_ids[i]]
            self.models[i] = model_c

        set_start_method('spawn', force=True)
        processes: List[Process] = []
        for i in range(3, self.n_workers):
            p = Process(target=crossover_prepare, args=(self.in_queue, self.size, self.selected_ids,
                        self.old_models, crossover_mode, i, self.old_models, ))
            processes.append(p)
        for p in processes:
            p.start()
        for p in processes:
            p.join()
        pass

    def mutate(self):
        mutate_power = weights_mutate_power * mutation_decrement ** self.generation_id
        print("\nMutating, Power={}, Finished: ".format(mutate_power), end=" ")
        for i in range(3, self.size):
            print(i, end=" ")
            for conv in [self.models[i].conv1, self.models[i].conv2, self.models[i].conv3]:
                # for i in range(conv[0].weight.size()[0]):
                if np.random.random() < mutation_prob:
                    noise = torch.randn(1).mul_(mutate_power).to(device)
                    #add noise to each of neuron
                    conv[0].weight.data.add_(noise[0])
        print("")

    def succession(self):
        print("\nSuccession: worse ids:", end=" ")
        for i in range(self.size):
            if self.fitnesses[i].to().numpy() < self.old_fitnesses[i].to().numpy():
                self.models[i] = self.old_models[i]
                print(i, end=" ")

    def backup(self):
        for i in range(self.size):
            torch.save(self.models[i], "models_backup/tetris_backup_{}".format(i))
            pd.DataFrame(self.fitnesses.to().numpy()).to_csv('models_backup/fitnesses_backup.csv')




