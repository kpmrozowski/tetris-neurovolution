from multiprocessing import Process
from typing import List

import numpy as np
import torch
from torch.multiprocessing import Pool, Process, set_start_method
from src.deep_q_network import DeepQNetwork

from test_fit import one_thread_workout, crossover_prepare
import pandas as pd

mutation_prob = 1.0
crossover_prob = 0.5
weights_mutate_power = 0.05
mutation_decrement = 0.95
tournament_size = 50
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
            elite_count=3,
            crossover_mode="mean",
            selection_mode="ranking",
            generation_id=0,
            size=9,
            seed_a=101,
            n_workers=8,
            games_per_evaluation=3):
        self.gpe = games_per_evaluation
        self.seed_a = seed_a
        np.random.seed(self.seed_a)
        self.size = size
        self.n_workers = n_workers
        self.generation_id = generation_id
        self.fitnesses = torch.zeros(self.size)
        self.elite_count = elite_count
        self.elite_to_skip = np.zeros(self.elite_count)
        self.selected_ids = np.zeros(self.size)
        self.old_fitnesses = torch.zeros(self.size)

        self.in_queue = [np.floor_divide(self.size, self.n_workers) for _ in range(self.n_workers)]
        for i in range(np.remainder(self.size, self.n_workers)):
            self.in_queue[i] += 1

        if old_population is None:
            if generation_id != 0:
                self.old_models = [torch.load("models_backup/tetris_backup_{}".format(i)) for i in range(size)]
                self.models = [torch.load("models_backup/tetris_backup_{}".format(i)) for i in range(size)]
                self.old_fitnesses = np.genfromtxt('models_backup/fitnesses_backup.csv', delimiter=',')
                self.old_fitnesses = torch.from_numpy(self.old_fitnesses)
                self.elite_to_skip = np.ones(self.elite_count)
                self.selection(selection_mode, backup=True)
                self.crossover(crossover_mode)
            else:
                # self.old_models = [torch.load("trained_models/tetris") for _ in range(size)]
                self.old_models = [DeepQNetwork() for _ in range(size)]
                # self.models = [torch.load("trained_models/tetris") for _ in range(size)]
                self.models = [DeepQNetwork() for _ in range(size)]
            self.mutate()
            self.evaluate()
            self.backup()
        else:
            #1. Population
            self.old_models = old_population.models
            self.models = [torch.load("trained_models/tetris") for _ in range(size)]
            #self.models = [DeepQNetwork() for _ in range(size)]
            self.old_fitnesses = old_population.fitnesses

            self.selection(selection_mode)
            self.crossover(crossover_mode)
            self.mutate()
            self.evaluate()
            # self.fitnesses = np.array([Test(self.models[i], i) for i in range(size)])
            # self.succession()
            self.backup()

    def selection(self, selection_mode="ranking", backup=False):
        np.random.seed(self.seed_a)
        print("Selection")
        old_fitnesses = self.old_fitnesses.to().numpy()
        sum_fitnesses = np.sum(np.power(old_fitnesses, 1))
        probs = np.array([np.power(old_fitnesses[i], 1) / sum_fitnesses for i in range(self.size)])

        sort_ids = np.argsort(probs)[::-1]
        old_models_sorted = [torch.load("trained_models/tetris") for _ in range(self.size)]
        old_fitnesses_sorted = torch.zeros(self.size)
        probs_sorted = probs
        for i in range(self.size):
            old_models_sorted[i] = self.old_models[sort_ids[i]]
            old_fitnesses_sorted[i] = self.old_fitnesses[sort_ids[i]]  # sorted self.old_fitnesses
            probs_sorted[i] = probs[sort_ids[i]]                            # sorted selection probabilities
        self.old_models = old_models_sorted
        self.old_fitnesses = old_fitnesses_sorted
        probs = probs_sorted
        old_fitnesses_unsorted = old_fitnesses
        old_fitnesses = self.old_fitnesses.to().numpy()    # sorted old_fitnesses

        for i in range(self.elite_count):
            self.models[i] = self.old_models[i]
            self.fitnesses[i] = self.old_fitnesses[i]
            if sort_ids[i] == i or round(old_fitnesses[i]) == round(old_fitnesses_unsorted[i]):
                self.elite_to_skip[i] = 1
        if not backup:
            print('\nall fitnesses: ', [old_fitnesses[i].astype(int)
                                        if i % 25 else print(old_fitnesses[i].astype(int)) for i in range(self.size)])
            pd.DataFrame([old_fitnesses[i] for i in
                          range(self.size)]).to_csv('best_models/fitness_history{}.csv'.format(self.generation_id))
            best_model = self.old_models[0]
            print('best model fitness: {}'.format(self.old_fitnesses[0]))
            torch.save(best_model, "best_models/tetris_{}_{}".format(self.generation_id,
                                                                     self.old_fitnesses.to().numpy()[0].astype(int)))

        if selection_mode == "ranking":
            for i in range(self.size):
                rand = np.random.rand()
                selected_id = -1
                while rand > 0:
                    selected_id += 1
                    rand -= probs[selected_id]
                self.selected_ids[i] = selected_id
            self.selected_ids = self.selected_ids.astype(int)

        if selection_mode == "tournament":
            for i in range(self.size):
                rand = np.random.randint(0, self.size, 5)
                idx = -1
                fitness = -1
                for element in rand:
                    actual_max_fitness = old_fitnesses[element]
                    if actual_max_fitness > fitness:
                        idx = element
                        fitness = actual_max_fitness
                        
                self.selected_ids[i] = idx
            self.selected_ids = self.selected_ids.astype(int)
        print('seeds: ', np.random.get_state()[1][0:5])

    def crossover(self, crossover_mode="mean"):
        print("Crossver: done:", end=" ")
        set_start_method('spawn', force=True)
        processes: List[Process] = []
        for i in range(self.n_workers):
            p = Process(target=crossover_prepare, args=(crossover_prob, self.elite_count, self.in_queue, self.size,
                        self.selected_ids, self.old_models, crossover_mode, i, self.old_models, self.seed_a, ))
            processes.append(p)
        for p in processes:
            p.start()
        for p in processes:
            p.join()
        print('seeds: ', np.random.get_state()[1][0:5])

    def mutate(self):
        # np.random.seed(self.seed_a)
        mutate_power = weights_mutate_power * mutation_decrement ** self.generation_id
        print("\nMutating, Power={}, Finished: ".format(mutate_power), end=" ")
        for i in range(self.elite_count, self.size):
            print(i, end=" ")
            if i % 25 == 0:
                print("")
            # torch.manual_seed(self.seed_a + i)
            # np.random.seed(self.seed_a + i)
            for conv in [self.models[i].conv1, self.models[i].conv2, self.models[i].conv3]:
                if np.random.random() < mutation_prob:
                    noise = torch.randn(1).mul_(mutate_power).to(device)
                    conv[0].weight.data.add_(noise[0])
        print('seeds: ', np.random.get_state()[1][0:5])
        print("")

    def evaluate(self):
        print("Evaluating {} tetris neural-nets.......".format(self.size))
        set_start_method('spawn', force=True)
        processes: List[Process] = []
        self.fitnesses.share_memory_()
        self.old_fitnesses.share_memory_()
        for i in range(self.n_workers):
            p = Process(target=one_thread_workout, args=(self.models, i, self.in_queue, self.fitnesses,
                                                         self.old_fitnesses, self.elite_to_skip, self.seed_a,
                                                         self.gpe))
            p.start()
            processes.append(p)
        for p in processes:
            p.join()
        np.random.seed(self.seed_a)

    def succession(self):
        print("\nSuccession: worse ids:", end=" ")
        for i in range(self.size):
            if self.fitnesses[i].to().numpy() < self.old_fitnesses[i].to().numpy():
                self.models[i] = self.old_models[i]
                print(i, end=" ")
                if i % 25 == 0:
                    print("")

    def backup(self):
        print("\nBackup: ", end=" ")
        for i in range(self.size):
            torch.save(self.models[i], "models_backup/tetris_backup_{}".format(i))
            pd.DataFrame(self.fitnesses.to().numpy()).to_csv('models_backup/fitnesses_backup.csv')
            print(i, end=" ")
            if i % 25 == 0:
                print("")




