import gc
from multiprocessing import Process
from typing import List

import numpy as np
import torch
from torch.multiprocessing import Pool, Process, set_start_method
from src.deep_q_network import DeepQNetwork

from test_fit import one_thread_workout, crossover_prepare
import pandas as pd

mutation_prob = 1.0
crossover_prob = 0.75
weights_mutate_power = 0.005
mutation_decrement = 0.9
tournament_size = 384
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

        # choose unique
        elite_diversal_buffer_ids = []
        last = torch.tensor(0.).to().int()
        for i in range(self.size):
            new = torch.round(self.old_fitnesses[i]).to().int()
            if new != last:
                elite_diversal_buffer_ids.append(i)
                last = torch.round(self.old_fitnesses[i]).to().int()
            else:
                continue
            if len(elite_diversal_buffer_ids) == self.elite_count:
                break

        for i in range(self.elite_count):
            if i < len(elite_diversal_buffer_ids):
                idx = elite_diversal_buffer_ids[i]
                self.old_models[i] = self.old_models[idx]
                self.old_fitnesses[i] = self.old_fitnesses[idx]
            self.models[i] = self.old_models[i]
            self.fitnesses[i] = self.old_fitnesses[i]
            if sort_ids[i] == i or np.round(self.old_fitnesses[i].to().numpy()) == round(old_fitnesses[i]):
                self.elite_to_skip[i] = 1
        if not backup:
            print('\nall fitnesses: ', np.round(self.old_fitnesses.to().numpy()))
            print('elite_to_skip: ', self.elite_to_skip)
            pd.DataFrame([self.old_fitnesses[i].to().numpy() for i in
                          range(self.size)]).to_csv('best_models/fitness_history{}.csv'.format(self.generation_id))
            best_model = self.old_models[0]
            print('best model fitness: {}'.format(self.old_fitnesses[0]))
            torch.save(best_model, "best_models/tetris_{}_{}".format(self.generation_id,
                                                                     self.old_fitnesses[0].to().numpy().astype(int)))

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
                rand = np.random.randint(0, self.size, tournament_size)
                idx = -1
                fitness = -1
                for element in rand:
                    actual_max_fitness = self.old_fitnesses[element].to().numpy()
                    if actual_max_fitness > fitness:
                        idx = element
                        fitness = actual_max_fitness
                        
                self.selected_ids[i] = idx
            self.selected_ids = self.selected_ids.astype(int)

    def crossover(self, crossover_mode="mean"):
        print("Crossver: done:", end=" ")
        if crossover_mode == "mean":
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

        if crossover_mode == "two_point":
            for i in range(self.size):
                mother_id = np.random.randint(self.size)
                father_id = np.random.randint(self.size)

                model_a = self.old_models[self.selected_ids[mother_id]]
                model_b = self.old_models[self.selected_ids[father_id]]
                model_c = torch.load("trained_models/tetris")

                conv_a = [model_a.conv1, model_a.conv2, model_a.conv3]
                conv_b = [model_b.conv1, model_b.conv2, model_b.conv3]
                conv_c = [model_c.conv1, model_c.conv2, model_c.conv3]
                for c_i in range(len(conv_b)):
                    # 4. Crossover
                    for conv in range(3):
                        for j in range(conv_c[c_i][0].weight.size()[1]):
                            cr_rand = np.random.random()
                            if crossover_prob < cr_rand:
                                point_one = np.random.randint(0, conv_c[c_i][0].weight.size()[0])
                                point_two = np.random.randint(0, conv_c[c_i][0].weight.size()[0])

                                if point_one > point_two:
                                    a = point_one
                                    point_one = point_two
                                    point_two = a
                                conv_b_transpose = conv_b[c_i][0].weight.data.t()
                                conv_a_transpose = conv_a[c_i][0].weight.data.t()
                                conv_c[c_i][0].weight.data.t()[j][0:point_one] = conv_b_transpose[j][0:point_one].t()
                                conv_c[c_i][0].weight.data.t()[j][point_one:point_two] = \
                                    conv_a_transpose[j][point_one:point_two].t()
                                conv_c[c_i][0].weight.data.t()[j][point_two:] = conv_b_transpose[j][point_two:].t()
                            else:
                                conv_c = conv_a
        print('seeds: ', np.random.get_state()[1][0:5])

    def mutate(self):
        # np.random.seed(self.seed_a)
        mutate_power = weights_mutate_power * mutation_decrement ** self.generation_id
        print("Mutating, Power={}, Finished: ".format(mutate_power), end=" ")
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
        for i in range(self.n_workers):
            queued_count = 0
            for k in range(i):
                queued_count += self.in_queue[k]
            models_part = []
            for j in range(queued_count, queued_count + self.in_queue[i]):
                models_part.append(self.models[j])
            p = Process(target=one_thread_workout, args=(models_part, i, self.in_queue, self.fitnesses,
                                                         self.elite_to_skip, self.seed_a, self.gpe))
            del models_part
            gc.collect()
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




