from multiprocessing import Process
from typing import List

import numpy as np
import torch
import torch.nn as nn
import functools
from torch.multiprocessing import Pool, Process, set_start_method


from src.deep_q_network import DeepQNetwork
from test import one_thread_workout
from test import Test
import pandas as pd

elitism_pct = 0.2
mutation_prob = 0.2
weights_mutate_power = 0.01

device = 'cuda'

#Genetic algorithm

#1. Population
#2. Fitness function
#3. Parents selection
#4. Crossover
#5. Mutation

class Population:
    def __init__(self, size=50, old_population=None, crossover_mode="mean", selection_mode="ranking", generation_id=0):
        self.size = size
        self.generation_id = generation_id
        self.fitnesses = np.zeros(self.size)
        if old_population is None:
            self.old_models = [torch.load("trained_models/tetris") for i in range(size)]
            self.models = [torch.load("trained_models/tetris") for i in range(size)]
            self.mutate()
            self.multi_test(size)
            # self.fitnesses = np.array([Test(self.models[i], i) for i in range(size)])
        else:
            #1. Population
            self.old_models = old_population.models
            self.models = []
            self.old_fitnesses = old_population.fitnesses

            self.crossover_mode = crossover_mode
            self.selection_mode = selection_mode
            self.crossover(crossover_mode, selection_mode)
            self.mutate()
            self.multi_test(size)
            # self.fitnesses = np.array([Test(self.models[i], i) for i in range(size)])

    def multi_test(self, size):
        set_start_method('spawn', force=True)
        processes: List[Process] = []
        for i in range(2):
            p = Process(target=one_thread_workout, args=(self.old_models, i, 2,))
            p.start()
            processes.append(p)
        for p in processes:
            p.join()
        # fitnesses = [item for sublist in p for item in sublist]
        fitnesses = []
        print(p)
        for sublist in p:
            for item in sublist:
                fitnesses.append(item)
        print(fitnesses)
        self.fitness = fitnesses


    def crossover(self,crossover_mode="mean", selection_mode="ranking"):
        print("Crossver")
        #2. Fitness
        sum_fitnesses = np.sum(self.old_fitnesses)
        probs = [self.old_fitnesses[i] / sum_fitnesses for i in
                 range(self.size)]

        # Sorting descending NNs according to their fitnesses
        #3. Parents selection
        sort_indices = np.argsort(probs)[::-1]
        best_model = self.old_models[sort_indices[0]]
        torch.save(best_model, "best_models/tetris_{}".format(self.generation_id))
        pd.DataFrame(self.old_fitnesses).to_csv('best_models/conv{}.csv'.format(self.generation_id))

        for i in range(self.size):
            if i < self.size * elitism_pct:
                # Add the top performing childs - parents selection
                model_c = self.old_models[sort_indices[i]]
                self.models.append(model_c)
            else:
                #selekcja rankingowa
                if selection_mode == "ranking":
                    a = sort_indices[0]
                    b = sort_indices[1]
                # sum_parent = self.old_fitnesses[a] + self.old_fitnesses[b]

                model_a, model_b = self.old_models[a], self.old_models[b]
                model_c = torch.load("trained_models/tetris")

                conv_a = [model_a.conv1, model_a.conv2, model_a.conv3]
                conv_b = [model_b.conv1, model_b.conv2, model_b.conv3]
                conv_c = [model_c.conv1, model_c.conv2, model_c.conv3]

                for c_i in range(len(conv_b)):
                    #4. Crossover
                    for conv in range(3):
                        for i in range(conv_c[c_i][0].weight.size()[0]):
                            for j in range(conv_c[c_i][0].weight.size()[1]):
                                if crossover_mode == "mean":
                                    a = np.random.random()
                                    conv_c[c_i][0].weight.data[i][j] = a*conv_b[c_i][0].weight.data[i][j] + (1-a)*conv_a[c_i][0].weight.data[i][j]
                                if crossover_mode == "two_point":
                                    point_one = np.random.random()
                                    point_two = np.random.random()

                                    if point_one > point_two:
                                        a = point_one
                                        point_one = point_two
                                        point_two = a

                                    conv_c[c_i][0].weight.data[0:point_one][j] = conv_b[c_i][0].weight.data[0:point_one][j]
                                    conv_c[c_i][0].weight.data[point_one:point_two][j] = conv_c[c_i][0].weight.data[point_one:point_two][j]
                                    conv_c[c_i][0].weight.data[point_two:][j] = conv_b[c_i][0].weight.data[point_two:][j]
                self.models.append(model_c)

    #5. Mutate
    def mutate(self):
        print("Mutating")
        for model in self.models:
            convs = [model.conv1, model.conv2, model.conv3]
            for conv in convs:
                # for i in range(conv[0].weight.size()[0]):
                if np.random.random() < mutation_prob:
                    noise = torch.randn(1).mul_(weights_mutate_power).to(device)
                    #add noise to each of neuron
                    conv[0].weight.data.add_(noise[0])


