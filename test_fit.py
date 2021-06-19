"""
@author: Viet Nguyen <nhviet1009@gmail.com>
"""
import argparse
import torch
import pandas as pd
import numpy as np
from src.tetris_slim import Tetris
import torch.nn as nn
import random

tetris_width = 10
tetris_height = 20
tetris_block_size = 30


def crossover_prepare(elite_count, crossovers_in_queue, size, selected_ids, old_models, crossover_mode, process_id, models, seed):
    # random.seed(seed)
    # torch.manual_seed(seed)
    # np.random.seed(seed)
    queued_count = 0
    for k in range(process_id):
        queued_count += crossovers_in_queue[k]
    for model_id in range(queued_count, queued_count + crossovers_in_queue[process_id]):
        if model_id >= elite_count:
            models[model_id] = multi_crossover(size, selected_ids, old_models, crossover_mode, model_id, seed,)


def multi_crossover(size, selected_ids, old_models, crossover_mode, model_id, seed):
    # random.seed(seed)
    # torch.manual_seed(seed)
    # np.random.seed(seed)
    # 3. Parents selection
    mother_id = np.random.randint(size)
    father_id = np.random.randint(size)

    model_a = old_models[selected_ids[mother_id]]
    model_b = old_models[selected_ids[father_id]]
    model_c = torch.load("trained_models/tetris")

    conv_a = [model_a.conv1, model_a.conv2, model_a.conv3]
    conv_b = [model_b.conv1, model_b.conv2, model_b.conv3]
    conv_c = [model_c.conv1, model_c.conv2, model_c.conv3]

    for c_i in range(len(conv_b)):
        # 4. Crossover
        for conv in range(3):
            for j in range(conv_c[c_i][0].weight.size()[1]):
                if crossover_mode == "mean":
                    for i in range(conv_c[c_i][0].weight.size()[0]):
                        a = np.random.random()
                        conv_c[c_i][0].weight.data[i][j] = a * conv_b[c_i][0].weight.data[i][j] + (1 - a) * \
                            conv_a[c_i][0].weight.data[i][j]
                if crossover_mode == "two_point":
                    point_one = np.random.randint(0, conv_c[c_i][0].weight.size()[0])
                    point_two = np.random.randint(0, conv_c[c_i][0].weight.size()[0])

                    if point_one > point_two:
                        a = point_one
                        point_one = point_two
                        point_two = a
                    conv_b_transpose = conv_b[c_i][0].weight.data.t()
                    conv_c[c_i][0].weight.data.t()[j][0:point_one] = conv_b_transpose[j][0:point_one].t()
                    conv_c[c_i][0].weight.data.t()[j][point_one:point_two] = \
                        conv_b_transpose[j][point_one:point_two].t()
                    conv_c[c_i][0].weight.data.t()[j][point_two:] = conv_b_transpose[j][point_two:].t()
    print(model_id, end=" ")
    if model_id % 25 == 0:
        print("")
    return model_c


def one_thread_workout(models, i, tests_in_queue, fitnesses, old_fitnesses, elite_to_skip, seed, games_per_evaluation):
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    queued_count = 0
    for k in range(i):
        queued_count += tests_in_queue[k]
    results = np.array([])
    for j in range(queued_count, queued_count + tests_in_queue[i]):
        fitnesses_to_mean = np.zeros(games_per_evaluation)
        if j < len(elite_to_skip):
            if elite_to_skip[j] == 1:
                results = np.append(results, old_fitnesses[j])
                fitnesses[j] = old_fitnesses[j]
            else:
                for game_id in range(games_per_evaluation):
                    fitnesses_to_mean[game_id] = test(models[j], seed + game_id)
                mean_fitness = np.mean(fitnesses_to_mean)
                results = np.append(results, mean_fitness)
                fitnesses[j] = mean_fitness
            print("{}. result={}, 'fitnesses_to_mean = {}".format(j, fitnesses[j], fitnesses_to_mean))
        else:
            for game_id in range(games_per_evaluation):
                fitnesses_to_mean[game_id] = test(models[j], seed + game_id)
            mean_fitness = np.mean(fitnesses_to_mean)
            results = np.append(results, mean_fitness)
            fitnesses[j] = mean_fitness
            print("{}. result={}, 'fitnesses_to_mean = {}".format(j, fitnesses[j], fitnesses_to_mean))
        file_object = open('best_models/all_fitnesses.txt', 'a')
        file_object.write('{},{}\n'.format(j, fitnesses[j]))
        file_object.close()
    #print('paial_fitnesseses:', results.astype(int))
    return fitnesses


def test(model, seed):
    if False: # save weights
        ii = 1
        for layer in model.modules():
            if isinstance(layer, nn.Linear):
                if ii == 1:
                    weights1 = layer.weight.cpu()
                    weights1 = weights1.detach().numpy()
                    pd.DataFrame(weights1).to_csv('trained_models/conv{}.csv'.format(ii))
                if ii == 2:
                    weights2 = layer.weight.cpu()
                    weights2 = weights2.detach().numpy()
                    pd.DataFrame(weights2).to_csv('trained_models/conv{}.csv'.format(ii))
                if ii == 3:
                    weights3 = layer.weight.cpu()
                    weights3 = weights3.detach().numpy()
                    pd.DataFrame(weights3).to_csv('trained_models/conv{}.csv'.format(ii))
                ii += 1
    if False: # load csv weights
        conv1 = np.genfromtxt('trained_models/conv1.csv', delimiter=',')
        conv2 = np.genfromtxt('trained_models/conv2.csv', delimiter=',')
        conv3 = np.genfromtxt('trained_models/conv3.csv', delimiter=',')
        ii = 1
        for layer in model.modules():
            if isinstance(layer, nn.Linear):
                with torch.no_grad():
                    if ii == 1:
                        layer.weight.data = torch.Tensor(conv1).cuda()
                    if ii == 2:
                        layer.weight.data = torch.Tensor(conv2).cuda()
                    if ii == 3:
                        layer.weight.data = torch.Tensor(conv3).cuda()
                    ii += 1

    env = Tetris(width=tetris_width, height=tetris_height, block_size=tetris_block_size, seed=seed)
    env.reset()
    if torch.cuda.is_available():
        model.cuda()
    while True:
        next_steps = env.get_next_states()
        next_actions, next_states = zip(*next_steps.items())
        next_states = torch.stack(next_states)
        if torch.cuda.is_available():
            next_states = next_states.cuda()
        with torch.no_grad():
            predictions = model(next_states)[:, 0]
        index = torch.argmax(predictions).item()
        action = next_actions[index]
        result, done = env.step(action, render=False)

        if done:
            return result
        
