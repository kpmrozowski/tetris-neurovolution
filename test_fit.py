"""
@author: Viet Nguyen <nhviet1009@gmail.com>
"""
import argparse
import torch
import cv2
import pandas as pd
import numpy as np
from src.tetris_slim import Tetris
import torch.nn as nn
from src.deep_q_network import DeepQNetwork
from torch.multiprocessing import Pool, Process, set_start_method

tetris_width = 10
tetris_height = 20
tetris_block_size = 30

def crossover_prepare(crossovers_in_queue, size, selected_ids, old_models, crossover_mode, model_id, models):
    queued_count = 0
    for k in range(model_id):
        queued_count += crossovers_in_queue[k]
    results = []
    for j in range(queued_count, queued_count + crossovers_in_queue[model_id]):
        models[model_id] = multicrossover(size, selected_ids, old_models, crossover_mode, model_id, models)

def multicrossover(size, selected_ids, old_models, crossover_mode, model_id, models):
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
            for i in range(conv_c[c_i][0].weight.size()[0]):
                for j in range(conv_c[c_i][0].weight.size()[1]):
                    if crossover_mode == "mean":
                        a = np.random.random()
                        conv_c[c_i][0].weight.data[i][j] = a * conv_b[c_i][0].weight.data[i][j] + (1 - a) * \
                            conv_a[c_i][0].weight.data[i][j]
                    if crossover_mode == "two_point":
                        point_one = np.random.random()
                        point_two = np.random.random()

                        if point_one > point_two:
                            a = point_one
                            point_one = point_two
                            point_two = a
                        # To jest zle: (conv_c[1][0].weight.data[0:0.12345][4]) ?
                        conv_c[c_i][0].weight.data[0:point_one][j] = conv_b[c_i][0].weight.data[0:point_one][j]
                        conv_c[c_i][0].weight.data[point_one:point_two][j] = \
                            conv_c[c_i][0].weight.data[point_one:point_two][j]
                        conv_c[c_i][0].weight.data[point_two:][j] = conv_b[c_i][0].weight.data[point_two:][j]
    print("cr-", model_id, end=" ", sep="")
    return model_c

def one_thread_workout(models, i, tests_in_queue, fitnesses):
    queued_count = 0
    for k in range(i):
        queued_count += tests_in_queue[k]
    results = []
    for j in range(queued_count, queued_count + tests_in_queue[i]):
        results.append(Test(models[j],j, fitnesses))
    print('paial_results:', results)
    return results


def Test(model, i, fitnesses):
    if torch.cuda.is_available():
        torch.cuda.manual_seed(123)
    else:
        torch.manual_seed(123)

    # model = DeepQNetwork()
    # model.eval()
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

    env = Tetris(width=tetris_width, height=tetris_height, block_size=tetris_block_size)
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
            print("{}. result={}".format(i, result))
            file_object = open('best_models/all_fitnesses.txt', 'a')
            file_object.write('{},{}\n'.format(i, result))
            file_object.close()
            fitnesses[i] = result
            return result
        
