"""
@author: Viet Nguyen <nhviet1009@gmail.com>
"""
import argparse
import torch
import cv2
import pandas as pd
import numpy as np
from src.tetris import Tetris
import torch.nn as nn
from src.deep_q_network import DeepQNetwork
from torch.multiprocessing import Pool, Process, set_start_method

tetris_width = 10
tetris_height =20
tetris_block_size = 30


def Test(model, i):
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
            return result
        
