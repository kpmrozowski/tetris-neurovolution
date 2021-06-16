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
from torch.multiprocessing import Pool, Process, set_start_method
from src.deep_q_network import DeepQNetwork
from src.gen_algo import Population


def get_args():
    parser = argparse.ArgumentParser(
        """Implementation of Deep Q Network to play Tetris""")

    parser.add_argument("--width", type=int, default=10, help="The common width for all images")
    parser.add_argument("--height", type=int, default=20, help="The common height for all images")
    parser.add_argument("--block_size", type=int, default=30, help="Size of a block")
    parser.add_argument("--fps", type=int, default=300, help="frames per second")
    parser.add_argument("--saved_path", type=str, default="trained_models")
    parser.add_argument("--output", type=str, default="output.mp4")

    args = parser.parse_args()
    return args


def test(opt, conv1, conv2, conv3):
    if torch.cuda.is_available():
        torch.cuda.manual_seed(123)
    else:
        torch.manual_seed(123)
    # if torch.cuda.is_available():
    #     model = torch.load("{}/tetris".format(opt.saved_path))
    # else:
    #     model = torch.load("{}/tetris".format(opt.saved_path), map_location=lambda storage, loc: storage)

    model = DeepQNetwork()
    model.eval()
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
    if True: # load csv weights
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

    env = Tetris(width=opt.width, height=opt.height, block_size=opt.block_size)
    env.reset()
    if torch.cuda.is_available():
        model.cuda()
    out = cv2.VideoWriter(opt.output, cv2.VideoWriter_fourcc(*"MJPG"), opt.fps,
                          (int(1.5*opt.width*opt.block_size), opt.height*opt.block_size))
    while True:
        next_steps = env.get_next_states()
        next_actions, next_states = zip(*next_steps.items())
        next_states = torch.stack(next_states)
        if torch.cuda.is_available():
            next_states = next_states.cuda()
        predictions = model(next_states)[:, 0]
        index = torch.argmax(predictions).item()
        action = next_actions[index]
        result, done = env.step(action, render=True, video=out)

        if done:
            out.release()
            return result
        


if __name__ == "__main__":
    options = get_args()

    nn1 = np.genfromtxt('/Users/joannafrankiewicz/tetris/tetris-neurovolution/trained_models/conv1.csv', delimiter=',')
    nn2 = np.genfromtxt('/Users/joannafrankiewicz/tetris/tetris-neurovolution/trained_models/conv2.csv', delimiter=',')
    nn3 = np.genfromtxt('/Users/joannafrankiewicz/tetris/tetris-neurovolution/trained_models/conv3.csv', delimiter=',')
    try:
        set_start_method('spawn')  
        score = test(options, nn1, nn2, nn3)
        print('score = ', score)
    except RuntimeError:
        pass
    
