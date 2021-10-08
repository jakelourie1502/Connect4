import numpy as np 
from IPython.display import clear_output
from tabulate import tabulate
import copy
import time
import torch
from torch.nn import Linear, ReLU, Softmax, Sigmoid
# from functions import training_loop, run_episode, tensorfy_data, dataloader_func, train_system, pit_two_agents
from CONV_model_and_dataloader import *
from game_loop import create_board, is_valid_location, get_valid_locations, get_next_open_row, drop_piece, winning_move, turn, play_game
from agents import random_agent, Alpha_player, AZ_node, monte_carlo_player, simulation
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

model = Conv_Model()
model = torch.load('/Users/jacoblourie/RNN_games/Connect4/model_CPs/CONVmodel329.pt')
monty = monte_carlo_player(2000)
for i in range(10):
    board = create_board()
    play_game(board, agent_1='human',agent_2=monty, starting_player= (-1)**(i+1))
