import numpy as np 
from IPython.display import clear_output
from tabulate import tabulate
import copy
import time
import torch
from torch.nn import Linear, ReLU, Softmax, Sigmoid
# from functions import training_loop, run_episode, tensorfy_data, dataloader_func, train_system, pit_two_agents
from model_and_dataloader import Linear_Model, Dataset
from game_loop import create_board, is_valid_location, get_valid_locations, get_next_open_row, drop_piece, winning_move, turn, play_game
from agents import random_agent, Alpha_player, AZ_node, monte_carlo_player, simulation
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

def training_loop(model,base_model, training_epochs, games_per_epoch, mcts_search, batch_size, crit1, crit2, mini_epochs, cupt, rand_factor=0.8):
    candidate_mod = Linear_Model()
    candidate_mod.load_state_dict(model.state_dict())
    for i in range(training_epochs):
        m = 0
        episodes = {}
 
        for game in range(games_per_epoch):
            states, prob_vecs, rewards = [], [] ,[]
            states, prob_vecs, rewards= run_episode(mcts_search, candidate_mod, cupt, states, prob_vecs, rewards, rand_factor=rand_factor)
            m+=len(rewards)
            episodes[game] = {
                              'states':states,
                              'prob_vecs': prob_vecs,
                              'rewards':rewards,
                                'm': m
                             }
            
            
        x, y_probs, y_rewards = tensorfy_data(episodes,m)    
        dLoader = dataloader_func(x, y_probs, y_rewards, batch_size)
        candidate_mod = train_system(dLoader, candidate_mod, optimizer, crit1, crit2, mini_epochs)

    candidate_agent = Alpha_player(simulations = mcts_searches, model = candidate_mod, cupt=1)
    old_agent = Alpha_player(simulations = mcts_searches, model = base_model, cupt=1)
    score_v_monty = pit_two_agents(candidate_agent, magic_monty1, 4)
    
    print(f'Score v Monty: {score_v_monty}')
    score_v_old_agent = pit_two_agents(candidate_agent, old_agent, 4)
    print(f'Score v old agent: {score_v_old_agent}')
    return candidate_mod

def run_episode(mcts_searches,candidate_mod,cupt, states, prob_vecs, rewards, gamma=0.9,rand_factor=0.2):
    
    board = create_board()
    current_player = 1
    is_done = False
    
    while True:
        
        root = AZ_node(board, action= None, player=current_player, model=candidate_mod, cupt=cupt, gamma=gamma) #node with starting board
        
        for i in range(mcts_searches):
            node, reward = root.perform_mcts_search() 
            node.mcts_propagate(reward)
            
        #log 
        
        states.append(root.board.flatten() * root.player) #if current_player = -1, store the state as *= -1 so it's in first person mode.
        prob_vecs.append(root.sampled_prob_vector())
        rewards.append(current_player) #we can use this to multiply by the reward later
        print(board*root.player)
        #makemove
        if np.random.uniform(0,1) > rand_factor:
            col = np.argmax(root.sampled_prob_vector())
        else:
            opps = get_valid_locations(board, ROW_COUNT=6,COLUMN_COUNT= 7) 
            col = np.random.choice(opps)
        row = get_next_open_row(board,col)
        drop_piece(board,row,col,current_player)    
        
        #check win and add rewards
        if winning_move(board,current_player):
            reward = current_player
            
            rewards = [x*reward for x in rewards] 
            break
        if board.all() != 0:
            reward = 0
            rewards = [x*0 for x in rewards]
            break
        current_player*=-1
    print(rewards, prob_vecs,states)
    time.sleep(20)
    return states, prob_vecs, rewards

def tensorfy_data(episodes,m):
    x = torch.zeros((m,42))
    y_P = torch.zeros((m,7))
    y_R = torch.zeros((m,1))
    progress = 0 
    for value in episodes.values():
        x[progress:value['m']] = torch.from_numpy(np.array(value['states']))
        y_P[progress:value['m']] = torch.from_numpy(np.array(value['prob_vecs']))
        y_R[progress:value['m']] = torch.from_numpy(np.array(value['rewards']).reshape(-1,1))
        progress = value['m']
    return x, y_P, y_R

def dataloader_func(x, y_probs, y_rewards, batch_size):
    dataset = Dataset(x, y_probs, y_rewards)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,shuffle=True)
    return dataloader
    
def train_system(dataloader, model, optimizer, crit1, crit2, epochs):
    loss_count, mse_loss, bce_loss = 0,0,0
    n = 0
    model.train()
    for x, y_p, y_r in dataloader:
        x, y_p, y_r = x.float(), y_p.float(), y_r.float()
        probas, reward = model(x)
        loss1 = crit1(reward, y_r)
        
        loss2 = crit2(probas, y_p)
        loss = loss1 - loss2
        loss.backward()
        loss_count+=loss; mse_loss += loss1; bce_loss += loss2; n+= 1
        optimizer.step(); optimizer.zero_grad()
    print(f'Average loss {loss_count/n}, mse_loss: {mse_loss/n}, bce loss: {bce_loss/n}')
    print(f'total examples: {n}')
    return model

def pit_two_agents(agent_1, agent_2, matches):
    winners = []
    for i in range(matches//2):
        board = create_board()
        try:
            w = play_game(board, agent_1, agent_2, printy= False)
            winners.append(w)
        except: print(board)
        
        board=create_board()
        try:
            w = play_game(board, agent_1, agent_2, printy= False, starting_player=-1)
            winners.append(w)
        except:
            print(board)
        
        
    return winners

random_ronald = random_agent(0,7)
magic_monty1 = monte_carlo_player(1250,cupt=1)
model = Linear_Model()
# model = torch.load('/Users/jacoblourie/RNN_games/Connect4/model_CPs/model7.pt')
model_base = Linear_Model()
model_base.load_state_dict(model.state_dict())
optimizer = torch.optim.RMSprop(model.parameters(),lr=0.01)
# crit_1 = torch.nn.MSELoss()
def mse_loss(reward, y_r):
    return ((reward-y_r).T @ (reward-y_r))/reward.shape[0]
def prob_loss(probas, y_p):
    return (torch.sum(y_p * torch.log2(probas)))/probas.shape[0]
crit_1 = mse_loss
crit_2 = prob_loss

#HParameters
mcts_searches = 120
training_epochs = 10
games_per_epoch = 10
batch_size = 32
mini_epochs = 2
cupt = 2
loops = 50
for loop in range(loops):
    model = training_loop(model, model_base, training_epochs, games_per_epoch, mcts_searches, batch_size, crit_1, crit_2, mini_epochs, cupt)
    model_save_name = f'model{loop+1}.pt'
    path = f"/Users/jacoblourie/RNN_games/Connect4/model_CPs/{model_save_name}"
    torch.save(model, path)