import numpy as np 
from IPython.display import clear_output
from tabulate import tabulate
import copy
import time
import torch
from torch.nn import Linear, ReLU, Softmax, Sigmoid
from model_and_dataloader import Linear_Model, Dataset

def training_loop(model, training_epochs, games_per_epoch, mcts_search, batch_size, crit1, crit2, mini_epochs, cupt):
    candidate_mod = Linear_Model()
    candidate_mod.load_state_dict(model.state_dict())
    for i in range(training_epochs):
        m = 0
        episodes = {}
 
        for game in range(games_per_epoch):
            states, prob_vecs, rewards = [], [] ,[]
            states, prob_vecs, rewards= run_episode(mcts_search, candidate_mod, cupt, states, prob_vecs, rewards)
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

    candidate_agent = Alpha_player(simulations = mcts_searches, model = candidate_model, cupt=1)
    old_agent = Alpha_player(simulations = mcts_searches, model = model, cupt=1)
    score_v_monty = pit_two_agents(candidate_agent, magic_monty1, 4)
    print(f'Score v Monty: {np.mean(score_v_monty)}')
    score_v_old_agent = pit_two_agents(candidate_agent, magic_monty1, 5)
    print(f'Score v old agent: {np.mean(score_v_old_agent)}')
    return candidate_mod, score_v_monty, score_v_old_agent

def run_episode(mcts_searches,candidate_mod,cupt, states, prob_vecs, rewards, gamma=0.9):
    
    board = create_board()
    current_player = 1
    is_done = False
    
    while True:
        
        root = AZ_node(board, action= None, player=current_player, model=candidate_mod, cupt=cupt, gamma=gamma) #node with starting board
        
        for i in range(mcts_searches):
            node, reward = root.perform_mcts_search() #gets valid options frm board
            node.mcts_propagate(reward)
            
        #log 
        states.append(root.board.flatten() * root.player) #if current_player = -1, store the state as *= -1 so it's in first person mode.
        prob_vecs.append(root.sampled_prob_vector())
        rewards.append(current_player) #we can use this to multiply by the reward later
        
        #makemove
        col = np.argmax(root.sampled_prob_vector())
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
    model.train()
    for x, y_p, y_r in dataloader:
        x, y_p, y_r = x.float(), y_p.float(), y_r.float()
        probas, reward = model(x)
        loss1 = crit1(reward, y_r)
        loss2 = crit2(probas, y_p)
        loss = loss1 - loss2
        print(loss)
        loss.backward()
        optimizer.step(); optimizer.zero_grad()
    return model

def pit_two_agents(agent_1, agent_2, matches):
    winners = []
    for i in range(matches//2):
        board = create_board()
        w = play_game(board, agent_1, agent_2, printy= False)
        winners.append(w)
        board=create_board()
        w = play_game(board, agent_1, agent_2, printy= False, starting_player=-1)
        winners.append(w)
        
    return winners