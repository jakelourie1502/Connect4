import numpy as np 
from IPython.display import clear_output
from tabulate import tabulate
import copy
import time
import torch
from torch.nn import Linear, ReLU, Softmax, Sigmoid
from game_loop import *

class random_agent():
    def __init__(self,low=0,high=7,agent_name="random ronald"):
        self.name = agent_name
        self.low = low
        self.high = high
    def make_choice(self,board,valid_acts,player):
        return np.random.choice(valid_acts)

random_ronald = random_agent()
#monte carlo
class Node():
    def __init__(self,board, action, player, parent = None,actions=[0,1,2,3,4,5,6], done=False,cupt = 5, gamma = 0.9):
        
        self.board = copy.deepcopy(board)
        self.parent = parent
        self.action = action
        self.children = []
        self.n = 0
        self.player = player
        self.q_value = 0
        self.actions = actions
        self.done = done
        self.cupt = cupt
        self.gamma = gamma
        
    def is_root(self):
        return self.parent is None
    
    def select_best_leaf(self):

        if self.n == 0 or self.done:
            return self
        if len(self.children)==0:
            self.expand()
        best_child = self.children[np.argmax([x.ucb_value() for x in self.children])]
        return best_child.select_best_leaf()
         
        
    def ucb_value(self):
        #q values are for the player in that state, so the neg of the q value is the player taking the action that gets you there.
        return -self.q_value + (self.cupt/np.sqrt(2))*np.sqrt(2*np.log(self.parent.n))/(self.n+1e-5)
        
    def expand(self):
        for i in self.actions:
            board_copy = copy.deepcopy(self.board)
            if is_valid_location(board_copy,i):
                row = get_next_open_row(board_copy,i)
                drop_piece(board_copy,row,i,self.player) 
            done = winning_move(board_copy,self.player)
            child = Node(board=board_copy,action=i,player=self.player*-1,parent=self,done=done,cupt=self.cupt, gamma=self.gamma)
            self.children.append(child)

        #should expand and then end with 'select best_leaf'
    
    def propagate(self,winner):
        self.q_value = (self.q_value * self.n ) + winner*self.player
        self.n+=1
        self.q_value /= self.n
        if not self.is_root():
            self.parent.propagate(winner)

            
def simulation(board,player,iterations, actions=[0,1,2,3,4,5,6], cupt=1, gamma = 0.9):
    reward_log =[]
    starting_point = board
    game_over = False
    root = Node(board, action=None,player=player,parent=None,actions=actions, cupt=cupt, gamma = gamma)
    for i in range(iterations):
        #select the ubc recursion
        node = root
        node = node.select_best_leaf()
        boardy = copy.deepcopy(node.board)
        
        #check if that board is complete
        if node.done: 
            reward = -node.player

        elif boardy.all() != 0:
            reward = 0
        #play out game randomly
        else:
            reward = play_game(boardy,agent_1 = random_ronald, agent_2 = random_ronald, printy=False, starting_player=node.player) 
        node.propagate(reward)
    return root

class monte_carlo_player:
    def __init__(self,simulations, actions=7,gamma=0.9, cupt =5):
        self.simulations = simulations
        self.actions = actions
        self.gamma = gamma
        self.cupt = cupt
    def make_choice(self,board,actions_, player):
        available_options = []
        root = simulation(board, player, self.simulations, actions=actions_, cupt = self.cupt, gamma = self.gamma)
        action = actions_[np.argmin([x.q_value for x in root.children])]
        return action

#alpha zero
class AZ_node():
    def __init__(self, board, action, player, model, actions=7, parent=None, done=False, cupt=1, gamma =0.9, valid=True, agent_rand=0.05):
        self.n = 0
        self.board = board
        self.player = player
        self.parent = parent
        self.children = []
        self.action = action
        self.model = model
        self.prob_vector, self.exp_value = self.model((torch.from_numpy(board.flatten()*self.player)).float())
        self.prob_vector = self.prob_vector.detach().numpy()
        self.agent_rand = 0.05
        self.q_value = 0
        self.cupt = cupt
        self.actions = 7
        self.gamma = gamma
        self.done = done
        self.valid = valid

        
    def perform_mcts_search(self):
        if self.n == 0:
            return self,self.exp_value * self.player #multiply by self.player so it puts in right format

        if self.done:
            return self,-self.player
        

        if len(self.children)==0:
            self.expand()
        best_child = self.children[np.argmax([x.ucb_value() for x in self.children])]
        return best_child.perform_mcts_search()

    def ucb_value(self):
        if not self.valid: 
            return float('-inf')
        return np.random.normal(loc=1,scale=self.agent_rand)*( -self.q_value + ( (self.cupt * self.parent.prob_vector[self.action] * np.sqrt(self.parent.n)) / (1+self.n) ))

    def expand(self):
        for i in range(self.actions):
            board_copy = copy.deepcopy(self.board)
            if is_valid_location(board_copy,i):
                row = get_next_open_row(board_copy,i)
                drop_piece(board_copy,row,i,self.player) 
                won = winning_move(board_copy,self.player)
                child = AZ_node(board_copy, i, self.player*-1, model=self.model,parent=self,done=won, cupt=self.cupt, gamma=self.gamma)
            else:
                child = AZ_node(board_copy, i, self.player*-1, model=self.model,parent=self, done = False, cupt=self.cupt, gamma=self.gamma, valid =False)
            self.children.append(child)

    def mcts_propagate(self, reward):
        self.q_value = self.q_value*self.n + reward*self.player
        self.n+=1
        self.q_value = self.q_value / self.n
        if self.parent is not None:
            self.parent.mcts_propagate(reward)

    def sampled_prob_vector(self):
        return [x.n/self.n for x in self.children]

class Alpha_player():
    def __init__(self,model,simulations, actions=7,gamma=0.9, cupt =1):
        self.simulations = simulations
        self.actions = actions
        self.gamma = gamma
        self.cupt = cupt
        self.model = model
        
    def make_choice(self,board,actions_,player):
        
        root = AZ_node(board, model = self.model, player=player,action=None, cupt=self.cupt)
        for i in range(self.simulations):
            node, reward = root.perform_mcts_search() #gets valid options frm board
            node.mcts_propagate(reward)
        col = np.argmax(root.sampled_prob_vector())
        return col