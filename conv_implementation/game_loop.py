import numpy as np 
from IPython.display import clear_output
from tabulate import tabulate
import copy
import time

ROW_COUNT = 6
COLUMN_COUNT = 7
 
def create_board():
    board = np.zeros((6,7))
    return board
 
def drop_piece(board,row,col,piece):
    board[row][col]= piece

def get_valid_locations(board, ROW_COUNT,COLUMN_COUNT):
    top_row = board[ROW_COUNT-1]
    valid_acts = []
    for i in range(COLUMN_COUNT):
        if top_row[i] == 0:
            valid_acts.append(i)
    return valid_acts

def is_valid_location(board,col):
    return  col < COLUMN_COUNT and board[5][col]==0
 
def get_next_open_row(board,col):
    for r in range(ROW_COUNT):
        if board[r][col]==0:
            return r
    
def winning_move(board, piece):
    # Check horizontal locations for win
    for c in range(COLUMN_COUNT-3):
        for r in range(ROW_COUNT):
            if board[r][c] == piece and board[r][c+1] == piece and board[r][c+2] == piece and board[r][c+3] == piece:
                return True
 
    # Check vertical locations for win
    for c in range(COLUMN_COUNT):
        for r in range(ROW_COUNT-3):
            if board[r][c] == piece and board[r+1][c] == piece and board[r+2][c] == piece and board[r+3][c] == piece:
                return True
 
    # Check positively sloped diaganols
    for c in range(COLUMN_COUNT-3):
        for r in range(ROW_COUNT-3):
            if board[r][c] == piece and board[r+1][c+1] == piece and board[r+2][c+2] == piece and board[r+3][c+3] == piece:
                return True
 
    # Check negatively sloped diaganols
    for c in range(COLUMN_COUNT-3):
        for r in range(3, ROW_COUNT):
            if board[r][c] == piece and board[r-1][c+1] == piece and board[r-2][c+2] == piece and board[r-3][c+3] == piece:
                return True
    
    return False

def turn(player,board,agent='human',valid=True):
    
    if agent == 'human':
        if player == -1: name = 2
        else: name = 1
        
        if valid: 
            text = 'Make your Selection(0-6):'
        else: 
            text = 'Invalid choice, make new selection(0-6):'

        col = int(input(f"Player {name}, {text}"))
    
    else:
        valid_acts = get_valid_locations(board,ROW_COUNT=ROW_COUNT,COLUMN_COUNT = COLUMN_COUNT)
        col = agent.make_choice(board,valid_acts,player)

    if is_valid_location(board,col):
        row = get_next_open_row(board,col)
        drop_piece(board,row,col,player)    
        return col
    else: 
        turn(player,board,agent=agent, valid=False)

            
def play_game(init_board = np.zeros((ROW_COUNT,COLUMN_COUNT)), agent_1 = 'human',agent_2 = 'human',printy=True,starting_player=1):
    board = init_board
    
    if printy: 
        print(tabulate(np.flip(board,0)))
    game_over = False
    player = starting_player
    agent = agent_1
    
    while not game_over:
        if player == 1:
            agent = agent_1
        else:
            agent = agent_2
        
        t = turn(player,board, agent)
        
        if printy: 
            clear_output()
            printable_board = np.where(board==-1,2,board)
            print(tabulate(np.flip(printable_board,0)))
            print(f'Move played: {t}')

        if winning_move(board, player): 
            if printy: 
                print(f'player {player} won')
                time.sleep(5)
            game_over = True
            return player
        
        if board.all() != 0:
            game_over = True
            return 0
        player*=-1
        