import numpy as np 

def check_legal_move(game_board,coors):
    y, x = coors
    Y,X = game_board.shape[0], game_board.shape[1]
    if x not in range(X) or y not in range(Y): return False #is in the realm of the board?
    if game_board[y,x] != 0: return False #is this spot already taken?
    if y == Y-1: return True #is it the bottom
    elif game_board[y+1,x] != 0: return True #spot below it should be taken otherwise it can't go there
    else: return False
   
def player_plays_move(game_board, player_code, spot):
    x, y = spot
    if check_legal_move(game_board, (x,y)):
        game_board[x,y] = player_code
        return game_board
    else: 
        print('you fucked up and you cant go there you absolute spoon!')
        return game_board

def print_board(game_board):
    print(game_board)

def game_play(game_board, player_codes):
    print_board(game_board)
    for go in range(20):
        cur_player = 1 if go % 2 == 0 else 2
        y = int(input(f'player {cur_player}: pick row'))-1
        x = int(input(f'player {cur_player}: pick column'))-1
        game_board = player_plays_move(game_board, player_codes[cur_player-1], (y,x))
        print_board(game_board)

"""Game Loop"""
#set up the board
X, Y = 7, 7 #x refers to columns and goes from left to thright, y refers to rows and goes from top to bottom
game_board = np.zeros((Y,X))
game_play(game_board, (1,2))

