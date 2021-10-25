# based on code from https://stackabuse.com/minimax-and-alpha-beta-pruning-in-python

import time

import numpy as np

class Game:
    MINIMAX = 0
    ALPHABETA = 1
    HUMAN = 2
    AI = 3
    
    #In-Game Notation
    CROSS  = 'X'
    NOUGHT = 'O'
    EMPTY  = '.'
    
    def __init__(self, recommend = True, board_size = 3, blocs_num = 0, winning_line_length = 3):
        self.board_size = board_size
        self.blocs_num = blocs_num
        self.winning_line_length = winning_line_length
        
        
        self.initialize_game()
        self.recommend = recommend
    
    #Validation on game settings.
    #Validation by specification:
        # n in [3 .. 10]
        # b in [0 .. 2n]
        # s in [3 ..  n]
    @property
    def board_size(self):
        return self._board_size
    
    @board_size.setter
    def board_size(self, value):
        name = "Board size"
        minimum = 3
        maximum = 10
        
        if type(value) is not int:
            raise TypeError(f"{name} expects an int, not a {type(value).__name__}!")
        if minimum <= value <= maximum:
            self._board_size = value
        else:
            raise ValueError(f"{name} should be between {minimum} and {maximum}, not {value}!")
    
    @property
    def blocs_num(self):
        return self._blocs_num
    
    @blocs_num.setter
    def blocs_num(self, value):
        name = "Number of blocs"
        minimum = 0
        maximum = self.board_size * 2
        
        if type(value) is not int:
            raise TypeError(f"{name} expects an int, not a {type(value).__name__}!")
        if minimum <= value <= maximum:
            self._blocs_num = value
        else:
            raise ValueError(f"{name} should be between {minimum} and {maximum}(2 * board size), not {value}!")
    
    @property
    def winning_line_length(self):
        return self._winning_line_length
    
    @winning_line_length.setter
    def winning_line_length(self, value):
        name = "Winning line length"
        minimum = 3
        maximum = self.board_size
        
        if type(value) is not int:
            raise TypeError(f"{name} expects an int, not a {type(value).__name__}!")
        if minimum <= value <= maximum:
            self._winning_line_length = value
        else:
            raise ValueError(f"{name} should be between {minimum} and {maximum}(board size), not {value}!")
    
    
    def initialize_game(self):
        self.current_state = np.full((self.board_size , self.board_size ), self.EMPTY, 'str')
        # Player X always plays first
        self.player_turn = self.CROSS

    def draw_board(self):
        print()
        print('\n'.join([''.join([cell for cell in row]) for row in self.current_state]))
        print()
        
    def is_valid(self, px, py):
        #invalid if there's a coordinate not in the board.
        if not 0 < px < self.board_size or 0 < py < self.board_size:
            return False
        #invalid if 
        elif self.current_state[px][py] != self.EMPTY:
            return False
        else:
            return True

    def is_end(self):
        #TODO: convert this to general case.
            
        # for player in [self.CROSS, self.NOUGHT]:
            
        # Vertical win
        # for row in range(self.board_size):
        #     if self.current_state[]
        
        for i in range(0, 3):
            if (self.current_state[0][i] != self.EMPTY and
                self.current_state[0][i] == self.current_state[1][i] and
                self.current_state[1][i] == self.current_state[2][i]):
                return self.current_state[0][i]
        # Horizontal win
        for i in range(0, 3):
            if (np.all(self.current_state[i] == [self.CROSS] * self.board_size)):
                return self.CROSS
            elif (np.all(self.current_state[i] == [self.NOUGHT] * self.board_size)):
                return self.NOUGHT
        # Main diagonal win
        if (self.current_state[0][0] != self.EMPTY and
            self.current_state[0][0] == self.current_state[1][1] and
            self.current_state[0][0] == self.current_state[2][2]):
            return self.current_state[0][0]
        # Second diagonal win
        if (self.current_state[0][2] != self.EMPTY and
            self.current_state[0][2] == self.current_state[1][1] and
            self.current_state[0][2] == self.current_state[2][0]):
            return self.current_state[0][2]
        # Is whole board full?
        for i in range(0, 3):
            for j in range(0, 3):
                # There's an empty field, we continue the game
                if (self.current_state[i][j] == self.EMPTY):
                    return None
        # It's a tie!
        return self.EMPTY

    def check_end(self):
        self.result = self.is_end()
        # Printing the appropriate message if the game has ended
        if self.result != None:
            if self.result == self.CROSS:
                print('The winner is X!')
            elif self.result == self.NOUGHT:
                print('The winner is O!')
            elif self.result == self.EMPTY:
                print("It's a tie!")
            self.initialize_game()
        return self.result

    def input_move(self):
        while True:
            print(F'Player {self.player_turn}, enter your move:')
            px = int(input('enter the x coordinate: '))
            py = int(input('enter the y coordinate: '))
            if self.is_valid(px, py):
                return (px,py)
            else:
                print('The move is not valid! Try again.')

    def switch_player(self):
        if self.player_turn == self.CROSS:
            self.player_turn = self.NOUGHT
        elif self.player_turn == self.NOUGHT:
            self.player_turn = self.CROSS
        return self.player_turn

    def minimax(self, max=False):
        # Minimizing for 'X' and maximizing for 'O'
        # Possible values are:
        # -1 - win for 'X'
        # 0  - a tie
        # 1  - loss for 'X'
        # We're initially setting it to 2 or -2 as worse than the worst case:
        value = 2
        if max:
            value = -2
        x = None
        y = None
        result = self.is_end()
        if result == self.CROSS:
            return (-1, x, y)
        elif result == self.NOUGHT:
            return (1, x, y)
        elif result == self.EMPTY:
            return (0, x, y)
        for i in range(0, 3):
            for j in range(0, 3):
                if self.current_state[i][j] == self.EMPTY:
                    if max:
                        self.current_state[i][j] = self.NOUGHT
                        (v, _, _) = self.minimax(max=False)
                        if v > value:
                            value = v
                            x = i
                            y = j
                    else:
                        self.current_state[i][j] = self.CROSS
                        (v, _, _) = self.minimax(max=True)
                        if v < value:
                            value = v
                            x = i
                            y = j
                    self.current_state[i][j] = self.EMPTY
        return (value, x, y)

    def alphabeta(self, alpha=-2, beta=2, max=False):
        # Minimizing for 'X' and maximizing for 'O'
        # Possible values are:
        # -1 - win for 'X'
        # 0  - a tie
        # 1  - loss for 'X'
        # We're initially setting it to 2 or -2 as worse than the worst case:
        value = 2
        if max:
            value = -2
        x = None
        y = None
        result = self.is_end()
        if result == self.CROSS:
            return (-1, x, y)
        elif result == self.NOUGHT:
            return (1, x, y)
        elif result == self.EMPTY:
            return (0, x, y)
        for i in range(0, 3):
            for j in range(0, 3):
                if self.current_state[i][j] == self.EMPTY:
                    if max:
                        self.current_state[i][j] = self.NOUGHT
                        (v, _, _) = self.alphabeta(alpha, beta, max=False)
                        if v > value:
                            value = v
                            x = i
                            y = j
                    else:
                        self.current_state[i][j] = self.CROSS
                        (v, _, _) = self.alphabeta(alpha, beta, max=True)
                        if v < value:
                            value = v
                            x = i
                            y = j
                    self.current_state[i][j] = self.EMPTY
                    if max: 
                        if value >= beta:
                            return (value, x, y)
                        if value > alpha:
                            alpha = value
                    else:
                        if value <= alpha:
                            return (value, x, y)
                        if value < beta:
                            beta = value
        return (value, x, y)

    def play(self,algo=None,player_x=None,player_o=None):
        if algo == None:
            algo = self.ALPHABETA
        if player_x == None:
            player_x = self.HUMAN
        if player_o == None:
            player_o = self.HUMAN
        while True:
            self.draw_board()
            if self.check_end():
                return
            start = time.time()
            if algo == self.MINIMAX:
                if self.player_turn == self.CROSS:
                    (_, x, y) = self.minimax(max=False)
                else:
                    (_, x, y) = self.minimax(max=True)
            else: # algo == self.ALPHABETA
                if self.player_turn == self.CROSS:
                    (m, x, y) = self.alphabeta(max=False)
                else:
                    (m, x, y) = self.alphabeta(max=True)
            end = time.time()
            if (self.player_turn == self.CROSS and player_x == self.HUMAN) or (self.player_turn == self.NOUGHT and player_o == self.HUMAN):
                if self.recommend:
                    print(F'Evaluation time: {round(end - start, 7)}s')
                    print(F'Recommended move: x = {x}, y = {y}')
                (x,y) = self.input_move()
            if (self.player_turn == self.CROSS and player_x == self.AI) or (self.player_turn == self.NOUGHT and player_o == self.AI):
                print(F'Evaluation time: {round(end - start, 7)}s')
                print(F'Player {self.player_turn} under AI control plays: x = {x}, y = {y}')
            self.current_state[x][y] = self.player_turn
            self.switch_player()

def main():
    g = Game(recommend=True)
    g.play(algo=Game.ALPHABETA,player_x=Game.AI,player_o=Game.AI)
    g.play(algo=Game.MINIMAX,player_x=Game.AI,player_o=Game.HUMAN)

if __name__ == "__main__":
    main()

