# based on code from https://stackabuse.com/minimax-and-alpha-beta-pruning-in-python
import numpy as np
import string
import time

class Game:
	MINIMAX = 0
	ALPHABETA = 1
	HUMAN = 2
	AI = 3
	
	#In-Game Notation
	CROSS  = '○' #'◦'
	CIRCLE = '●' #'•'
	BLOC   = 'X' #'☒' is too wide
	EMPTY  = '□' #'☐' is too wide

	def __init__(self,recommend = True, board_size=10,blocs=100,win_line_length=10):
		self.board_size = board_size
		self.blocs = blocs
		self.win_line_length = win_line_length

		self.initialize_game()
		self.recommend = recommend
		
		self.prev_move_y =0
		self.prev_move_x =0
		#generalised
	

	def board_size(self):
		return self.board_size

	#board_size set function
	def set_board_size(self, value):
		name = "board size"
		minimum = 3
		maximum = 10

		if type(value) is not int:
			raise TypeError(f"{name} expected an int, not a{type(value)._name_}!")
		if value > maximum:
			raise ValueError(f"{name} should be between {minimum} and {maximum},not{value}!")
		if value < minimum:
			raise ValueError(f"{name} should be between {minimum} and {maximum},not{value}!")
		else:
			self.board_size = value

	def blocs(self):
		return self.blocs
	
	def set_blocs(self,value):
		name = "number of blocs"
		minimum = 0
		maximum = self.board_size*2

		if type(value) is not int:
			raise TypeError(f"{name} expected an int, not a{type(value)._name_}!")
		if value > maximum:
			raise ValueError(f"{name} should be between {minimum} and {maximum},not{value}!")
		if value < minimum:
			raise ValueError(f"{name} should be between {minimum} and {maximum},not{value}!")
		else:
			self.blocs = value

	def win_line_length(self):
		return self.win_line_length

	def set_win_line_length(self,value):
		name="Lenght of the winning line"
		minimum = 3
		maximum = self.board_size

		if type(value) is not int:
			raise TypeError(f"{name} expected an int, not a{type(value)._name_}!")
		if value > maximum:
			raise ValueError(f"{name} should be between {minimum} and {maximum},not{value}!")
		if value < minimum:
			raise ValueError(f"{name} should be between {minimum} and {maximum},not{value}!")
		else:
			self.win_line_length = value

	def initialize_game(self):
		self.current_state =np.full((self.board_size,self.board_size),self.EMPTY,'str')
		
		# Player X always plays first
		self.player_turn = self.CROSS

	#done
	def draw_board(self):
		print(F"Game board of size {self.board_size}")
		for i in range(0, self.board_size):
			for j in range(0, self.board_size):
				print(F'{self.current_state[i][j]}', end="")
			print()
		print()

		
	#used kevin's code since the is_end function uses it(testing purposes)
	def remember_turn(self,x,y,notation):
		self.prev_move_x =x
		self.prev_move_y =y
		self.current_state[x][y] = notation

	def commit_turn(self,x,y,notation):
		if not self.is_valid(x,y):
			raise Exception(f"Player{self.player_turn} is disqualified for playing an illegal move.")
		self.current_state[x][y]
	
	#done
	def is_valid(self, px, py):
		if px < 0 or px >= self.board_size or py < 0 or py >= self.board_size :
			return False
		elif self.current_state[px][py] != self.EMPTY:
			return False
		else:
			return self.current_state[px][py] == self.EMPTY

	#to be modified THis is kevin's is_end function as mine did not work
	def is_end(self):
		def check_lines(win_line):
			if np.count_nonzero(win_line) <self.win_line_length:
				return False
			(unmarked,) = (~np.concatenate(([False],win_line,[False]))).nonzero()
			if (unmarked[1:]-unmarked[:-1]> self.win_line_length).any():
				return True

		for player in [self.CROSS,self.CIRCLE]:
			occupied_state = self.current_state == player

			if(check_lines(occupied_state[self.prev_move_y])):
				return player

			if(check_lines(np.diag(occupied_state,self.prev_move_x-self.prev_move_y))):
				return player
			
			if(check_lines(np.diag(np.fliplr(occupied_state),self.board_size-1-self.prev_move_x-self.prev_move_y))):
				return player

		if np.isin(self.EMPTY,self.current_state):
			return None

		return self.EMPTY
		

	#done
	def check_end(self):
		self.result = self.is_end()
		# Printing the appropriate message if the game has ended
		if self.result != None:
			if self.result == self.CROSS:
				print('The winner is X!')
			elif self.result == self.CIRCLE:
				print('The winner is O!')
			elif self.result == self.EMPTY:
				print("It's a tie!")
			self.initialize_game()
		return self.result

	#unchanged
	def input_move(self):
		while True:
			print(F'Player {self.player_turn}, enter your move:')
			px = int(input('enter the x coordinate: '))
			py = int(input('enter the y coordinate: '))
			if self.is_valid(px, py):
				return (px,py)
			else:
				print('The move is not valid! Try again.')
	
	#done
	def switch_player(self):
		if self.player_turn == self.CROSS:
			self.player_turn = self.CIRCLE
		elif self.player_turn == self.CIRCLE:
			self.player_turn = self.CROSS
		return self.player_turn

	#done
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
		elif result == self.CIRCLE:
			return (1, x, y)
		elif result == self.EMPTY:
			return (0, x, y)
		for i in range(0, self.board_size):
			for j in range(0, self.board_size):
				if self.current_state[i][j] == self.EMPTY:
					if max:
						self.current_state[i][j] = self.CIRCLE
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

	#done
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
		elif result == self.CIRCLE:
			return (1, x, y)
		elif result == self.EMPTY:
			return (0, x, y)
			#TO-DO
		for i, j in np.argwhere(self.current_state == self.EMPTY):
			if self.current_state[i][j] == self.EMPTY:
				if max:
					self.current_state[i][j] = self.CIRCLE
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

	#done
	def play(self,algo=None,player_x=None,player_o=None):
		if algo == None:
			algo = self.ALPHABETA
		if player_x == None:
			player_x = self.HUMAN
		if player_o == None:
			player_o = self.HUMAN
			
		size_board = int(input('enter the board_size:'))
		self.set_board_size(size_board)

		win_lenght = int(input('enter the win length requirement:'))
		self.set_win_line_length(win_lenght)

		num_blocs = int(input('enter the number of blocs:'))
		self.set_blocs(num_blocs)
		if(num_blocs >=1):
			self.player_turn = self.BLOC
			for i in range(0,num_blocs):
				(x,y) = self.input_move()
				self.current_state[x][y] = self.player_turn
			self.player_turn = self.CROSS

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
			if (self.player_turn == self.CROSS and player_x == self.HUMAN) or (self.player_turn == self.CIRCLE and player_o == self.HUMAN):
					if self.recommend:
						print(F'Evaluation time: {round(end - start, 7)}s')
						print(F'Recommended move: x = {x}, y = {y}')
					(x,y) = self.input_move()
			if (self.player_turn == self.CROSS and player_x == self.AI) or (self.player_turn == self.CIRCLE and player_o == self.AI):
						print(F'Evaluation time: {round(end - start, 7)}s')
						print(F'Player {self.player_turn} under AI control plays: x = {x}, y = {y}')
			self.current_state[x][y] = self.player_turn
			self.switch_player()

#unchanged
def main():
	g = Game(recommend=True)
	g.play(algo=Game.ALPHABETA,player_x=Game.AI,player_o=Game.AI)
	g.play(algo=Game.MINIMAX,player_x=Game.AI,player_o=Game.HUMAN)

#unchanged
if __name__ == "__main__":
	main()