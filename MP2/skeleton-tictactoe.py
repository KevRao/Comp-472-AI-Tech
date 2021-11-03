# based on code from https://stackabuse.com/minimax-and-alpha-beta-pruning-in-python
import string
import time

import numpy as np

class Game:
	MINIMAX = 0
	ALPHABETA = 1
	HUMAN = 2
	AI = 3

	#In-Game Notation
	WHITE = '○' #'◦'
	BLACK = '●' #'•'
	BLOC  = '╳' #'☒' is too wide
	EMPTY = '□' #'☐' is too wide

	def __init__(self, recommend = True, board_size = 3, blocs_num = 0, coordinates = None, winning_line_length = 3):
		self.board_size = board_size
		self.blocs_num = blocs_num
		self.coordinates = coordinates
		self.winning_line_length = winning_line_length

		self.initialize_game()
		self.recommend = recommend

		self.initialize_formatting()

		#Dirty track
		self.prev_move_x = 0
		self.prev_move_y = 0

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
		#add Blocs
		for (i, j) in self.coordinates or []:
			self.current_state[i][j] = self.BLOC
		# Player X always plays first
		self.player_turn = self.WHITE

	def initialize_formatting(self):
		border_format = f"{{bar}}{{hcross}}{'{bar}{cross}' * (self.board_size - 1)}{{bar}}{{stop}}"
		header = f"   ║ {' │ '.join(string.ascii_uppercase[:self.board_size])} │"
		header_border	= border_format.format(bar="═══", hcross="╬", cross="╪", stop="╡")
		self.body_border = border_format.format(bar="───", hcross="╫", cross="┼", stop="┤")
		self.footer	  = border_format.format(bar="───", hcross="╨", cross="┴", stop="┘")
		self.header = f"{header}\n{header_border}"
		# header_border = f"═══╬{'═══╪'.join(['']*self.board_size)}═══╡"
		# body_border   = f"───╫{'───┼'.join(['']*self.board_size)}───┤"
		# footer_border = f"───╨{'───┴'.join(['']*self.board_size)}───┘"

	# Remember the cell played, since only the horizontal/vertical/diagonals of that cell needs to be check for the game end.
	def remember_turn(self, x, y, notation):
		self.prev_move_x = x
		self.prev_move_y = y
		self.current_state[x][y] = notation

	#When a move is committed, the AI can be disqualified if it provides an invalid move.
	def commit_turn(self, x, y, notation):
		# Humans should have a saving check beforehand.
		# Sch that only AI can do invalid move.
		if not self.is_valid(x, y):
			raise Exception(f"Player {self.player_turn} is disqualified for playing an illegal move.")
		self.remember_turn(x, y, notation)

	def draw_board(self):
		# Draw with borders.
		# inner .join is to concatenate cells of a row.
		# outer .join is to concatenate rows of the board.
		body = f'\n{self.body_border}\n'.join([f" {index} ║ {' │ '.join([cell for cell in row])} │" for index, row in enumerate(self.current_state)])
		print(f"\n{self.header}\n{body}\n{self.footer}\n")

	def is_valid(self, px, py):
		#invalid if it's a coordinate not on the board.
		if not (0 <= px < self.board_size and 0 <= py < self.board_size):
			return False
		#valid if empty
		return self.current_state[px][py] == self.EMPTY

	#Returns the winning player, a tie, otherwise None.
	def is_end(self):
		#Check if the given line contains enough consecutive True entries to win.
		def check_line(winnable_line):
			#cells must be booleans.
			#Line doesn't contain enough to win.
			if np.count_nonzero(winnable_line) < self._winning_line_length:
				return False
			# Check if space between unoccupied cells is greater than the winning length.
			# If there is, it must mean there are enough consecutive plays to win.
			(unmarked, ) = (~np.concatenate(([False], winnable_line, [False]))).nonzero()
			if (unmarked[1:]-unmarked[:-1] > self._winning_line_length).any():
				return True

		#check the state of each player.
		for player in [self.WHITE, self.BLACK]:
			#boolean matrix for where the player has played. organized by rows.
			occupied_state = self.current_state==player

			#check Horizontal win on row played.
			if(check_line(occupied_state[self.prev_move_y])):
				return player
			#check Vertical win on column played.
			if(check_line(occupied_state.T[self.prev_move_x])):
				return player
			#check Main diagonal win of played cell
			if(check_line(np.diag(occupied_state, self.prev_move_x - self.prev_move_y))):
				return player
			#check Anti diagonal win of played cell
			if(check_line(np.diag(np.fliplr(occupied_state), self.board_size - 1 - self.prev_move_x - self.prev_move_y))):
				return player

		# Is whole board not full?
		if np.isin(self.EMPTY, self.current_state):
			return None

		# It's a tie!
		return self.EMPTY

	#check if the game ended and return the result, no modification needed
	def check_end(self):
		self.result = self.is_end()
		# Printing the appropriate message if the game has ended
		if self.result != None:
			if self.result == self.WHITE:
				print('The winner is X!')
			elif self.result == self.BLACK:
				print('The winner is O!')
			elif self.result == self.EMPTY:
				print("It's a tie!")
			self.initialize_game()
		return self.result

	#player inputs his move, no modification needed
	def input_move(self):
		while True:
			print(F'Player {self.player_turn}, enter your move:')
			px = int(input('enter the x coordinate: '))
			py = int(input('enter the y coordinate: '))
			if self.is_valid(px, py):
				return (px,py)
			else:
				print('The move is not valid! Try again.')

	#switch the player, no modification needed
	def switch_player(self):
		if self.player_turn == self.WHITE:
			self.player_turn = self.BLACK
		elif self.player_turn == self.BLACK:
			self.player_turn = self.WHITE
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
		if result == self.WHITE:
			return (-1, x, y)
		elif result == self.BLACK:
			return (1, x, y)
		elif result == self.EMPTY:
			return (0, x, y)
		for i, j in np.argwhere(self.current_state == self.EMPTY):
			if max:
				self.current_state[i][j] = self.BLACK
				(v, _, _) = self.minimax(max=False)
				if v > value:
					value = v
					x = i
					y = j
			else:
				self.current_state[i][j] = self.WHITE
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
		if result == self.WHITE:
			return (-1, x, y)
		elif result == self.BLACK:
			return (1, x, y)
		elif result == self.EMPTY:
			return (0, x, y)
		for i, j in np.argwhere(self.current_state == self.EMPTY):
			if max:
				self.remember_turn(i, j, self.BLACK)
				(v, _, _) = self.alphabeta(alpha, beta, max=False)
				if v > value:
					value = v
					x = i
					y = j
			else:
				self.remember_turn(i, j, self.WHITE)
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
				if self.player_turn == self.WHITE:
					(_, x, y) = self.minimax(max=False)
				else:
					(_, x, y) = self.minimax(max=True)
			else: # algo == self.ALPHABETA
				if self.player_turn == self.WHITE:
					(m, x, y) = self.alphabeta(max=False)
				else:
					(m, x, y) = self.alphabeta(max=True)
			end = time.time()
			if (self.player_turn == self.WHITE and player_x == self.HUMAN) or (self.player_turn == self.BLACK and player_o == self.HUMAN):
				if self.recommend:
					print(F'Evaluation time: {round(end - start, 7)}s')
					print(F'Recommended move: x = {x}, y = {y}')
				(x,y) = self.input_move()
			if (self.player_turn == self.WHITE and player_x == self.AI) or (self.player_turn == self.BLACK and player_o == self.AI):
				print(F'Evaluation time: {round(end - start, 7)}s')
				print(F'Player {self.player_turn} under AI control plays: x = {x}, y = {y}')
			self.commit_turn(x, y, self.player_turn)
			self.switch_player()

def main():
	boardSize = int(input("Size of board: "))
	numBloc =  int(input("Number of blocs: "))
	coordinates = []
	for i in range(numBloc):
		#TODO: validation on inputs (optional)
		x_bloc = int(input(f"Enter the x{i+1}-coordinate of bloc position: "))
		y_bloc = int(input(f"Enter the y{i+1}-coordinate of bloc position: "))
		coordinates.append((x_bloc, y_bloc))
	winLine = int(input("Enter the number of winning line: "))
	g = Game(board_size = boardSize, blocs_num = numBloc, coordinates = coordinates, winning_line_length = winLine, recommend=True)
	g.draw_board()
	g.play(algo=Game.ALPHABETA,player_x=Game.AI,player_o=Game.AI)
	g.play(algo=Game.MINIMAX,player_x=Game.AI,player_o=Game.HUMAN)

if __name__ == "__main__":
	main()
