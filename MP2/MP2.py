# based on code from https://stackabuse.com/minimax-and-alpha-beta-pruning-in-python

import time



class Game:
	MINIMAX = 0
	ALPHABETA = 1
	HUMAN = 2
	AI = 3
	

	#modified
	def __init__(self, boardSize, numBloc, coordinates, winLine, recommend = True):
		self.boardSize = boardSize
		self.numBloc = numBloc
		self.coordinates = coordinates
		for i in self.coordinates:
			print(i)
		self.winLine = winLine
		self.initialize_game()
		self.recommend = recommend

	#modified	
	def initialize_game(self):    
		self.current_state = [[0] *self.boardSize] *self.boardSize
		for i in range(0, self.boardSize):
			for j in range(0, self.boardSize):
				for bloc in self.coordinates:
					if(bloc == (i,j)):
						self.current_state[i][j] = 'b'
					else:
						self.current_state[i][j] = '.'

							#self.current_state = [['.'] *self.boardSize] *self.boardSize
		# Player X always plays first
		self.player_turn = 'X'

	#draw the starting board, modified
	def draw_board(self):
		print()
		for y in range(0, self.boardSize):
			for x in range(0, self.boardSize):
				print(F'{self.current_state[x][y]}', end="")
			print()
		print()

	#check the coordinates given by user, modified	
	def is_valid(self, px, py):
		if px < 0 or px > (self.boardSize - 1) or py < 0 or py > (self.boardSize - 1):
			return False
		elif self.current_state[px][py] != '.':
			return False
		else:
			return True

	#modified
	def is_end(self):
		countLine = 0 
		# Vertical win
		for i in range(0, self.boardSize):
			for n in range(0, self.boardSize - 1): #unsure!!!!!
				if (self.current_state[n][i] != '.' and
					self.current_state[n][i] == self.current_state[n+1][i]): 
					#and
					#self.current_state[1][i] == self.current_state[2][i]):
					countLine += 1
					if(countLine == self.winLine):
						return self.current_state[0][i]
		# Horizontal win
		for i in range(0, self.boardSize):
			if (self.current_state[i] == (['X']*self.winLine)):
				return 'X'
			elif (self.current_state[i] == (['O']*self.winLine)):
				return 'O'
		# Main diagonal win, UNSUREEE!
		for i in range(0, self.boardSize - 1):
			if (self.current_state[i][i] != '.' and
				self.current_state[i][i] == self.current_state[i+1][i+1]): # and
				#self.current_state[0][0] == self.current_state[2][2]):
				countLine += 1
				if(countLine == self.winLine):
					return self.current_state[0][0]
		# Second diagonal win, UNSUREEEE!!!
		for i in range(self.boardSize - 1, 0, -1):
			if (self.current_state[(self.boardSize - 1) - i][i] != '.' and
				self.current_state[(self.boardSize - 1) - i][i] == self.current_state[self.boardSize - i][i - 1]): #and
				#self.current_state[0][2] == self.current_state[2][0]):
				countLine += 1
				if(countLine == self.winLine):
					return self.current_state[0][self.boardSize - 1]
		# Is whole board full?
		for i in range(0, self.boardSize):
			for j in range(0, self.boardSize):
				# There's an empty field, we continue the game
				if (self.current_state[i][j] == '.'):
					return None
		# It's a tie!
		return '.'

	#check if the game ended and return the result, no modification needed
	def check_end(self):
		self.result = self.is_end()
		# Printing the appropriate message if the game has ended
		if self.result != None:
			if self.result == 'X':
				print('The winner is X!')
			elif self.result == 'O':
				print('The winner is O!')
			elif self.result == '.':
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
		if self.player_turn == 'X':
			self.player_turn = 'O'
		elif self.player_turn == 'O':
			self.player_turn = 'X'
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
		if result == 'X':
			return (-1, x, y)
		elif result == 'O':
			return (1, x, y)
		elif result == '.':
			return (0, x, y)
			#MODIFIED
		for i in range(0, self.boardSize):
			for j in range(0, self.boardSize):
				if self.current_state[i][j] == '.':
					if max:
						self.current_state[i][j] = 'O'
						(v, _, _) = self.minimax(max=False)
						if v > value:
							value = v
							x = i
							y = j
					else:
						self.current_state[i][j] = 'X'
						(v, _, _) = self.minimax(max=True)
						if v < value:
							value = v
							x = i
							y = j
					self.current_state[i][j] = '.'
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
		if result == 'X':
			return (-1, x, y)
		elif result == 'O':
			return (1, x, y)
		elif result == '.':
			return (0, x, y)
			#MODIFIED
		for i in range(0, self.boardSize):
			for j in range(0, self.boardSize):
				if self.current_state[i][j] == '.':
					if max:
						self.current_state[i][j] = 'O'
						(v, _, _) = self.alphabeta(alpha, beta, max=False)
						if v > value:
							value = v
							x = i
							y = j
					else:
						self.current_state[i][j] = 'X'
						(v, _, _) = self.alphabeta(alpha, beta, max=True)
						if v < value:
							value = v
							x = i
							y = j
					self.current_state[i][j] = '.'
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
				if self.player_turn == 'X':
					(_, x, y) = self.minimax(max=False)
				else:
					(_, x, y) = self.minimax(max=True)
			else: # algo == self.ALPHABETA
				if self.player_turn == 'X':
					(m, x, y) = self.alphabeta(max=False)
				else:
					(m, x, y) = self.alphabeta(max=True)
			end = time.time()
			if (self.player_turn == 'X' and player_x == self.HUMAN) or (self.player_turn == 'O' and player_o == self.HUMAN):
					if self.recommend:
						print(F'Evaluation time: {round(end - start, 7)}s')
						print(F'Recommended move: x = {x}, y = {y}')
					(x,y) = self.input_move()
			if (self.player_turn == 'X' and player_x == self.AI) or (self.player_turn == 'O' and player_o == self.AI):
						print(F'Evaluation time: {round(end - start, 7)}s')
						print(F'Player {self.player_turn} under AI control plays: x = {x}, y = {y}')
			self.current_state[x][y] = self.player_turn
			self.switch_player()

def main():
	boardSize = int(input("Size of board: "))
	numBloc =  int(input("Number of blocs: "))
	coordinates = []
	for i in range(numBloc):
		x_bloc = int(input(f"Enter the x{i+1}-coordinate of bloc position: "))
		y_bloc = int(input(f"Enter the y{i+1}-coordinate of bloc position: "))
		coordinates.append((x_bloc, y_bloc))
	winLine = int(input("Enter the number of winning line: ")) 
	#while(not(boardSize >= 3 and boardSize <= 10)):
	#	boardSize = int(input("Size must be between 3 and 10! Try Again... Size of board: "))
	g = Game(boardSize, numBloc, coordinates, winLine, recommend=True)
	g.draw_board()
	#g.play(algo=Game.ALPHABETA,player_x=Game.AI,player_o=Game.AI)
	#g.play(algo=Game.MINIMAX,player_x=Game.AI,player_o=Game.HUMAN)

if __name__ == "__main__":
	main()
