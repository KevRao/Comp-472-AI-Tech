# based on code from https://stackabuse.com/minimax-and-alpha-beta-pruning-in-python
import os
import string
import time
import os
import numpy as np
import pickle

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

	#Heuristic quick-lookup
	HEURISTIC_SCORE = [100**x for x in range(11)] #10 is max board_size. index corresponds to length along board. Last index for winning.
	HEURISTIC_SCORE[-1] = HEURISTIC_SCORE[-1]*HEURISTIC_SCORE[-1] #make it really big for good measure

	def __init__(self, recommend = True, board_size = 3, blocs_num = 0, coordinates = [], winning_line_length = 3, max_depth_white = 3, max_depth_black = 3, turn_time_limit = 2, output_directory = ''):
		self.board_size = board_size
		self.blocs_num = blocs_num
		self.coordinates = coordinates
		self.winning_line_length = winning_line_length
		self.max_depth_white = max_depth_white
		self.max_depth_black = max_depth_black
		self.turn_time_limit = turn_time_limit #in seconds

		self.play_1, self.play_2 = " ", " "
		self.heuristics_refs = {"e1": self.heuristic_e1, "e2": self.heuristic_e2}


		self.initialize_game()
		self.recommend = recommend

		self.initialize_formatting()

		#output folder directory.
		self.output_directory = output_directory
		self.output_scoreboard = os.path.join(output_directory, 'scoreboard.txt')
		#TODO: set these values to the algorithm of each player.
		self.player_algorithm = {self.WHITE: {"name": "", "value": None}, self.BLACK: {"name": "", "value": None}}
		#TODO: replace the algorithm used in play() with this one, and switch inside switch_player().
		self.current_algorithm = None
		#TODO: set these values to the heuristic of each player. Maybe be a reference to a method.
		self.player_heuristic = {self.WHITE: {"name": "", "value":  None}, self.BLACK: {"name": "", "value": None}}
		#TODO: use the heuristic in the algorithms and switch inside switch_player().
		self.current_heuristic = None

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
			self.current_state[j][i] = self.BLOC
		# Player X always plays first
		#Max depth to consider for the turn.
		self.current_max_depth = self.max_depth_white
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

	def heuristic_e1(self):
		winWhite = 0
		winBlack = 0
		count = 0
		#Horizontal count
		winWhite += sum(1 for i in range(self.board_size) if((self.WHITE in self.current_state[i]  and self.BLACK not in self.current_state[i])
		 or (self.WHITE not in self.current_state[i]  and self.BLACK not in self.current_state[i])))
		winBlack += sum(1 for i in range(self.board_size) if((self.BLACK in self.current_state[i]  and self.WHITE not in self.current_state[i])
		or (self.WHITE not in self.current_state[i]  and self.BLACK not in self.current_state[i])))
		#Vertical count
		column = list(zip(*self.current_state))
		winWhite += sum(1 for i in range(len(column)) if((self.WHITE in column[i] and self.BLACK not in column[i])
		or (self.WHITE not in column[i]  and self.BLACK not in column[i])))
		winBlack += sum(1 for i in range(len(column)) if((self.BLACK in column[i] and self.WHITE not in column[i])
		or (self.BLACK not in column[i]  and self.WHITE not in column[i])))
		#Diagonal count
		if ((self.WHITE in np.diag(self.current_state)) and (self.BLACK not in np.diag(self.current_state))) or ((self.WHITE not in np.diag(self.current_state)) and (self.BLACK not in np.diag(self.current_state))):
			winWhite += 1
		elif ((self.BLACK in np.diag(self.current_state)) and (self.WHITE not in np.diag(self.current_state))) or ((self.WHITE not in np.diag(self.current_state)) and (self.BLACK not in np.diag(self.current_state))):
			winBlack += 1
		#AntiDiagonal count
		if ((self.WHITE in np.fliplr(self.current_state).diagonal()) and (self.BLACK not in np.fliplr(self.current_state).diagonal())) or ((self.WHITE not in np.fliplr(self.current_state).diagonal()) and (self.BLACK not in np.fliplr(self.current_state).diagonal())):
			winWhite += 1
		elif ((self.BLACK in np.fliplr(self.current_state).diagonal()) and (self.WHITE not in np.fliplr(self.current_state).diagonal())) or ((self.WHITE not in np.fliplr(self.current_state).diagonal()) and (self.BLACK not in np.fliplr(self.current_state).diagonal())):
			winBlack += 1
		count = winWhite - winBlack
		return count

	#When a move is committed, the AI can be disqualified if it provides an invalid move.
	def commit_turn(self, x, y, notation):
		# Humans should have a saving check beforehand.
		# Sch that only AI can do invalid move.
		self.depth =[]
		if not self.is_valid(x, y):
			raise Exception(f"Player {self.player_turn} is disqualified for playing an illegal move.")
		self.remember_turn(x, y, notation)

	def draw_board(self):
		# Draw with borders.
		# inner .join is to concatenate cells of a row.
		# outer .join is to concatenate rows of the board.
		body = f'\n{self.body_border}\n'.join([f" {index} ║ {' │ '.join([cell for cell in row])} │" for index, row in enumerate(self.current_state)])
	#	print(f"\n{self.header}\n{body}\n{self.footer}\n")
		return f"\n{self.header}\n{body}\n{self.footer}\n"

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
			if(check_line(occupied_state[self.prev_move_x])):
				return player
			#check Vertical win on column played.
			if(check_line(occupied_state.T[self.prev_move_y])):
				return player
			#check Main diagonal win of played cell
			if(check_line(np.diag(occupied_state, self.prev_move_y - self.prev_move_x))):
				return player
			#check Anti diagonal win of played cell
			if(check_line(np.diag(np.fliplr(occupied_state), self.board_size - 1 - self.prev_move_y - self.prev_move_x))):
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
			py = input('enter the y coordinate: ')
			y_index =['A','B','C','D','E','F','G','H','I','J']
			index = y_index.index(py)

			if self.is_valid(px, index):
				return (px,index)
			else:
				print('The move is not valid! Try again.')

	#switch the player.
	def switch_player(self):
		if self.player_turn == self.WHITE:
			self.current_max_depth = self.max_depth_black
			self.player_turn = self.BLACK
		elif self.player_turn == self.BLACK:
			self.current_max_depth = self.max_depth_white
			self.player_turn = self.WHITE
		self.current_heuristic = self.player_heuristic[self.player_turn]['value']
		return self.player_turn


	# Compute the value of the current state of the board.
	def getHeuristic(self):
		return self.current_heuristic()

	def heuristic_e2(self):
		return self.getPlayerHeuristic(self.BLACK) - self.getPlayerHeuristic(self.WHITE)

	# Compute the value of the current state of the board for the given player.
	def getPlayerHeuristic(self, player):
		#TODO: idea, for each tile, swap its color and accumulate how much it blocked.

		#Gather all the horizontal, vertical and diagonal lines. They will be split apart based on those lines and seperators within.
		#Horizontal and Vertical.
		orthogonal_lines = np.stack((self.current_state, self.current_state.T))
		#Add columns of bloc to the sides.
		#This step is to prepare np.split after np.flatten, such that the splits do not join separate rows.
		# Also add column at the start so that the split rows have the same format.
		bloc_column = np.full((2, self.board_size, 1), self.BLOC)
		board_barrier = np.dstack((bloc_column, orthogonal_lines, bloc_column))

		#Flatten the np.array, so that it plays nicely with np.split.
		flat_board_barrier = board_barrier.ravel()

		#Find the diagonal lines.
		#Prep the diagonal lines, by surrounding the edges of the board and its flipped counterpart with blocs.
		# Do this by filling the whole space with blocs, then replacing the center with the boards.
		# The '+2' is to account for the new corners due to larger array size.
		# The '1:-1' slices mean to ignore the first and last entries. In this case, it means the border.
		board_surrounded = np.full((2, self.board_size + 2, self.board_size + 2), self.BLOC)
		board_surrounded[:, 1:-1, 1:-1] = [self.current_state, np.fliplr(self.current_state)]

		#Furthest offset to find diagonals with sufficient length.
		diagonal_distance = self._board_size - self._winning_line_length
		#Find the combinations of valid length-ed diagonals. Do both main- and anti- diagonals at the same time.
		# range() has '+1', because the end is exclusive (we want inclusive end).
		diagonal_lines = [np.diagonal(board_surrounded, offset=diag_offset, axis1=1, axis2=2)
					for diag_offset in range(-diagonal_distance, diagonal_distance + 1)]

		#Flatten the list by extracting the inner lists.
		flat_diags = []
		for sublist in diagonal_lines:
			flat_diags.extend(sublist.ravel())

		#Join the lines together. These should be all the possible lines.
		flat_board_barrier = np.concatenate((flat_board_barrier, flat_diags))

		#Prep the splitting indexes mask. These should be where the evaluated player's opponent has played and the bloc locations.
		# Since the aforementioned tiles are mutually exclusive with the evaluated player's tiles along the empty tiles, those are used to find the negative space instead.
		flat_board_barrier_state = ~((flat_board_barrier==player)|(flat_board_barrier==self.EMPTY))

		#Split the lines into regions of consecutive tiles for the evaluated player. These include winnable and unwinnable lines.
		splits_board = np.split(flat_board_barrier, *flat_board_barrier_state.nonzero())

		#Determine the progress to completing winnable lines by the amount of the evaluated player's tile therein.
		# Unwinnable lines have the following property:
			# - Maximum possible consecutive tiles for the evaluated player is lower than the winning length.
			# eg winning length is 3, but there's only space enough to put 2.
		# Keep as second in tuple how long the line is compared to the winning length. This will be used as a weight when counting.
		#Each progress is weighted by how much the available length exceeds the winning length, with exponential growth (or however the self.HEURISTIC_SCORE is actually set up).
		progress_player = [(np.count_nonzero(split==player), self.HEURISTIC_SCORE[len(split)-self.winning_line_length-1]) for split in splits_board if len(split) > self.winning_line_length] or [(0, 0)]
		# Organize the progresses into 'bins' (index of a list) by their count, influenced by the assigned weight.
		counts_player = np.bincount(*zip(*progress_player), minlength=1)
		# '@' as an infix operator with np is the matrix multiplication. Since it's used on vectors, it is equivalent to dot product (sum of products).
		heuristic_player = counts_player @ self.HEURISTIC_SCORE[:len(counts_player)]
		return heuristic_player

	def minimax(self, current_depth = 0, max=False):
		# Minimizing for 'X' and maximizing for 'O'
		# Possible values are:
		# -1 - win for 'X'
		# 0  - a tie
		# 1  - loss for 'X'

		#when time limit is reached, return the board's evaluated value.
		ard_depth =[]
		self.timenow = time.perf_counter()
		result = self.is_end()

		if(current_depth==0):
			self.depth = []
			self.turn_start_time = time.perf_counter()

		leeway = 0.0001 * current_depth*current_depth * self.board_size*self.board_size

		#Return heuristic when reaching a leaf node (time limit, depth limit, game end).
		# Makes it so the AI doesn't just give up entirely if it doesn't think it can win. At least lose with a better position.
		if (((self.timenow - self.turn_start_time >= self.turn_time_limit - leeway) and current_depth > 0) or current_depth >= self.current_max_depth or result!=None):
			self.depth.append(current_depth)
			return (self.getHeuristic(), None, None,current_depth)



		# We're initially setting it to 2 or -2 as worse than the worst case:
		value = self.HEURISTIC_SCORE[-1]*2
		if max:
			value = -self.HEURISTIC_SCORE[-1]*2
		x = None
		y = None
		for i, j in np.argwhere(self.current_state == self.EMPTY):
			if max:
				self.remember_turn(i, j, self.BLACK)
				(v, _, _, child_depth) = self.minimax(current_depth = current_depth + 1, max=False)

				if v > value:
					value = v
					x = i
					y = j
			else:
				self.remember_turn(i, j, self.WHITE)
				(v, _, _, child_depth) = self.minimax(current_depth = current_depth + 1, max=True)
				if v < value:
					value = v
					x = i
					y = j
			ard_depth.append(child_depth)
			self.current_state[i][j] = self.EMPTY
		return (value, x, y, ard_depth)

	def alphabeta(self, alpha=-HEURISTIC_SCORE[-1]*2, beta=HEURISTIC_SCORE[-1]*2, current_depth = 0, max=False):
		# Minimizing for 'X' and maximizing for 'O'
		# Possible values are:
		# -1 - win for 'X'
		# 0  - a tie
		# 1  - loss for 'X'
		ard_depth =[]
		#when time limit is reached, return the board's evaluated value.
		# When maximum depth is reached, return the board's evaluated value.
		self.timenow = time.perf_counter()
		result = self.is_end()

		if(current_depth==0):
			self.depth = []
			self.turn_start_time = time.perf_counter()

		#Return heuristic when reaching a leaf node (time limit, depth limit, game end).
		# Makes it so the AI doesn't just give up entirely if it doesn't think it can win. At least lose with a better position.
		leeway = 0.0001 * current_depth*current_depth * self.board_size*self.board_size
		if (((self.timenow - self.turn_start_time >= self.turn_time_limit - leeway) and current_depth > 0) or current_depth >= self.current_max_depth or result!=None):
			self.depth.append(current_depth)
			return (self.getHeuristic(), None, None, current_depth)

		# We're initially setting it to 2 or -2 as worse than the worst case:
		value = -self.HEURISTIC_SCORE[-1]*2 if max else self.HEURISTIC_SCORE[-1]*2
		x = None
		y = None
		for i, j in np.argwhere(self.current_state == self.EMPTY):
			if max:
				self.remember_turn(i, j, self.BLACK)
				(v, _, _, child_depth) = self.alphabeta(alpha, beta, current_depth = current_depth + 1, max=False)
				if v > value:
					value = v
					x = i
					y = j
			else:
				self.remember_turn(i, j, self.WHITE)
				(v, _, _, child_depth) = self.alphabeta(alpha, beta, current_depth = current_depth + 1, max=True)
				if v < value:
					value = v
					x = i
					y = j
			self.current_state[i][j] = self.EMPTY
			ard_depth.append(child_depth)
			if max:
				if value >= beta:
					return (value, x, y, ard_depth)
				if value > alpha:
					alpha = value
			else:
				if value <= alpha:
					return (value, x, y, ard_depth)
				if value < beta:
					beta = value

		return (value, x, y, ard_depth)


	def runScoreboardSeries(self, rounds=5):
		#Initialize the win counts.
		wins = {self.player_heuristic[self.WHITE]["name"]: 0, self.player_heuristic[self.BLACK]["name"]: 0}
		#TODO: initialize the sum_game_end_stats properly to match 2.5.1-6
		sum_game_end_stats = {}
		#Play a batch of rounds twice, swapping the heuristic's side in between.
		for _ in range(2):
			#Play the batch of rounds, and tally up the wins for each heuristic.
			for _ in range(rounds):
				#TODO: self.play parameters incl. heuristc of players, and make it return the winner and game end stats.
				winner, game_end_stats = self.play(player_x=self.AI,player_o=self.AI)
				#Count the win for the heuristic used.
				wins[self.player_heuristic[winner]["name"]] += 1
				#Add up the game end stats.
				for stat, value in game_end_stats.items():
					sum_game_end_stats[stat] += value
			#Switch the heuristics' side
			#TODO: Implement this method. Swap the sides the heuristics are on. ie if (p1,p2) were using (e1,e2), then they should swap to (e2,e1).
			#May not have to be a class method.
			self.swap_heuristic()
		#Average out the game end stats.
		avg_game_end_stats = sum_game_end_stats.copy()
		for stat, value in avg_game_end_stats.items():
			avg_game_end_stats[stat] = value/(rounds*2)
		#TODO: filename placeholder for now.
		filename = 'scoreboard.txt'
		#Output findings to a file. Append to previous.
		with open(os.path.join(self.output_directory, filename), 'a') as output_file:
			output_file.write("-------------------------------------------------")
			self.outputScoreboard(rounds, wins, avg_game_end_stats, output_file)

	def outputScoreboard(self, rounds, wins, aggregated_average_games, output_file):
		#use inside with open(...) as ...:

		#TODO: Assign values to attributes related in Algorithm Used, Heuristic Used

		#1. Parameters of the game.
		output_file.write("1. Parameters of the game:")
		output_file.write(f"Board Size      n: {self.board_size}")
		output_file.write(f"Number of Blocs b: {self.blocs_num}")
		output_file.write(f"Winning Length  s: {self.board_size}")
		output_file.write(f"Time per Turn   t: {self.turn_time_limit} seconds")

		#2. Parameters of the players
		output_file.write("\n2. Parameters of the players:")
		output_file.write(f"Max Search Depth d1, d2: {self.max_depth_white}, {self.max_depth_black}")
		output_file.write(f"Algorithm Used   a1, a2: {self.player_algorithm[self.WHITE]['name']}, {self.player_algorithm[self.BLACK]['name']}")
		output_file.write(f"Heuristic Used   e1, e2: {self.player_heuristic[self.WHITE]['name']}, {self.player_heuristic[self.BLACK]['name']}")

		#3. Games played
		output_file.write("\n3. Games played:")
		output_file.write(f"Games played: {rounds * 2}")

		#4. Wins and ratio
		output_file.write("\n4. Game wins:")
		output_file.write(f"Player e1 wins, ratio: {wins[0]}, {wins[0]/(rounds*2):.2%}")
		output_file.write(f"Player e2 wins, ratio: {wins[1]}, {wins[1]/(rounds*2):.2%}")

		#5. Averaged gametrace
		#TODO: make sure the format of aggregated_average_games is good.
		#TODO: take average across round*2 games
		output_file.write("\n5. Average gametrace:")
		for stat, value in aggregated_average_games.items():
			output_file.write(f"{stat}: {value}")


	def play(self,algo=None,player_x=None,player_o=None, player_x_e=None, player_o_e=None):
		global gametrace_history
		if algo == None:
			algo = self.ALPHABETA
		if player_x == None:
			player_x = self.HUMAN
		if player_o == None:
			player_o = self.HUMAN
		if player_x_e == None:
			player_x_e = "e1"
		if player_o_e == None:
			player_o_e = "e2"

		self.player_heuristic[self.WHITE]["name"]  = player_x_e
		self.player_heuristic[self.WHITE]["value"] = self.heuristics_refs[player_x_e]
		self.player_heuristic[self.BLACK]["name"]  = player_o_e
		self.player_heuristic[self.BLACK]["value"] = self.heuristics_refs[player_o_e]
		self.current_heuristic = self.player_heuristic[self.WHITE]["value"]

		gametrace_history = {"e1": np.array([]), "e2": np.array([])}
		turn_counts = {"e1": 0, "e2": 0}

		def bindepthToString(bindepth):
			depth_list = [f"depth {depth}: {heuristic_count}" for depth, heuristic_count in enumerate(bindepth)]
			depth_eval= ", ".join(depth_list)
			return depth_eval

		output_fullname = f'gametrace-{self.board_size}{self.blocs_num}{self.winning_line_length}{int(self.turn_time_limit)}.txt'
		output_fullpath = os.path.join(self.output_directory, output_fullname)

		body = f'\n{self.body_border}\n'.join([f" {index} ║ {' │ '.join([cell for cell in row])} │" for index, row in enumerate(self.current_state)])

		alphaMax = ("True" if algo == self.ALPHABETA else "False")
		self.play_1 = ("Human" if player_x == self.HUMAN else "AI")
		self.play_2 = ("Human" if player_o == self.HUMAN else "AI")

		# 1 2 3 4.
		with open(output_fullpath, 'w') as gameTrace:
			gameTrace.writelines("GAME TRACE \n\n")
			gameTrace.writelines(["n=", str(self.board_size), ", b=", str(self.blocs_num), ", s=", str(self.winning_line_length), ", t=", str(self.turn_time_limit), "\n"])
			gameTrace.writelines(["blocs=["])
			for i in range (len(self.coordinates)):
				gameTrace.writelines([str(self.coordinates[i]), " "])
			gameTrace.writelines(["] \n\n"])
			gameTrace.writelines(["Player 1: ", self.play_1, " d=", str(self.max_depth_white), " a=", alphaMax, " e=", self.player_heuristic[self.WHITE]["name"], " \n"])
			gameTrace.writelines(["Player 2: ", self.play_2, " d=", str(self.max_depth_black), " a=", alphaMax, " e=", self.player_heuristic[self.BLACK]["name"], " \n"])


		#main game loop
		while True:
			print(self.draw_board())
			current_turn_heuristic_name = self.player_heuristic[self.player_turn]["name"]
			with open(output_fullpath, 'a', encoding="utf-8") as output_file:
				output_file.write(self.draw_board())
				output_file.write("\n")
			winner = self.check_end()
			if winner:
				win_msg = f"The winner is {winner}!\n\n" if winner != self.EMPTY else "There is no winner. The game is a tie!\n\n"
				with open(output_fullpath, 'a', encoding='utf-8') as output_file:
					output_file.write(win_msg)
				#Return to stop the game.
				break

			start = time.time()
			if algo == self.MINIMAX:
				if self.player_turn == self.WHITE:
					(_, x, y, ard) = self.minimax(max=False)
				else:
					(_, x, y, ard) = self.minimax(max=True)
			else: # algo == self.ALPHABETA
				if self.player_turn == self.WHITE:
					(m, x, y, ard) = self.alphabeta(max=False)
				else:
					(m, x, y, ard) = self.alphabeta(max=True)
			end = time.time()
			elapsed_time = (end - start)
			if (self.player_turn == self.WHITE and player_x == self.HUMAN) or (self.player_turn == self.BLACK and player_o == self.HUMAN):
				if self.recommend:
					print(F'Evaluation time: {round(end - start, 7)}s')
					print(F'Recommended move: x = {x}, y = {y}')
				(x,y) = self.input_move()
			if (self.player_turn == self.WHITE and player_x == self.AI) or (self.player_turn == self.BLACK and player_o == self.AI):
				print(F'Evaluation time: {round(end - start, 7)}s')
				print(F'Player {self.player_turn} under AI control plays: x = {x}, y = {y}')

				index = string.ascii_uppercase[y]
				move_made = index + str(x)
				eval_time = round(end - start, 7)


				turn_eval = self.getTurnGameStats(elapsed_time, ard)

				#convert bindepth to a string format.
				depth_eval= bindepthToString(turn_eval[2])


				# 2.5.1- step 5.
				with open(output_fullpath, 'a', encoding="utf-8") as output_file:
					output_file.writelines(["Player ", self.player_turn," under AI control plays:", move_made, "\n\n"])
					output_file.writelines([
                                  "i   Evaluation time: "              + str(turn_eval[0])+ "\n",
                                  "ii  Heuristic evaluations: "        + str(turn_eval[1])+ "\n",
								  "iii Evaluation by depth: {"         + str(depth_eval  )+ "}\n",
								  "iv  Average evaluation depth (AD): "+ str(turn_eval[3])+ "\n",
                                  "v   Average recursion depth (ARD): "      + str(turn_eval[4])+ "\n\n"
								  ])

				if (elapsed_time > self.turn_time_limit):
					raise Exception(f"Player(AI) {self.player_turn} is disqualified for taking too long to play a move.")

			#Prep next turn.
			turn_counts[current_turn_heuristic_name] += 1
			self.commit_turn(x, y, self.player_turn)
			self.switch_player()
			#if empty, initialize the history. otherwise, append turn evalutations to the history.
			if gametrace_history[current_turn_heuristic_name].size == 0:
				gametrace_history[current_turn_heuristic_name] = turn_eval
			else:
				gametrace_history[current_turn_heuristic_name] = np.vstack((gametrace_history[current_turn_heuristic_name], turn_eval))
		#Post-game
		# 2.5.1- step 6.
		with open(output_fullpath, 'a', encoding="utf-8") as output_file:
			for e_name, heuristic_history in gametrace_history.items():
				avg_eval_time, total_evals, avg_avg_depth, total_bindepth, avg_ard, total_turns = self.getEndGameStats(heuristic_history, turn_counts[e_name])

				#convert bindepth to a string format.
				depth_eval= bindepthToString(total_bindepth)

				output_file.write(f"Summary of the game heuristics {e_name}: \n")
				output_file.writelines([
					"i   Average evaluation time: "   + str(avg_eval_time)+ "\n",
					"ii  Total states visited: "      + str(total_evals  )+ "\n",
					"iii Average AD: {"               + str(avg_avg_depth)+ "}\n",
					"iv  Total evaluations by depth: "+ str(depth_eval   )+ "\n",
					"v   Average ARD: "               + str(avg_ard      )+ "\n",
					"vi  Total number of moves: "     + str(total_turns  )+ "\n\n"
				])

		return gametrace_history

	def getEndGameStats(self, heuristic_history, turn_count):
		e_sum = heuristic_history.sum(axis=0)
		e_avg = heuristic_history.mean(axis=0)

		#i avg eval time
		avg_eval_time = e_avg[0]

		#ii total num states
		total_evals = e_sum[1]

		#iii avg [avg depth]
		avg_avg_depth = e_avg[3]

		#iv total each depth visit
		total_bindepth = e_sum[2]

		#v avg ard
		avg_ard = e_avg[4]

		#vi total turns played
		total_turns = turn_count

		game_eval = np.array([avg_eval_time, total_evals, avg_avg_depth, total_bindepth, avg_ard, total_turns], dtype=object)
		return game_eval

	def getTurnGameStats(self, elapsed_time, ard):
		bindepth = np.bincount(self.depth)
		heu_eval = bindepth.sum()

		avg_eval_depth = (bindepth @ np.arange(len(bindepth)))/heu_eval

		#TO-DO
		#dictionnary for all 5 outputs
		def find_ard(node):
			if not isinstance(node,list):
				return node
			else:
				ard_list =[find_ard(child) for child in node]
				ard_list = np.array(ard_list)
			return np.mean(ard_list)

		avg_recur_depth = find_ard(ard)
		turn_eval = np.array([elapsed_time, heu_eval, bindepth, avg_eval_depth, avg_recur_depth], dtype=object)
		return turn_eval


def askBoolean(msg):
	valid_inputs = {
		"true" : True,
		"t"    : True,
		"yes"  : True,
		"y"    : True,
		"false": False,
		"f"    : False,
		"no"   : False,
		"n"    : False,
	}
	while True:
		try:
			return valid_inputs[input(msg).strip().casefold()]
		except KeyError:
			print("Input provided is not a boolean! Valid inputs are: ", list(valid_inputs.keys()))

def askPlayMode(msg):
	valid_inputs = {
		"0": (Game.AI   , Game.AI   ),
		"1": (Game.HUMAN, Game.HUMAN),
		"2": (Game.HUMAN, Game.AI   ),
		"3": (Game.AI   , Game.HUMAN),
	}
	while True:
		try:
			return valid_inputs[input(msg).strip().casefold()]
		except KeyError:
			print("Input provided is not valid! Valid inputs are: ", list(valid_inputs.keys()))

def askPlayHeuristics(msg):
	valid_inputs = {
		"0": ("e1", "e1"),
		"1": ("e2", "e2"),
		"2": ("e2", "e1"),
		"3": ("e1", "e2"),
	}
	while True:
		try:
			return valid_inputs[input(msg).strip().casefold()]
		except KeyError:
			print("Input provided is not valid! Valid inputs are: ", list(valid_inputs.keys()))

def askInt(msg):
	while True:
		try:
			return abs(int(input(msg)))
		except ValueError:
			print("Input provided is not valid! Valid inputs are integers.")

def askFloat(msg):
	while True:
		try:
			return abs(float(input(msg)))
		except ValueError:
			print("Input provided is not valid! Valid inputs are floats.")

#Write
local_directory = os.path.dirname(__file__)
output_directory = os.path.join(local_directory, 'output')
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

	depth_prompt = "Max depth for {}: "
	max_depth_white, max_depth_black = (askInt(depth_prompt.format("white")), askInt(depth_prompt.format("black")))

	turn_time_limit_prompt = "Turn time limit: "
	turn_time_limit = askFloat(turn_time_limit_prompt)

	algorithm_prompt = "Use ALPHABETA? (Alternative is MINIMAX.) "
	algorithm = Game.ALPHABETA if askBoolean(algorithm_prompt) else Game.MINIMAX

	mode_prompt = (
		"Select a game mode:\n"
		"\t0 - AI    vs AI\n"
		"\t1 - Human vs Human\n"
		"\t2 - Human vs AI\n"
		"\t3 - AI    vs Human\n"
	)
	player_one, player_two = askPlayMode(mode_prompt)
	heuristic_prompt = (
		"Select a game mode:\n"
		"\t0 - e1 vs e1\n"
		"\t1 - e2 vs e2\n"
		"\t2 - e2 vs e1\n"
		"\t3 - e1 vs e2\n"
	)
	player_one_e, player_two_e = askPlayHeuristics(heuristic_prompt)
	g = Game(board_size = boardSize,
		  blocs_num = numBloc,
		  coordinates = coordinates,
		  winning_line_length = winLine,
		  max_depth_white = max_depth_white,
		  max_depth_black = max_depth_black,
		  turn_time_limit = turn_time_limit,
		  output_directory = output_directory,
		  recommend=True)
#	g = Game(board_size = 9,
#		  blocs_num = 0,
#		  coordinates = None,
#		  winning_line_length = 3,
#		  max_depth_white = 3,
#		  max_depth_black = 9,
#		  turn_time_limit = 5,
#		  recommend=True)





	g.play(algo=algorithm, player_x=player_one, player_o=player_two)
#	g.play(algo=Game.ALPHABETA, player_x=Game.AI, player_o=Game.AI)
#	g.play(algo=Game.MINIMAX,player_x=Game.AI,player_o=Game.HUMAN)

if __name__ == "__main__":
	main()
