# -*- coding: utf-8 -*-
"""
Created on Mon Nov 15 17:14:12 2021

"""
import os

import numpy as np

from MP2 import Game

experiments = []
def initializeConfig():
	global experiments
	local_directory = os.path.dirname(__file__)
	output_directory = os.path.join(local_directory, 'output')

	rng = np.random.default_rng()

	board  = [4, 5, 8]
	b_num  = [4, 5, 6]
	length = [3, 4, 5]
	depth  = [2 ,6]
	time   = [1, 5]
	algo   = [Game.MINIMAX, Game.ALPHABETA]

	def randBlocLocs(size, num):
		board_indexes = np.argwhere(np.ones((size, size)))
		rng.shuffle(board_indexes)
		return board_indexes[:num]

	#presets
	minimax_play = {
		"player_x_algo": Game.MINIMAX,
		"player_o_algo": Game.MINIMAX,
		"player_x"     : Game.AI,
		"player_o"     : Game.AI,
		"player_x_e"   : Game.E1,
		"player_o_e"   : Game.E2
	}
	alphabeta_play = {
		"player_x_algo": Game.ALPHABETA,
		"player_o_algo": Game.ALPHABETA,
		"player_x"     : Game.AI,
		"player_o"     : Game.AI,
		"player_x_e"   : Game.E1,
		"player_o_e"   : Game.E2
	}
	#reset the config.
	experiments = []
	#config 1
	experiments.append(({
		"board_size"          : board[0],
		"blocs_num"           : b_num[0],
		"coordinates"         : randBlocLocs(board[0], b_num[0]),
		"winning_line_length" : length[0],
		"max_depth_white"     : depth[1],
		"max_depth_black"     : depth[1],
		"turn_time_limit"     : time[1],
		"output_directory"    : output_directory,
		"recommend"           : True
		},
		minimax_play
	))
	#TODO: below
	#config 2
	experiments.append(({
		"board_size"          : board[0],
		"blocs_num"           : b_num[0],
		"coordinates"         : randBlocLocs(board[0], b_num[0]),
		"winning_line_length" : length[0],
		"max_depth_white"     : depth[1],
		"max_depth_black"     : depth[1],
		"turn_time_limit"     : time[1],
		"output_directory"    : output_directory,
		"recommend"           : True
		},
		alphabeta_play
	))
	#config 3
	experiments.append(({
		"board_size"          : board[0],
		"blocs_num"           : b_num[0],
		"coordinates"         : randBlocLocs(board[0], b_num[0]),
		"winning_line_length" : length[0],
		"max_depth_white"     : depth[1],
		"max_depth_black"     : depth[1],
		"turn_time_limit"     : time[1],
		"output_directory"    : output_directory,
		"recommend"           : True
		},
		alphabeta_play
	))
	#config 4
	experiments.append(({
		"board_size"          : board[0],
		"blocs_num"           : b_num[0],
		"coordinates"         : randBlocLocs(board[0], b_num[0]),
		"winning_line_length" : length[0],
		"max_depth_white"     : depth[1],
		"max_depth_black"     : depth[1],
		"turn_time_limit"     : time[1],
		"output_directory"    : output_directory,
		"recommend"           : True
		},
		alphabeta_play
	))
	#config 5
	experiments.append(({
		"board_size"          : board[0],
		"blocs_num"           : b_num[0],
		"coordinates"         : randBlocLocs(board[0], b_num[0]),
		"winning_line_length" : length[0],
		"max_depth_white"     : depth[1],
		"max_depth_black"     : depth[1],
		"turn_time_limit"     : time[1],
		"output_directory"    : output_directory,
		"recommend"           : True
		},
		alphabeta_play
	))
	#config 6
	experiments.append(({
		"board_size"          : board[0],
		"blocs_num"           : b_num[0],
		"coordinates"         : randBlocLocs(board[0], b_num[0]),
		"winning_line_length" : length[0],
		"max_depth_white"     : depth[1],
		"max_depth_black"     : depth[1],
		"turn_time_limit"     : time[1],
		"output_directory"    : output_directory,
		"recommend"           : True
		},
		alphabeta_play
	))
	#config 7
	experiments.append(({
		"board_size"          : board[0],
		"blocs_num"           : b_num[0],
		"coordinates"         : randBlocLocs(board[0], b_num[0]),
		"winning_line_length" : length[0],
		"max_depth_white"     : depth[1],
		"max_depth_black"     : depth[1],
		"turn_time_limit"     : time[1],
		"output_directory"    : output_directory,
		"recommend"           : True
		},
		alphabeta_play
	))
	#config 8
	experiments.append(({
		"board_size"          : board[0],
		"blocs_num"           : b_num[0],
		"coordinates"         : randBlocLocs(board[0], b_num[0]),
		"winning_line_length" : length[0],
		"max_depth_white"     : depth[1],
		"max_depth_black"     : depth[1],
		"turn_time_limit"     : time[1],
		"output_directory"    : output_directory,
		"recommend"           : True
		},
		alphabeta_play
	))


def getConfig():
	if not experiments:
		initializeConfig()
	return experiments