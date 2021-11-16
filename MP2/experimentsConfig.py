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
	#time   = [60, 60]

	def randBlocLocs(size, num):
		board_indexes = np.argwhere(np.ones((size, size)))
		rng.shuffle(board_indexes)
		return board_indexes[:num].tolist()

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
	#bloc position to add
	experiments.append(({
		"board_size"          : board[0],
		"blocs_num"           : b_num[0],
		"coordinates"         : [(0,0),(0,3),(3,0),(3,3)],
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
	#need to add position of blocs
	experiments.append(({
		"board_size"          : board[0],
		"blocs_num"           : b_num[0],
		"coordinates"         : [(0,0),(0,3),(3,0),(3,3)],
		"winning_line_length" : length[0],
		"max_depth_white"     : depth[1],
		"max_depth_black"     : depth[1],
		"turn_time_limit"     : time[0],
		"output_directory"    : output_directory,
		"recommend"           : True
		},
		alphabeta_play
	))
	#config 3
	#done
	experiments.append(({
		"board_size"          : board[1],
		"blocs_num"           : b_num[0],
		"coordinates"         : randBlocLocs(board[1], b_num[0]),
		"winning_line_length" : length[1],
		"max_depth_white"     : depth[0],
		"max_depth_black"     : depth[1],
		"turn_time_limit"     : time[0],
		"output_directory"    : output_directory,
		"recommend"           : True
		},
		alphabeta_play
	))
	#config 4
	#done
	experiments.append(({
		"board_size"          : board[1],
		"blocs_num"           : b_num[0],
		"coordinates"         : randBlocLocs(board[1], b_num[0]),
		"winning_line_length" : length[1],
		"max_depth_white"     : depth[1],
		"max_depth_black"     : depth[1],
		"turn_time_limit"     : time[1],
		"output_directory"    : output_directory,
		"recommend"           : True
		},
		alphabeta_play
	))
	#config 5
	#done
	experiments.append(({
		"board_size"          : board[2],
		"blocs_num"           : b_num[1],
		"coordinates"         : randBlocLocs(board[2], b_num[1]),
		"winning_line_length" : length[2],
		"max_depth_white"     : depth[0],
		"max_depth_black"     : depth[1],
		"turn_time_limit"     : time[0],
		"output_directory"    : output_directory,
		"recommend"           : True
		},
		alphabeta_play
	))
	#config 6
	#done
	experiments.append(({
		"board_size"          : board[2],
		"blocs_num"           : b_num[1],
		"coordinates"         : randBlocLocs(board[2], b_num[1]),
		"winning_line_length" : length[2],
		"max_depth_white"     : depth[0],
		"max_depth_black"     : depth[1],
		"turn_time_limit"     : time[0],
		"output_directory"    : output_directory,
		"recommend"           : True
		},
		alphabeta_play
	))
	#config 7
	#done
	experiments.append(({
		"board_size"          : board[2],
		"blocs_num"           : b_num[2],
		"coordinates"         : randBlocLocs(board[2], b_num[2]),
		"winning_line_length" : length[2],
		"max_depth_white"     : depth[1],
		"max_depth_black"     : depth[1],
		"turn_time_limit"     : time[0],
		"output_directory"    : output_directory,
		"recommend"           : True
		},
		alphabeta_play
	))
	#config 8
	#done
	experiments.append(({
		"board_size"          : board[2],
		"blocs_num"           : b_num[2],
		"coordinates"         : randBlocLocs(board[2], b_num[2]),
		"winning_line_length" : length[2],
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