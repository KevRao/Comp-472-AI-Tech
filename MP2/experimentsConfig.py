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
		"board_size"          : 9,
		"blocs_num"           : 15,
		"coordinates"         : [(7,2),(8,8),(1,1),(5,2),(5,7),(4,2),(1,8),(0,3),(3,1),(4,3),(3,0),(0,6),(2,3),(8,5),(3,7)],
		"winning_line_length" : 4,
		"max_depth_white"     : 6,
		"max_depth_black"     : 6,
		"turn_time_limit"     : 8,
		"output_directory"    : output_directory,
		"recommend"           : True
		},
		alphabeta_play
	))


def getConfig():
	if not experiments:
		initializeConfig()
	return experiments