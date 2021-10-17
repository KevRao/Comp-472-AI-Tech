# -*- coding: utf-8 -*-
"""
Created on Sat Oct 16 20:01:12 2021
Config file for Mini-project 1.

@author: Kevin
"""

import os;

#track the number of figures created
__Figure_Counter = 0
#Call this when making a figure to track figure count.
def getFigureCount():
    global __Figure_Counter
    __Figure_Counter += 1
    return __Figure_Counter
local_directory = os.path.dirname(__file__)