# -*- coding: utf-8 -*-
"""
Created on Sat Oct 16 20:01:12 2021
Config file for Mini-project 1.

@author: Kevin
"""

#%% Imports
import os;

#%% Shared
#track the number of figures created
__Figure_Counter = 0
#Call this when making a figure to track figure count.
def getFigureCount():
    global __Figure_Counter
    __Figure_Counter += 1
    return __Figure_Counter
local_directory = os.path.dirname(__file__)
output_directory = os.path.join(local_directory, 'output')

#%% Task 1
#Read
Task1_input_directoryname = 'BBC'
Task1_train_size_proportion = 0.80
#reason for choosing particular favorite words: 
#'year' is the most frequently used noun in the entire corpus (besides 'mr' and 36 other non-noun words) at 2309 times. 
# This measurement makes a distinction between singular and plurals with different spellings (eg 'years' appears 1003 times).
#'bbc', because the corpus is sourced from them, making the term a self-reference. Appears 767 times.
# Other strings containing 'bbc' seldom appear (cumulatively 14 times).
favorite_words = ['year', 'bbc']

#Write
Task1_distribution_graph_title = "BBC-distribution"
Task1_output_performance_fullname = 'bbc-performance.txt'
#%% Task 2
#Read
Task2_input_filename = 'drug200.csv'
Task2_nominal_columns = ['Sex']
Task2_ordinal_columns = ['BP', 'Cholesterol']

#Reasoning for weighted recall as scoring:
#1. There is an imbalanced distribution, so accuracy is out and weighted metric is preferred.
#2. Better recall(about false negatives) is more important than precision (about false positives), reasoning:
#-Pretend the model can return several drug suggestions. Suppose it recommends
#drug A to everyone. This results in low precision, but we don't care much about
#handing out too many of a drug. Low precision is not a negative, so we don't 
#care about it.
#-Suppose it instead recommends all the drugs except the one the patient needs. This is bad. 
#We want our model to give the drug the patient needs. The metric that becomes low 
#is recall. Low recall is bad, we care about it.
#-F1 measure balances the two, but we only care about one of them, so it is not preferred.
#Therefore, weighted recall is the preferred scoring metric.
topDT_Scoring = 'recall_weighted'
topDT_param = {'criterion': ['gini','entropy'],
               'max_depth': [None, 5],
               'min_samples_split': [2, 4, 8]}
topMLP_Scoring = 'recall_weighted'
topMLP_param = {'activation': ['logistic', 'tanh', 'relu', 'identity'],
                'hidden_layer_sizes': [(30,50), (10,10,10)],
                'solver':['adam', 'sgd']}

# There's an extra ordinal label, so that .cat.codes returns [1, 2, 3] for ["LOW", "NORMAL", "HIGH"]
Task2_ordinal_values = ["", "LOW", "NORMAL", "HIGH"]

stability_iteration_count = 10

#Write
Task2_distribution_graph_title = "drug200-distribution"
Task2_output_performance_fullname = 'drug200-performance.txt'