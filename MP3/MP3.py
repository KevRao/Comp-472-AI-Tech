# -*- coding: utf-8 -*-
"""
Created on Sat Nov 27 18:44:28 2021

"""
import time
import os
import random

import pandas as pd
import numpy as np
import gensim.downloader as api

import csv

#%% Config

#Write
local_directory = os.path.dirname(__file__)
output_directory = os.path.join(local_directory, 'output')
#%% Init embeddings
start = time.perf_counter()
wv = api.load('word2vec-google-news-300')
print("help")
end = time.perf_counter()

print("Elapsed time:", end-start)
#%% Tasks

file = pd.read_csv("synonyms.csv")
#initialize the lists.
#List comprehension is doable, but very messy to declare variables inside.
guess_words  = []
guess_labels = []
for question, answer, *choices_all in file.values:
 	#exclude options that the model doesn't know about.
 	choices = [choice for choice in choices_all if wv.has_index_for(choice)]
 	#is a guess if there are no choices, or the model doesn't know the question-word
 	is_valid = bool(choices) and wv.has_index_for(question)

 	#if the question word exists, pick the most similar choice,
 	# otherwise, pick randomly from the valid choices,
 	# otherwise, pick randomly from all the choices.
 	guess_word = wv.most_similar_to_given(question, choices) if is_valid else random.choice(choices) if choices else random.choice(choices_all)
 	#remember the guess words, and whether it was determined through the model or random.
 	guess_words.append(guess_word)
 	guess_labels.append(not is_valid)

#the same, but with list comprehension
# guess_words, guess_labels = zip(*[(wv.most_similar_to_given(question, choices), not is_valid) if is_valid else (random.choice(choices) if choices else random.choice(choices_all), not is_valid)
# 			   for question, answer, choices_all, choices, is_valid in [(question, answer, choices_all, choices, bool(choices and wv.has_index_for(question)))
# 			   for question, answer, choices_all, choices in [(question, answer, choices_all, [choice for choice in choices_all if wv.has_index_for(choice)])
# 			   for question, answer, *choices_all in file.values]]])
# # guess_words = np.array(guess_words)
# # guess_labels = np.array(guess_labels)
# guess_words, guess_labels =  list(guess_words), list(guess_labels)


answers = file["answer"].values
correct_labels = answers == guess_words
# guess_labels = guess_words == None

labels = np.full_like(guess_words, "wrong")
labels[correct_labels] = "correct"
labels[guess_labels]   = "guess"

questions = file["question"].values
output = np.array([questions, answers, guess_words, labels]).T

filename = "word2vec-google-news-300-details.csv"
filepath = os.path.join(output_directory, filename)
with open(filepath, 'w', newline='') as output_file:
	writer = csv.writer(output_file)
	writer.writerows(output)
