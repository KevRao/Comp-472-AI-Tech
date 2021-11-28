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

#%%

#Write
local_directory = os.path.dirname(__file__)
output_directory = os.path.join(local_directory, 'output')
#%%
start = time.perf_counter()
wv = api.load('word2vec-google-news-300')
print("help")
end = time.perf_counter()

print("Elapsed time:", end-start)
#%% test

# question = "enormously"
# a = "appropriately"
# b = "uniquely"
# c = "tremendously"
# d = "decidedly"

# choices = [a, b, c, d]
# for choice in choices:
# 	print(wv.similarity(question, choice))

# print(wv.most_similar_to_given(question, choices))

file = pd.read_csv("synonyms.csv")
# file.values

# for row in file.values:
# 	answer = wv.most_similar_to_given(row[1], row[2:])
# for question, answer, *choices in file.values:
# 	#exclude options that the model doesn't know about.
# 	choices = [choice for choice in choices if wv.has_index_for(choice)]
# 	#is a guess if there are no choices, or the model doesn't know the question-word
# 	is_guess = bool(choices) or wv.has_index_for(question)
#
# 	guess_word =

# guess_words = [wv.most_similar_to_given(question, []) if wv.has_index_for(question) else None for question, answer, *choices in file.values]

# choices = [choice for choice in [choices for _, _, *choices in file.values] if wv.has_index_for(choice)]
# guess_words = [wv.most_similar_to_given(question, choices) if wv.has_index_for(question) else None for question, answer, *choices in file.values]

# guess_words = np.array([wv.most_similar_to_given(question, choices) if choices and wv.has_index_for(question) else None
# 			   for question, answer, choices in [(question, answer, [choice for choice in choices if wv.has_index_for(choice)])
# 			   for question, answer, *choices in file.values]])

#initialize the lists.
#List comprehension is doable, but very messy to declare variables inside.
guess_words  = []
guess_labels = []
for question, answer, *choices_all in file.values:
 	#exclude options that the model doesn't know about.
 	choices = [choice for choice in choices_all if wv.has_index_for(choice)]
 	#is a guess if there are no choices, or the model doesn't know the question-word
 	is_valid = bool(choices) and wv.has_index_for(question)

# 	if is_valid:
# 		guess_word = random.choice(choices) if choices else random.choice(choices_all)
# 	else:
# 		guess_word = wv.most_similar_to_given(question, choices)

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

# [(question, answer, [choice for choice in choices if wv.has_index_for(choice)])
# 			   for question, answer, *choices in file.values]
# question, answer, *choices = file.values[63]

# [(a, [b for b in bs if b>2]) for a, bs in [(1,(1,2,3,4)), (2,(4,-1,7,5))]]
# for choice in choices:
#  	print(wv.similarity(question, choice))
