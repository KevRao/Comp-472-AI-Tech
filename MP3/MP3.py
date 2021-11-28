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
analysis = []
model_name = "word2vec-google-news-300"
start = time.perf_counter()
wv = api.load(model_name)
end = time.perf_counter()

print("Elapsed time:", end-start)
#%% Task 1

#Read questionnaire file
file = pd.read_csv("synonyms.csv")
answers = file["answer"].values
questions = file["question"].values

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

correct_labels = answers == guess_words

labels = np.full_like(guess_words, "wrong")
labels[correct_labels] = "correct"
labels[guess_labels]   = "guess"

detail = np.array([questions, answers, guess_words, labels]).T

vocab_size = len(wv.index_to_key)
correct = np.count_nonzero(correct_labels)
valid = np.count_nonzero(~np.array(guess_labels))
acc = correct / valid
performance = [model_name, vocab_size, correct, valid, acc]
analysis.append(performance)

#Task 1 output details
filename = f"{model_name}-details.csv"
filepath = os.path.join(output_directory, filename)
with open(filepath, 'w', newline='') as output_file:
	writer = csv.writer(output_file)
	writer.writerows(detail)

#Task 1 output analysis
filename = "analysis.csv"
filepath = os.path.join(output_directory, filename)
with open(filepath, 'w', newline='') as output_file:
	writer = csv.writer(output_file)
	writer.writerows(analysis)

