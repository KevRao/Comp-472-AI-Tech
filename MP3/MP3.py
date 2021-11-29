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
#Read
model_names = ['word2vec-google-news-300', 'glove-twitter-200', 'glove-wiki-gigaword-200', 'glove-wiki-gigaword-50', 'glove-wiki-gigaword-300']#["GensimContinuousSkipgram-BritishNationalCorpus-300"]#["word2vec-google-news-300"]

#Write
local_directory = os.path.dirname(__file__)
output_directory = os.path.join(local_directory, 'output')

#%% Task 1
def get_model(model_name):
	start = time.perf_counter()
	wv = api.load(model_name)
	end = time.perf_counter()

	print(f"Elapsed time to get {model_name}: {end-start}s.")
	return wv

def get_synonym_test():
	#Read questionnaire file
	file = pd.read_csv("synonyms.csv")
	return file

def details_analysis(synonym_test, keyedVectors, model_name):
	file = synonym_test
	wv = keyedVectors

	answers = file["answer"].values
	questions = file["question"].values

# 	#initialize the lists.
# 	guess_words  = []
# 	guess_labels = []
# 	for question, answer, *choices_all in file.values:
# 		#exclude options that the model doesn't know about.
# 		choices = [choice for choice in choices_all if wv.has_index_for(choice)]
# 		#is a guess if there are no choices, or the model doesn't know the question-word
# 		is_valid = bool(choices) and wv.has_index_for(question)

# 		#if the question word exists, pick the most similar choice,
# 		# otherwise, pick randomly from the valid choices,
# 		# otherwise, pick randomly from all the choices.
# 		guess_word = wv.most_similar_to_given(question, choices) if is_valid else random.choice(choices or choices_all)
# 		#remember the guess words, and whether it was determined through the model or random.
# 		guess_words.append(guess_word)
# 		guess_labels.append(not is_valid)

	#Cleaner list comprehension.
	choices_all = [choices_all for _, _, *choices_all in file.values]
	#exclude options that the model doesn't know about.
	choices = [[choice for choice in choices if wv.has_index_for(choice)] for choices in choices_all]
	#is a guess if there are no choices, or the model doesn't know the question-word
	guess_labels = [not (choices and wv.has_index_for(question)) for choices, question in zip(choices, questions)]
	#if the question word exists, pick the most similar choice,
	# otherwise, pick randomly from the valid choices,
	# otherwise, pick randomly from all the choices.
	guess_words = [wv.most_similar_to_given(question, choices) if not guess_label else random.choice(choices or choices_all)
							  for question, choices_all, choices, guess_label in zip(questions, choices_all, choices, guess_labels)]

	correct_labels = answers == guess_words

	#Assign appropriate label to model's guess.
	labels = np.full_like(guess_words, "wrong")
	labels[correct_labels] = "correct"
	labels[guess_labels]   = "guess"

	#Collect the details of the synonym test.
	detail = np.array([questions, answers, guess_words, labels]).T

	vocab_size = len(wv.index_to_key)
	correct = np.count_nonzero(correct_labels)
	valid = np.count_nonzero(~np.array(guess_labels))
	accuracy = correct / valid
	#Collect the performance of the model in respect to the synonym test.
	performance = [model_name, vocab_size, correct, valid, accuracy]

	return detail, performance

def output_csv(output_filename, table):
	#Task 1 output details
	filename = output_filename
	filepath = os.path.join(output_directory, filename)
	with open(filepath, 'w', newline='') as output_file:
		writer = csv.writer(output_file)
		writer.writerows(table)

# %% Main flow
def main():
	analysis = []

	synonym_test = get_synonym_test()
	#Batch Task1 and Task 2 together.
	for model_name in model_names:
		wv = get_model(model_name)
		detail, performance = details_analysis(synonym_test, wv, model_name)

		#Step 1, output details
		filename = f"{model_name}-details.csv"
		output_csv(filename, detail)
		analysis.append(performance)

	#Batch Step 2, output analysis
	filename = "analysis.csv"
	output_csv(filename, analysis)


if __name__ == "__main__":
    begin_time = time.perf_counter()
    main()
    end_time = time.perf_counter()
    print("This script has taken", end_time - begin_time, "seconds to execute.")
