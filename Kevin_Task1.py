# -*- coding: utf-8 -*-
"""
Created on Sat Oct  2 12:44:22 2021.
Note: Do not have the produced figures open when running this script. It may 
cause conflict and have the script return the following error:
'OSError: [Errno 22] Invalid argument: [...].png'.
My guess is that Python has trouble writing to the filename when it is in use 
by another program.

@author: Kevin
"""
#%% Imports
import os;
#import sklearn;
from sklearn import datasets; 
import time;
import numpy as np;
import matplotlib.pyplot as plt;
from collections import Counter;
import collections;
from sklearn.feature_extraction.text import CountVectorizer

#%% Helper Functions Declaration
#Retrieve data from the corpus
def getCorpus(corpus_directory):
    #Encoding as specified in the mini-project document.
    corpus_encoding = 'latin1'

    return datasets.load_files(corpus_directory, encoding=corpus_encoding)

#Handle the data from the corpus
#In this case, determine the corpus distribution by classification
def determineCorpusDistribution(corpus):   
    #corpus_counter is a dict, where each key is the index of its corresponding classification name (from a different list).
    # so extract the values, and re-order the classification name by the key
    corpus_counter = Counter(corpus.target)
    #Separate the keys from the values of the dict.
    #Sorry, I don't know an English word for 'effectif'. It's French for 'number of instances appearing in each label' in the context of statistics. Super convenient.
    indexes, effectif = zip(*corpus_counter.items())
    
    #np arrays are easier to give arbitrary sorting,
    # so convert the names of categories into an np-array, then re-order them.
    #np array plays better with a list than it does with tuples.
    labels = np.array(corpus.target_names)[list(indexes)]
    
    return effectif, indexes, labels

def determineCorpusDetails_byClassification(corpus):
    ##Failed Code
    # for index, classification in enumerate(corpus.target_names):
    #     corpus_by_classification[index] = [data[1] for data in zip(corpus.target, corpus.data) if data[0]==index]
    #     for index2, classed_corpus in enumerate(corpus_by_classification):
    #         corpus_by_classification[index2] = vectorizer.fit_transform(classed_corpus)
    
    ##Obsolete Code
    # #Organize the corpus data into its classifications
    # corpus_collections = collections.defaultdict(list)
    # [corpus_collection[class_name].append(data) 
    #  for class_index, class_name 
    #  in enumerate(corpus.target_names) 
    #  for target_index, data 
    #  in zip(corpus.target, corpus.data) 
    #  if target_index==class_index]
    
    #Organize the corpus data into its classifications
    #Structure is a list in a dict in a dict
    corpus_collections = collections.defaultdict(lambda: collections.defaultdict(list));
    #bind the corpus data to its classification's data entry
    for target_index, data in zip(corpus.target, corpus.data):
        corpus_collections[corpus.target_names[target_index]]['data'].append(data)
    
    #Initiate countVectorizer for each classification
    for corpus_collection in corpus_collections.values():
        #create a new instance of countVectorizer
        collection_vectorizer = CountVectorizer()
        #attach the new countVectorizer's reference.
        corpus_collection['vectorizer'].append(collection_vectorizer)
        #process the countVectorizer to the data
        collection_vectorizer.fit_transform(corpus_collection['data'])
        
    return corpus_collections

#TODO: remove either of the generateBarGraph methods. Only one is needed. Choice between is in axis label order.

#Make a bar plot out of the corpus distribution.
def generateBarGraph(title, labels, values, value_indexes): 
    plot_title = f"Figure {getFigureCount()}. {title}"
    
    #in order arbitrarily decided by sklearn.datasets.load_files, seemingly alphabetical
    plt.barh(value_indexes, values, tick_label=labels, color="xkcd:battleship grey")
    #invert the category axis, since it's upside-down
    plt.gca().invert_yaxis()
    plt.title(plot_title)
    #provide values for each bar
    for count, value in enumerate(values):
        #value and count used as coordinates for the text.
        plt.text(value, value_indexes[count], str(value) + " ", va="center", ha="right", color="aliceblue")
    
    plt.savefig(os.path.join(output_directory, plot_title + ".png"), bbox_inches='tight', format="png")
    plt.show()

#Make a bar plot out of the corpus distribution.
def generateBarGraph_Alternate(title, labels, values): 
    plot_title = f"Figure {getFigureCount()}. {title}"
    
    #in order arbitrarily decided by Counter
    plt.barh(labels, values, color="xkcd:battleship grey")
    #invert the category axis, since it's upside-down
    plt.gca().invert_yaxis()
    plt.title(plot_title)
    #provide values for each bar
    for count, value in enumerate(values):
        #value and count used as coordinates for the text.
        plt.text(value, count, str(value) + " ", va="center", ha="right", color="aliceblue")
    
    plt.savefig(os.path.join(output_directory, plot_title + ".png"), bbox_inches='tight', format="png")
    plt.show()

#Call this when making a figure to track figure count.
def getFigureCount():
    global Figure_Counter
    Figure_Counter += 1
    return Figure_Counter

#%% Configuration and Globals Declarations
#configuration and such
script_directory = os.path.dirname(__file__)
#Read
BBC_directory = os.path.join(script_directory, 'BBC')
#Write
output_directory = os.path.join(script_directory, 'output')
#track the number of figures created
Figure_Counter = 0

#%% Main Flow
def main():
    print("Retrieving Files from:", BBC_directory, "...")
    corpus = getCorpus(BBC_directory)
    
    print("Processing...")
    effectif, indexes, labels = determineCorpusDistribution(corpus)
    
    print("Plotting graphs...")
    generateBarGraph('Corpus Distribution', labels, effectif, indexes)
    generateBarGraph_Alternate('Corpus Distribution', labels, effectif)
    
    print("Done! Output is located in:", output_directory, ".")
    
    vectorizer = CountVectorizer()
    whole_data = vectorizer.fit_transform(corpus.data)
    
    corpus_by_classification = determineCorpusDetails_byClassification(corpus)

if __name__ == "__main__":
    begin_time = time.perf_counter()
    main()
    end_time = time.perf_counter()
    print("This script has taken", end_time - begin_time, "seconds to execute.")