# -*- coding: utf-8 -*-
"""
Created on Sat Oct  2 12:44:22 2021.
Note: Do not have the produced figures open when running this script. It may 
cause conflict and have the script return 
'OSError: [Errno 22] Invalid argument: [...].png'.
My guess is that Python has trouble writing to the filename when it is in use 
by another program.

@author: Kevin
"""
#%% Imports
import os;
import sklearn; 
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter

#%% Helper Functions Declaration
#Retrieve data from the corpus
def getCorpus(corpus_directory):
    #Encoding as specified in the mini-project document.
    corpus_encoding = 'latin1'

    return sklearn.datasets.load_files(corpus_directory, encoding=corpus_encoding)

#Handle the data from the corpus
#In this case, determine the corpus distribution by classification
def handleCorpus(corpus):   
    #corpus_counter is a dict, where each key is the index of its corresponding classification name (from a different list).
    # so extract the values, and re-order the classification name by the key
    corpus_counter = Counter(corpus.target)
    #Sorry, I don't know an English word for 'effectif'. It's French for 'number of instances appearing in a group' in the context of statistics. Super convenient.
    effectif = corpus_counter.values()
    
    #np array plays better with a list than it does with dict_keys.
    indexes = list(corpus_counter.keys())
    #np arrays are easier to give arbitrary sorting,
    # so convert the names of categories into an np-array, then re-order them.
    labels = np.array((corpus.target_names))[indexes]
    
    return effectif, indexes, labels

#TODO: remove either of the generateBarGraph methods. Only one is needed. Choice between is in axis label order.

#Make a bar plot out of the corpus distribution.
def generateBarGraph(title, labels, values, value_indexes): 
    plot_title = f"Figure {getFigureCount()}. {title}"
    
    #in order arbitrarily decided by sklearn.datasets.load_files, seemingly alphabetical
    plt.barh(indexes, effectif, tick_label=labels, color="xkcd:battleship grey")
    #invert the category axis, since it's upside-down
    plt.gca().invert_yaxis()
    plt.title(plot_title)
    #provide values for each bar
    for count, value in enumerate(effectif):
        #value and count used as coordinates for the text.
        plt.text(value, indexes[count], str(value) + " ", va="center", ha="right", color="aliceblue")
    
    plt.savefig(os.path.join(output_directory, plot_title + ".png"), bbox_inches='tight', format="png")
    plt.show()

#Make a bar plot out of the corpus distribution.
def generateBarGraph_Alternate(title, labels, values): 
    plot_title = f"Figure {getFigureCount()}. {title}"
    
    #in order arbitrarily decided by Counter
    plt.barh(labels, effectif, color="xkcd:battleship grey")
    #invert the category axis, since it's upside-down
    plt.gca().invert_yaxis()
    plt.title(plot_title)
    #provide values for each bar
    for count, value in enumerate(effectif):
        #value and count used as coordinates for the text.
        plt.text(value, count, str(value) + " ", va="center", ha="right", color="aliceblue")
    
    plt.savefig(os.path.join(output_directory, plot_title + ".png"), bbox_inches='tight', format="png")
    plt.show()

#Call this when making a figure to track figure count.
def getFigureCount():
    global Figure_Counter
    Figure_Counter += 1
    return Figure_Counter

#main
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
#do stuff
print("Retrieving Files from:", BBC_directory, "...")
corpus = getCorpus(BBC_directory)

print("Processing...")
effectif, indexes, labels = handleCorpus(corpus)

print("Plotting graphs...")
generateBarGraph('Corpus Distribution', labels, effectif, indexes)
generateBarGraph_Alternate('Corpus Distribution', labels, effectif)

print("Done! Output is located in:", output_directory, ".")

