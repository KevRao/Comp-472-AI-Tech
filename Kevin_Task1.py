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
from sklearn.feature_extraction.text import CountVectorizer;
from sklearn.model_selection import train_test_split;
from sklearn.naive_bayes import MultinomialNB;

#%% Helper Functions Declaration
#Retrieve data from the corpus
def getCorpus(corpus_directory):
    #Encoding as specified in the mini-project document.
    corpus_encoding = 'latin1'
    description = "Corpus of BBC articles. "

    return datasets.load_files(corpus_directory, encoding=corpus_encoding, description=description)

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

#Optional vocabulary should be given for train/test corpuses, such that it matches the whole, entire corpus.
# Note: vocabulary applies to all classifications at the moment. It isn't yet a vocabulary by classification.
def determineCorpusDetails_byClassification(corpus, vocabulary=None):
    #Organize the corpus data into its classifications
    #Structure is a list in a dict in a dict
    corpus_collections = collections.defaultdict(lambda: collections.defaultdict(list));
    #bind the corpus data to its classification's data entry
    for target_index, data in zip(corpus['target'], corpus['data']):
        corpus_collections[corpus['target_names'][target_index]]['data'].append(data)
    
    #Initiate countVectorizer for each classification
    for corpus_collection in corpus_collections.values():
        #create a new instance of countVectorizer
        collection_vectorizer = CountVectorizer(vocabulary=vocabulary)
        #attach the new countVectorizer's reference.
        corpus_collection['vectorizer'].append(collection_vectorizer)
        #process the countVectorizer to the data
        corpus_collection['document-term'] = collection_vectorizer.fit_transform(corpus_collection['data'])
    
    #do the same, but for the single instance of the whole corpus
    whole_vectorizer = CountVectorizer(vocabulary=vocabulary)
    corpus_collections['*Whole']['data'] = corpus['data']
    corpus_collections['*Whole']['vectorizer'] = whole_vectorizer
    corpus_collections['*Whole']['document-term'] = whole_vectorizer.fit_transform(corpus['data'])
    
    return corpus_collections

#Split and return the data&labels into train and test lists.
def splitTrainTestData(data, labels, train_size_proportion):
    #Random_State as specified in the mini-project document.
    random_state = None
    
    #Split and shuffle according to given proportion.
    #can just be returned directly, but this helps readability of what the output variables are (since there're 4 of them).
    training_data, testing_data, training_labels, testing_labels = train_test_split(data, labels, train_size = train_size_proportion, random_state=random_state)
    return training_data, testing_data, training_labels, testing_labels

#Split and return the corpus into train and test corpuses.
def splitTrainTestData_2(corpus, train_size_proportion):
    #Random_State as specified in the mini-project document.
    random_state = None
    
    data_index_selection = labels_index_selection = range(len(corpus.data))
    #Split and shuffle according to given proportion.
    split_shuffled_indexes = train_test_split(data_index_selection, labels_index_selection, train_size = train_size_proportion, random_state=random_state)
    # if (   split_shuffled_indexes[0] != split_shuffled_indexes[2]
    #     or split_shuffled_indexes[1] != split_shuffled_indexes[3]):
    #     raise AssertionError("Somehow train_test_split did not return corresponding splits for train/test.")
    assert  split_shuffled_indexes[0] == split_shuffled_indexes[2] \
        and split_shuffled_indexes[1] == split_shuffled_indexes[3] \
        , "Somehow train_test_split did not return corresponding splits for train/test."
    
    #try_corpus = {split_corpus for data_index, split_corpus in enumerate(corpus) if data_index in split_shuffled_indexes[0]}
    #Copy over the corpus' entries for the corresponding indices. For both the train and test lists.
    train_corpus = collections.defaultdict(list)
    for data_index in split_shuffled_indexes[0]:
        train_corpus["data"     ].append(corpus.data     [data_index])
        train_corpus["filenames"].append(corpus.filenames[data_index])
        train_corpus["target"   ].append(corpus.target   [data_index])
    train_corpus["DESCR"] = corpus.DESCR + "Training data. "
    train_corpus["target_names"] = corpus.target_names
    
    test_corpus = collections.defaultdict(list)
    for data_index in split_shuffled_indexes[1]:
        test_corpus["data"     ].append(corpus.data     [data_index])
        test_corpus["filenames"].append(corpus.filenames[data_index])
        test_corpus["target"   ].append(corpus.target   [data_index])
    test_corpus["DESCR"] = corpus.DESCR + "Testing data. "
    test_corpus["target_names"] = corpus.target_names
    
    return train_corpus, test_corpus

def train_multinomialNB(train_data, labels):
    mNB = MultinomialNB()
    mNB.fit(train_data, labels)
    return mNB

#I haven't watched the evaluation lectures, so I can yet do this fully >.<
def test_multinomialNB(mNB, test_data):
    return mNB.predict(test_data)

#TODO: remove either of the generateBarGraph methods. Only one is needed. Choice between is in axis label order.

#Make a bar plot out of the corpus distribution.
def generateBarGraph(title, labels, values, value_indexes): 
    #File format as specified in the mini-project document.
    file_format = "pdf"
    file_extension = "." + file_format
    
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
    
    plt.savefig(os.path.join(output_directory, title + file_extension), bbox_inches='tight', format=file_format)
    plt.show()

#Make a bar plot out of the corpus distribution.
def generateBarGraph_Alternate(title, labels, values): 
    #File format as specified in the mini-project document.
    file_format = "pdf"
    file_extension = "." + file_format
    
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
    
    plt.savefig(os.path.join(output_directory, title + file_extension), bbox_inches='tight', format=file_format)
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
distribution_graph_title = "BBC-distribution"
train_size_proportion = 0.80
#Write
output_directory = os.path.join(script_directory, 'output')
#track the number of figures created
Figure_Counter = 0

#%% Main Flow
def main():
    #TODO: Delete these global statements below. Debug purposes only.
    global corpus
    global split_train_test_corpuses
    global corpus_by_classification
    global train_corpus_by_classification
    global model
    global test_prediction
    
    #Step 3
    print("Retrieving files from:", BBC_directory, "...")
    corpus = getCorpus(BBC_directory)
    #Step 2 part 1
    print("Processing read files...")
    effectif, indexes, labels = determineCorpusDistribution(corpus)
    #Step 2 part 2
    print("Plotting distribution graphs...")
    generateBarGraph(distribution_graph_title, labels, effectif, indexes)
    generateBarGraph_Alternate(distribution_graph_title + "_2", labels, effectif)
    
    print("Graph output is located in:", output_directory, ".")
    #Step 4 part 1
    print("Processing the corpus by classification...")
    corpus_by_classification = determineCorpusDetails_byClassification(corpus)
    corpus_vocabulary = corpus_by_classification["*Whole"]["vectorizer"].vocabulary_
    #Step 5
    print("Splitting into train and test data...")
    split_train_test_corpuses = splitTrainTestData_2(corpus, train_size_proportion)
    
    print("Processing the training corpus by classification...")
    train_corpus_by_classification = determineCorpusDetails_byClassification(split_train_test_corpuses[0], corpus_vocabulary)
    print("Processing the testing corpus by classification...")
    test_corpus_by_classification = determineCorpusDetails_byClassification(split_train_test_corpuses[1], corpus_vocabulary)
    
    #Step 6
    print("Training the model with the train data...")
    model = train_multinomialNB(train_corpus_by_classification["*Whole"]["document-term"], split_train_test_corpuses[0]["target"])
    print("Let the model predict the test data...")
    test_prediction = test_multinomialNB(model, test_corpus_by_classification["*Whole"]["document-term"])
    
    
    print("Done! For now.")

if __name__ == "__main__":
    begin_time = time.perf_counter()
    main()
    end_time = time.perf_counter()
    print("This script has taken", end_time - begin_time, "seconds to execute.")