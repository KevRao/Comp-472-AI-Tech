# -*- coding: utf-8 -*-
"""
Created on Sat Oct  2 12:44:22 2021.
Note 1: Do not have the produced figures open when running this script. It may 
cause conflict and have the script return the following error:
'OSError: [Errno 22] Invalid argument: [...].png'.
My guess is that Python has trouble writing to the filename when it is in use 
by another program.
** Since changing the extension to .pdf, this error has not yet appeared.

@author: Kevin
"""
#%% Imports
import os;
import time;

from sklearn import datasets;
import numpy as np;
import matplotlib.pyplot as plt;
from collections import Counter, defaultdict;
from sklearn.feature_extraction.text import CountVectorizer;
from sklearn.model_selection import train_test_split;
from sklearn.naive_bayes import MultinomialNB;
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, f1_score;

#share config contents with other modules.
import configMP1

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

#Get the document term with a CountVectorizer.
#Optional vocabulary should be given for the test corpus, such that it matches the train corpus.
def determineCorpusDetails(corpus, vocabulary=None):
    #Structure is a list in a dict
    corpus_collections = defaultdict(list);

    #create a new instance of countVectorizer
    whole_vectorizer = CountVectorizer(vocabulary=vocabulary)
    #bind the corpus data
    corpus_collections['data'] = corpus['data']
    #attach the new countVectorizer's reference.
    corpus_collections['vectorizer'] = whole_vectorizer
    #process the countVectorizer to the data
    corpus_collections['document-term'] = whole_vectorizer.fit_transform(corpus['data'])
    
    return corpus_collections

#Split and return the corpus into train and test corpuses, but without their filenames.
def splitTrainTestData(corpus, train_size_proportion):
    #Random_State as specified in the mini-project document.
    random_state = None
    
    #Split and shuffle according to given proportion.
    training_data, testing_data, training_labels, testing_labels = train_test_split(corpus.data, corpus.target, train_size = train_size_proportion, random_state=random_state)
    
    #Missing the filenames, but we probably won't ever need them.
    train_corpus = {
        "data": training_data, 
        "target": training_labels, 
        "DESCR": corpus["DESCR"] + "Training data. ",
        "target_names": corpus["target_names"]
    }
    
    test_corpus = {
        "data": testing_data, 
        "target": testing_labels, 
        "DESCR": corpus["DESCR"] + "Testing data. ",
        "target_names": corpus["target_names"]
    }
    
    return train_corpus, test_corpus

#Train a multinomial naive bayes model. the default smoothing is 1.0.
def train_multinomialNB(train_data, labels, smoothing=1.0):
    mNB = MultinomialNB(alpha=smoothing)
    mNB.fit(train_data, labels)
    return mNB

#Return the prediction of a model against test data.
def test_multinomialNB(mNB, test_data):
    return mNB.predict(test_data)

#Train a multinomial naive bayes model, then return it along with its prediction against the test data.
def train_test_multinomialNB(train_data, labels, test_data, smoothing=1.0):
    #train
    mNB = train_multinomialNB(train_data, labels, smoothing)
    #test
    prediction = test_multinomialNB(mNB, test_data)
    return mNB, prediction

#Generate a full section of the report for a given model.
def generate_performance_report(y_true, y_pred, model, class_names, train_vocabulary, opened_outputfile, header_title):
    #for formatting output
    highest_classification_length = max(map(len, class_names))
    
    #7. (a)Header
    opened_outputfile.writelines([
        "(a)\n",
        f"{'*'*(len(header_title)+10)}\n",
        f"**** {header_title} ****\n",
        f"{'*'*(len(header_title)+10)}\n"])
    #7. (b) Confusion Matrix
    #order the labels in ascending order
    cm = confusion_matrix(y_true, y_pred, labels=range(len(class_names)))
    opened_outputfile.writelines(["(b)\n",
                                  "Vertical axis shows predicted labels; Horizontal axis show true labels. \n",
                                  "ie first row shows a predicted label, first column shows a true labael.\n",
                                  "Columns (top to bottom) and Rows (left to right) are ordered thusly: \n",
                                  f"{class_names}.\n"])
    #save numpy-related entities with numpy's function.
    np.savetxt(opened_outputfile, cm, fmt="%3d")
    
    #7. (c) Precision, Recall, F1-measure
    opened_outputfile.write("(c)\n")
    report = classification_report(y_true, y_pred, target_names=class_names)
    #classification_report's output has excess lines, so remove those.
    truncated_report = ''.join(report.splitlines(keepends=True)[:-4])
    opened_outputfile.write(truncated_report)
    
    #7. (d) accuracy, macro-average F1 and weighted-average F1
    # Seems like these are some of the values truncated out in 7.(c). Specifications suggest to obtain the metrics through these methods though.
    opened_outputfile.writelines(["(d)\n",
                                  "Accuracy       : ", str(accuracy_score(y_true, y_pred)), "\n",
                                  "Macro-avg    F1: ", str(f1_score(y_true, y_pred, average="macro")), "\n",
                                  "Weighted-avg F1: ", str(f1_score(y_true, y_pred, average="weighted")), "\n"])
    
    #7 (e) Priors in the training data
    #Prior is calculated by dividing the number of documents in a class by the total number of documents.
    total_train_count = int(model.class_count_.sum())
    priors = {class_names[classification_index]: count/total_train_count 
              for classification_index, count 
              in zip(model.classes_, model.class_count_)}
    opened_outputfile.writelines(["(e)\nPriors:\n",
                                  ''.join([f"P({classification:<{highest_classification_length}}): {prior:4.2%}\n" for classification, prior in priors.items()])])
    
    #7 (f) Vocabulary size
    train_vocabulary_size = len(train_vocabulary)
    opened_outputfile.write(f"(f)\nNumber of unique words:\nIncluding only Vocabulary used in Training: {train_vocabulary_size}\n")
    
    #7 (g) Word count by class in the training data
    train_wordcount_by_classification = {class_names[classification_index]: int(count.sum())
                                         for classification_index, count 
                                         in zip(model.classes_, model.feature_count_)}
    opened_outputfile.writelines(["(g)\nNumber of words each:\n",
                                  ''.join([f"{classification:<{highest_classification_length}}: {word_count:6d}\n" for classification, word_count in train_wordcount_by_classification.items()])])
    
    #7 (h) Word count total in the training data
    train_word_count_total = int(model.feature_count_.sum())
    opened_outputfile.write(f"(h)\nNumber of words total:\n{train_word_count_total}\n")
    
    #7 (i) Words with frequency == 0 by class in the training data
    train_zero_frequencies_counts = {class_names[classification_index]: np.count_nonzero(count==0)
                                     for classification_index, count 
                                     in zip(model.classes_, model.feature_count_)}
    opened_outputfile.writelines(["(i)\nWords with zero frequency each:\n",
                                  ''.join([f"{classification:<{highest_classification_length}}: {z_f_c:5d} {z_f_c/train_vocabulary_size:4.2%}\n" for classification, z_f_c in train_zero_frequencies_counts.items()])])
    
    #7 (j) Words with frequency == 1 in the training data
    train_one_frequency_count = np.count_nonzero(model.feature_count_.sum(axis=0)==1)
    opened_outputfile.write(f"(j)\nWords with one frequency total:\n{train_one_frequency_count} {train_one_frequency_count/train_vocabulary_size:4.2%}\n")
    
    #7 (k) Favorite word appearance
    opened_outputfile.write("(k)\nFavorite words are:\n")
    opened_outputfile.writelines([favorite_word + ", " for favorite_word in favorite_words])
    opened_outputfile.write("\n")
    # Some help for below list-comprehension
    #train_vocabulary[favorite_word]                is assigned index of the favorite word used in the vocabulary.
    #word_log_prob[train_vocabulary[favorite_word]] is the log-prob computed by the model of the favorite word.
    #class_names[classification_index]              is the class' name
    # The colon inside the brackets are for formatting.
    # Also, the model stores the log prob in base e (ie natural logarithm). This applies the same to its log prob for class priors.
    opened_outputfile.writelines([f"ln(P({favorite_word:<{favorite_word_max_length}}|{class_names[classification_index]:<{highest_classification_length}}))= {word_log_prob[train_vocabulary[favorite_word]]}\n"
                                  for classification_index, word_log_prob 
                                  in zip(model.classes_, model.feature_log_prob_) 
                                  for favorite_word 
                                  in favorite_words])
    
    opened_outputfile.write("\n" * 2)

#Make a bar plot out of the corpus distribution.
def generateBarGraph(title, labels, values, value_indexes): 
    #File format as specified in the mini-project document.
    file_format = "pdf"
    file_extension = "." + file_format
    
    plot_title = f"Figure {configMP1.getFigureCount()}. {title}"
    
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

#%% Configuration and Globals Declarations
#configuration and such
local_directory = configMP1.local_directory
#Read
BBC_directory = os.path.join(local_directory, 'BBC')
distribution_graph_title = "BBC-distribution"
train_size_proportion = 0.80
#reason for choosing particular favorite words: 
#'year' is the most frequently used noun in the entire corpus (besides 'mr' and 36 other non-noun words) at 2309 times. 
# This measurement makes a distinction between singular and plurals with different spellings (eg 'years' appears 1003 times).
#'bbc', because the corpus is sourced from them, making the term a self-reference. Appears 767 times.
# Other strings containing 'bbc' seldom appear (cumulatively 14 times).
favorite_words = ['year', 'bbc']
#Write
output_directory = os.path.join(local_directory, 'output')
output_performance_fullpath = os.path.join(output_directory, 'bbc-performance.txt')
#Misc.
favorite_word_max_length = max(map(len, favorite_words))

#%% Main Flow
def main():
    #Step 3
    print("Retrieving files from:", BBC_directory, "...")
    corpus = getCorpus(BBC_directory)
    #Step 2 part 1
    print("Processing read files...")
    effectif, indexes, labels = determineCorpusDistribution(corpus)
    #Step 2 part 2
    print("Plotting distribution graph...")
    generateBarGraph(distribution_graph_title, labels, effectif, indexes)
    
    print("Graph output is located in:", output_directory, ".")
    #Step 4 part 1
    # However, the model must never see data of the test set, so including words that the training set should not see is bad.
    print("Processing the corpus...")
    corpus_details = determineCorpusDetails(corpus)
    corpus_vocabulary = corpus_details["vectorizer"].vocabulary_
    
    #Step 5
    print("Splitting into train and test data...")
    train_corpus, test_corpus = splitTrainTestData(corpus, train_size_proportion)
    
    print("Processing the training corpus...")
    train_corpus_details = determineCorpusDetails(train_corpus)
    train_corpus_vocabulary = train_corpus_details["vectorizer"].vocabulary_
    print("Processing the testing corpus...")
    test_corpus_details = determineCorpusDetails(test_corpus, train_corpus_vocabulary)
    
    #Step 6
    print("Training the model with the train data and computing its predictions...")
    model, test_prediction = train_test_multinomialNB(train_corpus_details["document-term"], train_corpus["target"], test_corpus_details["document-term"])
    
    print("Generating report in:", output_performance_fullpath, "...")
    with open(output_performance_fullpath, 'w') as output_performance_file:
        #Step 7
        header = 'MultinomialNB default values, try 1'
        print("Generating report... (1/4)")
        generate_performance_report(test_corpus['target'], test_prediction, model, corpus.target_names, train_corpus_vocabulary, output_performance_file, header)
        
        #Step 8
        header = 'MultinomialNB default values, try 2'
        print("Training the model with the train data and computing its predictions...")
        model_step8, test_prediction_step8 = train_test_multinomialNB(train_corpus_details["document-term"], train_corpus["target"], test_corpus_details["document-term"])
        print("Generating report... (2/4)")
        generate_performance_report(test_corpus['target'], test_prediction_step8, model_step8, corpus.target_names, train_corpus_vocabulary, output_performance_file, header)
        #Step 9
        smoothing = 0.0001
        header = f'MultinomialNB with {smoothing} smoothing'
        print("Training the model with the train data and computing its predictions...")
        model_step9, test_prediction_ste9 = train_test_multinomialNB(train_corpus_details["document-term"], train_corpus["target"], test_corpus_details["document-term"], smoothing=smoothing)
        print("Generating report... (3/4)")
        generate_performance_report(test_corpus['target'], test_prediction_ste9, model_step9, corpus.target_names, train_corpus_vocabulary, output_performance_file, header)
        
        #Step 10
        smoothing = 0.9
        header = f'MultinomialNB with {smoothing} smoothing'
        print("Training the model with the train data and computing its predictions...")
        model_step10, test_prediction_step10 = train_test_multinomialNB(train_corpus_details["document-term"], train_corpus["target"], test_corpus_details["document-term"], smoothing=smoothing)
        print("Generating report... (4/4)")
        generate_performance_report(test_corpus['target'], test_prediction_step10, model_step10, corpus.target_names, train_corpus_vocabulary, output_performance_file, header)
    
    print("Done!")

if __name__ == "__main__":
    begin_time = time.perf_counter()
    main()
    end_time = time.perf_counter()
    print("This script has taken", end_time - begin_time, "seconds to execute.")