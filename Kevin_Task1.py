# -*- coding: utf-8 -*-
"""
Created on Sat Oct  2 12:44:22 2021.
Note 1: Do not have the produced figures open when running this script. It may 
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
from collections import Counter, defaultdict;
from sklearn.feature_extraction.text import CountVectorizer;
from sklearn.model_selection import train_test_split;
from sklearn.naive_bayes import MultinomialNB;
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, f1_score;

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
    corpus_collections = defaultdict(lambda: defaultdict(list));
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
    opened_outputfile.writelines(["(e)\n",
                                  ''.join([f"P({classification:<{highest_classification_length}}): {prior:4.2%}\n" for classification, prior in priors.items()])])
    
    #7 (f) Vocabulary size
    train_vocabulary_size = len(train_vocabulary)
    opened_outputfile.write(f"(f)\nIncluding only Vocabulary used in Training: {train_vocabulary_size}\n")
    
    #7 (g) Word count by class in the training data
    train_wordcount_by_classification = {class_names[classification_index]: int(count.sum())
                                         for classification_index, count 
                                         in zip(model.classes_, model.feature_count_)}
    opened_outputfile.writelines(["(g)\n",
                                  ''.join([f"{classification:<{highest_classification_length}}: {word_count:6d}\n" for classification, word_count in train_wordcount_by_classification.items()])])
    
    #7 (h) Word count total in the training data
    train_word_count_total = int(model.feature_count_.sum())
    opened_outputfile.write(f"(h)\n{train_word_count_total}\n")
    
    #7 (i) Words with frequency == 0 by class in the training data
    train_zero_frequencies_counts = {class_names[classification_index]: np.count_nonzero(count==0)
                                     for classification_index, count 
                                     in zip(model.classes_, model.feature_count_)}
    opened_outputfile.writelines(["(i)\n", 
                                  ''.join([f"{classification:<{highest_classification_length}}: {z_f_c:5d} {z_f_c/train_vocabulary_size:4.2%}\n" for classification, z_f_c in train_zero_frequencies_counts.items()])])
    
    #7 (j) Words with frequency == 1 in the training data
    train_one_frequency_count = np.count_nonzero(model.feature_count_.sum(axis=0)==1)
    opened_outputfile.write(f"(j)\n{train_one_frequency_count} {train_one_frequency_count/train_vocabulary_size:4.2%}\n")
    
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
#reason for choosing particular favorite words: 
#'year' is the most frequently used noun in the entire corpus (besides 'mr' and 36 other non-noun words) at 2309 times. 
# This measurement makes a distinction between singular and plurals with different spellings (eg 'years' appears 1003 times).
#'bbc', because the corpus is sourced from them, making the term a self-reference. Appears 767 times.
# Other strings containing 'bbc' seldom appear (cumulatively 14 times).
favorite_words = ['year', 'bbc']
#Write
output_directory = os.path.join(script_directory, 'output')
output_performance_fullpath = os.path.join(output_directory, 'bbc-performance.txt')
#Misc.
#track the number of figures created
Figure_Counter = 0

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
    print("Plotting distribution graphs...")
    generateBarGraph(distribution_graph_title, labels, effectif, indexes)
    # generateBarGraph_Alternate(distribution_graph_title + "_2", labels, effectif)
    
    print("Graph output is located in:", output_directory, ".")
    #Step 4 part 1
    print("Processing the corpus by classification...")
    corpus_by_classification = determineCorpusDetails_byClassification(corpus)
    corpus_vocabulary = corpus_by_classification["*Whole"]["vectorizer"].vocabulary_
    
    #Step 5
    print("Splitting into train and test data...")
    # train_corpus, test_corpus = splitTrainTestData_2(corpus, train_size_proportion)
    train_corpus, test_corpus = splitTrainTestData(corpus, train_size_proportion)
    
    print("Processing the training corpus by classification...")
    train_corpus_by_classification = determineCorpusDetails_byClassification(train_corpus)
    train_corpus_vocabulary = train_corpus_by_classification["*Whole"]["vectorizer"].vocabulary_
    print("Processing the testing corpus by classification...")
    test_corpus_by_classification = determineCorpusDetails_byClassification(test_corpus, train_corpus_vocabulary)
    
    #Step 6
    print("Training the model with the train data...")
    model = train_multinomialNB(train_corpus_by_classification["*Whole"]["document-term"], train_corpus["target"])
    print("Let the model predict the test data...")
    test_prediction = test_multinomialNB(model, test_corpus_by_classification["*Whole"]["document-term"])
    
    print("Generating report in:", output_performance_fullpath, "...")
    with open(output_performance_fullpath, 'w') as output_performance_file:
        #Step 7
        header = 'MultinomialNB default values, try 1'
        print("Generating report... (1/4)")
        generate_performance_report(test_corpus['target'], test_prediction, model, corpus.target_names, train_corpus_vocabulary, output_performance_file, header)
        
        #Step 8
        model_step8 = train_multinomialNB(train_corpus_by_classification["*Whole"]["document-term"], train_corpus["target"])
        header = 'MultinomialNB default values, try 2'
        print("Generating report... (2/4)")
        generate_performance_report(test_corpus['target'], test_prediction, model_step8, corpus.target_names, train_corpus_vocabulary, output_performance_file, header)
        
        #Step 9
        smoothing = 0.0001
        header = f'MultinomialNB with {smoothing} smoothing'
        model_step9 = train_multinomialNB(train_corpus_by_classification["*Whole"]["document-term"], train_corpus["target"], smoothing=smoothing)
        print("Generating report... (3/4)")
        generate_performance_report(test_corpus['target'], test_prediction, model_step9, corpus.target_names, train_corpus_vocabulary, output_performance_file, header)
        
        #Step 10
        smoothing = 0.9
        header = f'MultinomialNB with {smoothing} smoothing'
        model_step10 = train_multinomialNB(train_corpus_by_classification["*Whole"]["document-term"], train_corpus["target"], smoothing=smoothing)
        print("Generating report... (4/4)")
        generate_performance_report(test_corpus['target'], test_prediction, model_step10, corpus.target_names, train_corpus_vocabulary, output_performance_file, header)
        
    
    print("Done!")

if __name__ == "__main__":
    begin_time = time.perf_counter()
    main()
    end_time = time.perf_counter()
    print("This script has taken", end_time - begin_time, "seconds to execute.")