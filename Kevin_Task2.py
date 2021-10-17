# -*- coding: utf-8 -*-
"""
Created on Sat Oct 16 15:36:14 2021

@author: Kevin
"""

#%% Imports
import os;

import pandas as pd
import matplotlib.pyplot as plt;
from collections import Counter;
from sklearn.model_selection import train_test_split;
from sklearn.naive_bayes import MultinomialNB;

#share config contents with other modules.
import configMP1

#%% Helper Functions Declaration
#Extract features and labels of input file
def readInput(input_file_directory):
    #load the file.
    data_full = pd.read_csv(drug200_directory)
    
    #features are picked from all but the last column.
    data_feature_headers = data_full.columns[:-1]
    data_features = data_full.loc[:, data_feature_headers]
    
    #labels are picked from the last column.
    data_label_header = data_full.columns[-1]
    data_labels = data_full.loc[:, data_label_header]
    
    return data_features, data_labels

#Determine the class distribution
def determineDistribution(target):
    target_counter = Counter(target)
    #Separate the keys from the values of the dict.
    labels, counts = zip(*sorted(target_counter.items()))
    
    return labels, counts

#Make a bar plot out of the distribution.
def generateBarGraph(title, labels, values): 
    #File format as specified in the mini-project document.
    file_format = "pdf"
    file_extension = "." + file_format
    
    plot_title = f"Figure {configMP1.getFigureCount()}. {title}"
    
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

# def convertDFColumnsTo(dataframe, cols, convertedType):
#     for col in cols:
#         dataframe[col] = dataframe[col].astype(convertedType)

#convert columns in the dataframe into nominal ones.
def convertDFColumnsToNominal(dataframe, cols):
    for col in cols:
        dataframe = pd.get_dummies(dataframe, columns=[col])
    return dataframe

#convert columns in the dataframe into ordinal ones.
def convertDFColumnsToOrdinal(dataframe, cols, ordered_categories):
    for col in cols:
        # dataframe[col] = dataframe[col].astype(convertedType)
        dataframe[col] = pd.Categorical(dataframe[col],
                                        categories=ordered_categories,
                                        ordered=True)
    return dataframe

#convert ordinal columns in the dataframe into numerical ones. Such that sklearn may make use of them.
def convertDFOrdinalToNumerical(dataframe):
    ordinal_cols = dataframe.select_dtypes(['category']).columns
    dataframe[ordinal_cols] = dataframe[ordinal_cols].apply(lambda ordinal_col: ordinal_col.cat.codes)
    return dataframe
    
#%% Configuration and Globals Declarations
#configuration and such
local_directory = configMP1.local_directory
#Read
drug200_directory = os.path.join(local_directory, 'drug200.csv')
nominal_columns = ['Sex']
ordinal_columns = ['BP', 'Cholesterol']
# There's an extra ordinal label, so that .cat.codes returns [1, 2, 3] for ["LOW", "NORMAL", "HIGH"]
ordinal_values = ["", "LOW", "NORMAL", "HIGH"]
distribution_graph_title = "drug200-distribution"
#Write
output_directory = os.path.join(local_directory, 'output')
output_performance_fullpath = os.path.join(output_directory, 'drug200-performance.txt')
#Misc.


#%% Main Flow
#Step 2, load input
print("Reading file from:", drug200_directory, "...")
data_features, data_labels = readInput(drug200_directory)

#Step 3, plot distribution
print("Processing read files and generating a distribution graph...")
generateBarGraph(distribution_graph_title, *determineDistribution(data_labels))

#Step 4, convert ordinal and nominal features into numerical format
data_features = convertDFColumnsToNominal(data_features, nominal_columns)
convertDFColumnsToOrdinal(data_features, ordinal_columns, ordinal_values)
convertDFOrdinalToNumerical(data_features)

#Step 5, split into train/test
#TODO: Placeholder
A = train_test_split(data_features, data_labels)

#Step 6, model training
#TODO: Placeholder
mNB = MultinomialNB()
mNB.fit(A[0], A[2])


