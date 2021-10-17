# -*- coding: utf-8 -*-
"""
Created on Sat Oct 16 15:36:14 2021

@author: Kevin
"""

#%% Imports
import os;
import time;
import statistics

import pandas as pd;
import numpy as np;
import matplotlib.pyplot as plt;
from collections import Counter, defaultdict;
from sklearn.model_selection import train_test_split;
from sklearn.naive_bayes import GaussianNB;
from sklearn.tree import DecisionTreeClassifier;
from sklearn.model_selection import GridSearchCV;
from sklearn.linear_model import Perceptron;
from sklearn.neural_network import MLPClassifier;
from sklearn.utils._testing import ignore_warnings;
from sklearn.exceptions import ConvergenceWarning, UndefinedMetricWarning;
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, f1_score;

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

#convert columns in the dataframe into nominal ones.
def convertDFColumnsToNominal(dataframe, cols):
    for col in cols:
        dataframe = pd.get_dummies(dataframe, columns=[col])
    return dataframe

#convert columns in the dataframe into ordinal ones.
def convertDFColumnsToOrdinal(dataframe, cols, ordered_categories):
    for col in cols:
        dataframe[col] = pd.Categorical(dataframe[col],
                                        categories=ordered_categories,
                                        ordered=True)
    return dataframe

#convert ordinal columns in the dataframe into numerical ones. Such that sklearn may make use of them.
def convertDFOrdinalToNumerical(dataframe):
    ordinal_cols = dataframe.select_dtypes(['category']).columns
    dataframe[ordinal_cols] = dataframe[ordinal_cols].apply(lambda ordinal_col: ordinal_col.cat.codes)
    return dataframe

#Split and return the data into train and test corpuses.
def splitTrainTestData(features, labels):
    #Split and shuffle according to default parameters.
    data_features_train, data_features_test, data_labels_train, data_labels_test = train_test_split(features, labels)
    
    return {'feature': data_features_train, 'label': data_labels_train}, {'feature': data_features_test, 'label': data_labels_test}

#Instantiate the models into a list.
def instantiateModels():
    models = {}
    #baseMLP parameters as specified in the mini-project.
    baseMLP_hiddenlayers = (100,)
    baseMLP_activation = 'logistic'
    baseMLP_solver = 'sgd'
    
    #Declare models.
    gNB = GaussianNB()
    baseDT = DecisionTreeClassifier()
    topDT = GridSearchCV(DecisionTreeClassifier(), topDT_param, scoring=topDT_Scoring)
    per = Perceptron()
    baseMLP = MLPClassifier(baseMLP_hiddenlayers, baseMLP_activation, solver=baseMLP_solver)
    topMLP = GridSearchCV(MLPClassifier(), topMLP_param, scoring=topDT_Scoring)
    
    #Bind models to output.
    models["NB"] = gNB
    models["Base-DT"] = baseDT
    models["Top-DT"] = topDT
    models["PER"] = per
    models["Base-MLP"] = baseMLP
    models["Top-MLP"] = topMLP
    
    return models

#Batch train the models with the train data. Ignore convergence warning of the MLPs.
@ignore_warnings(category=ConvergenceWarning)
def trainModels(models, data_train):
    for model in models.values():
        model.fit(*data_train.values())

#Batch predict against test features with the models.
def modelsPredict(models, data_features_test):
    return {key: model.predict(data_features_test) for key, model in models.items()}

#Batch instantiate, train and predict with models with train and test data.
def instantiateTrainPredictModels(data_train, data_features_test):
    #instantiate
    models = instantiateModels()
    #train
    trainModels(models, data_train)
    #predict
    predictions = modelsPredict(models, data_features_test)
    
    return models, predictions

#Compute Accuracy, Macro&Weighted F1.
def computeMetrics(y_true, y_pred):
    return {"Accuracy       ": accuracy_score(y_true, y_pred),
            "Macro-avg    F1": f1_score(y_true, y_pred, average="macro"),
            "Weighted-avg F1": f1_score(y_true, y_pred, average="weighted")}
    
#Assemble the lines to display as the header.
def constructHeader(header_title, model):
    #header_length is the length between the spacers on the side.
    header_length = len(header_title)
    additional_header_content = []
    #special considerations for GridSearchCV models.
    if isinstance(model, GridSearchCV):
        #length to the left of the colon.
        longest_length_param = max(map(len, model.best_params_.keys()))
        #length to the right of the colon (omitting space).
        longest_length_value = max(map(len, [str(value) for value in model.best_params_.values()]))
        #length as a whole, add two to account for the colon-space.
        header_length = max(header_length, longest_length_param + longest_length_value + 2)
        for parameter, value in model.best_params_.items():
            #construct the inner text.
            info = f"{parameter:<{longest_length_param}}: {str(value):>{longest_length_value}}"
            #pad with spacers on the side.
            info_line = f"**** {info + ' ':*<{header_length+5}}\n"
            #remember it.
            additional_header_content.append(info_line)
    
    #Congregate the lines to output.
    header_lines = ["(a)\n",
                    f"{'*'*(header_length+10)}\n",
                    f"**** {header_title + ' ':*<{header_length+5}}\n"]
    header_lines += additional_header_content
    header_lines.append(f"{'*'*(header_length+10)}\n")
    
    return header_lines

#Generate a full section of the report for a given model. Ignore undefined metric warnings due to lack of predictions for some labels.
@ignore_warnings(category=UndefinedMetricWarning)
def generate_performance_report_iteration(y_true, y_pred, model, opened_outputfile, header_title):
    #7. (a)Header
    opened_outputfile.writelines(constructHeader(header_title, model))
    
    #7. (b) Confusion Matrix
    #order the labels in ascending order
    cm = confusion_matrix(y_true, y_pred, labels=model.classes_)
    opened_outputfile.writelines(["(b)\n",
                                  "Vertical axis shows predicted labels; Horizontal axis show true labels. \n",
                                  "ie first row shows a predicted label, first column shows a true labael.\n",
                                  "Columns (top to bottom) and Rows (left to right) are ordered thusly: \n",
                                  f"{model.classes_}.\n"])
    #save numpy-related entities with numpy's function.
    np.savetxt(opened_outputfile, cm, fmt="%3d")
    
    #7. (c) Precision, Recall, F1-measure
    opened_outputfile.write("(c)\n")
    report = classification_report(y_true, y_pred, target_names=model.classes_)
    #classification_report's output has excess lines, so remove those.
    truncated_report = ''.join(report.splitlines(keepends=True)[:-4])
    opened_outputfile.write(truncated_report)
    
    #7. (d) accuracy, macro-average F1 and weighted-average F1
    # Seems like these are some of the values truncated out in 7.(c). Specifications suggest to obtain the metrics through these methods though.
    opened_outputfile.writelines(["(d)\n"] +
                                  [f"{metric}: {str(value)}\n" for metric, value in computeMetrics(y_true, y_pred).items()])
    opened_outputfile.write("\n" * 2)

#Generate the full report for the given models.
def generate_performance_report(y_true, y_preds, models, opened_outputfile):
    for model_key, model in models.items():
        generate_performance_report_iteration(y_true, y_preds[model_key], model, opened_outputfile, model_key)

#Generate the statistical report for the given models.
def generate_stability_performance_report(stats, opened_outputfile):
    opened_outputfile.write("Step 8\n")
    for model_names, metrics in stats.items():
        opened_outputfile.write("{model_names}:\n")
        opened_outputfile.writelines([f"\t{metric}: {stat}\n" for metric, stat in metrics.items()])
    opened_outputfile.write("\n"*2)

#%% Configuration and Globals Declarations
#configuration and such
local_directory = configMP1.local_directory
#Read
drug200_directory = os.path.join(local_directory, 'drug200.csv')
nominal_columns = ['Sex']
ordinal_columns = ['BP', 'Cholesterol']
#GridSearch parameters
#Reasoning for weighted recall as scoring:
#1. There is an imbalanced distribution, so accuracy is out and weighted metric is preferred.
#2. Better recall(about false negatives) is more important than precision (about false positives), reasoning:
#-Pretend the model can return several drug suggestions. Suppose it recommends
#drug A to everyone. This results in low precision, but we don't care much about
#handing out too many of a drug. Low precision is not a negative, so we don't 
#care about it.
#-Suppose it instead recommends all the drugs except the one the patient needs. This is bad. 
#We want our model to give the drug the patient needs. The metric that becomes low 
#is recall. Low recall is bad, we care about it.
#-F1 measure balances the two, but we only care about one of them, so it is not preferred.
#Therefore, weighted recall is the preferred scoring metric.
topDT_Scoring = 'recall_weighted'
topDT_param = {'criterion': ['gini','entropy'],
               'max_depth': [None, 5],
               'min_samples_split': [2, 4, 8]
}
topMLP_Scoring = 'recall_weighted'
topMLP_param = {'activation': ['logistic', 'tanh', 'relu', 'identity'],
               'hidden_layer_sizes': [(30,50), (10,10,10)],
               'solver':['adam', 'sgd']}
# There's an extra ordinal label, so that .cat.codes returns [1, 2, 3] for ["LOW", "NORMAL", "HIGH"]
ordinal_values = ["", "LOW", "NORMAL", "HIGH"]
distribution_graph_title = "drug200-distribution"
step8_iteration_count = 4
#Write
output_directory = os.path.join(local_directory, 'output')
output_performance_fullpath = os.path.join(output_directory, 'drug200-performance.txt')
#Misc.


#%% Main Flow
def main():
    #Step 2, load input
    print("Reading file from:", drug200_directory, "...")
    data_features, data_labels = readInput(drug200_directory)
    
    #Step 3, plot distribution
    print("Processing read files and generating a distribution graph...")
    generateBarGraph(distribution_graph_title, *determineDistribution(data_labels))
    
    print("Graph output is located in:", output_directory, ".")
    #Step 4, convert ordinal and nominal features into numerical format
    print("Converting data into parsable format...")
    data_features = convertDFColumnsToNominal(data_features, nominal_columns)
    convertDFColumnsToOrdinal(data_features, ordinal_columns, ordinal_values)
    convertDFOrdinalToNumerical(data_features)
    
    #Step 5, split into train/test
    print("Split data into train and test data.")
    data_train, data_test = splitTrainTestData(data_features, data_labels)
    
    #Step 6, model training
    print("Instantiating models...")
    models = instantiateModels()
    print("Training models... (This may take a while!)")
    trainModels(models, data_train)
    print("Let the models predict the test data...")
    predictions = modelsPredict(models, data_test['feature'])
    
    #Step 8 part 1
    #Instantiate, Train, Predict and Evaluate the models a number of times.
    print("Repeat to gather statistical performance...")
    for iteration in range(step8_iteration_count):
        print("On iteration", iteration, " of", step8_iteration_count, "...")
        runs = []
        predictions = instantiateTrainPredictModels(data_train, data_test['feature'])[1]
        runs.append({model_name: computeMetrics(data_test['label'], prediction) for model_name, prediction in predictions.items()})
    
    #Aggregate the evaluation metrics.
    print("Aggregating data from the runs...")
    #Structure is a list in a dict in a dict
    aggregates = defaultdict(lambda: defaultdict(list));
    for run in runs:
        for model_name, metrics in run.items():
            for metric, value in metrics.items():
                aggregates[model_name][metric].append(value)
    
    #Find statistics of the aggregated.
    print("Performing statistics on the aggregated...")
    #Structure is a float in a dict in a dict in a dict
    stats = defaultdict(lambda: defaultdict(lambda: defaultdict(float)));
    for model_name, aggregate in aggregates.items():
        for metric, values in aggregate.items():
            mean = statistics.mean(values)
            stats[model_name][metric]['mean'] = mean
            stats[model_name][metric]['pstd'] = statistics.pstdev(values, mean)
    
    #Step 7, Generate report
    print("Generating report in:", output_performance_fullpath, "...")
    with open(output_performance_fullpath, 'w') as output_performance_file:
        print("Generating report on sample result of the models' evaluation...")
        generate_performance_report(data_test['label'], predictions, models, output_performance_file)
        #Step 8 part 2
        print("Generating report on statistical find of the models' evaluation...")
        generate_stability_performance_report(stats, output_performance_file)
    
    print("Done!")

if __name__ == "__main__":
    begin_time = time.perf_counter()
    main()
    end_time = time.perf_counter()
    print("This script has taken", end_time - begin_time, "seconds to execute.")