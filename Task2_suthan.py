#Suthan Sinnathurai
#40086318
#Task 2

import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from collections import Counter
from sklearn.tree import DecisionTreeClassifier,plot_tree
from sklearn.metrics import accuracy_score,f1_score,precision_score,recall_score,confusion_matrix,classification_report
from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import Perceptron
from sklearn.neural_network import MLPClassifier



drug_data = pd.read_csv('drug200.csv',delimiter=',')

#Step 3
counter =Counter(drug_data['Drug'])
drug_name = counter.keys()
drug_name_counts = counter.values()

indexes = np.arange(len(drug_name))
width = 0.7
print(drug_name_counts)
plt.bar(indexes,drug_name_counts,width)
plt.xticks(indexes +width*0.5,drug_name)
plt.xlabel("Drug_type")
plt.ylabel("Count")
plt.title("Bar graph of the number of instances in each class of Drug_type")

plt.savefig("drug-distribution.pdf")
plt.show()

#step 4


#ordinal values 
#drug_data.Sex=pd.Categorical(drug_data.Sex,['F','M'],ordered = False)
drug_data.BP=pd.Categorical(drug_data.BP,['LOW','NORMAL','HIGH'],ordered = True)
drug_data.Cholesterol=pd.Categorical(drug_data.Cholesterol,['NORMAL','HIGH'],ordered = True)
#drug_data.Drug=pd.Categorical(drug_data.Drug,['drugY','drugC','drugX','drugA','drugB'],ordered = False)

drug_data.BP = drug_data.BP.cat.codes
drug_data.Cholesterol = drug_data.Cholesterol.cat.codes
#drug_data.Sex = drug_data.Sex.cat.codes
#drug_data.Drug = drug_data.Drug.cat.codes

drug_data['Sex'] =pd.get_dummies(drug_data['Sex'])
print(drug_data)

#step 5
#not working yet
y=drug_data['Drug']
x=drug_data[['Age','Sex','BP','Cholesterol','Na_to_K']].values
target_names = ['drugY','drugC','drugX','drugA','drugb']

X_train,X_test,Y_train,Y_test= train_test_split(x,y,random_state=1)
print(X_train.shape)
print(X_test.shape)
print(Y_train.shape)
print(Y_test.shape)

#configuration and such
script_directory = os.path.dirname(__file__)
#Write
output_directory = os.path.join(script_directory, 'output')
output_performance_fullpath = os.path.join(output_directory, 'drugs-performance.txt')

#step 6 


#a)Gaussian Naive bayes classifier
NB=GaussianNB()
NB.fit(X_train, Y_train)
predicted_NB = NB.predict(X_test)

print(predicted_NB)



#B)Decision tree
drugTree = DecisionTreeClassifier(criterion="entropy", max_depth=4)

drugTree.fit(X_train, Y_train)
predicted = drugTree.predict(X_test)

print(predicted)



#c) Top-DT

param_dict={
    "criterion":['gini','entropy'],
    "max_depth":range(5,6),
    "min_samples_split":range(2,4)
}

dt_grid = GridSearchCV(
    drugTree ,
    param_grid=param_dict,
    cv=10,
    verbose=1,
    n_jobs=1

)
dt_grid.fit(X_train,Y_train)
print(dt_grid.best_params_)

drugTree_top = DecisionTreeClassifier(criterion="gini", max_depth=5, min_samples_split=2)

drugTree_top.fit(X_train, Y_train)
predicted_topDt = drugTree_top.predict(X_test)





#d) Perceptron
pt = Perceptron()
pt.fit(X_train,Y_train)

predicted_pt = pt.predict(X_test)

print(predicted_pt)

print("\n Pereptron's Accuracy: ", accuracy_score(Y_test, predicted_pt))


#e)Base-MLP
b_MLP = MLPClassifier(hidden_layer_sizes=(100,),activation='logistic',solver ='sgd',max_iter=200 )
b_MLP.fit(X_train,Y_train)

predicted_MLP = b_MLP.predict(X_test)

print(predicted_MLP)

print("\n Base-MLP Pereptron's Accuracy: ", accuracy_score(Y_test, predicted_MLP))

#f)top-MLP

param_dict_MLP={
    "activation":['logistic','tanh','relu','identity'],
    "hidden_layer_sizes":[(30,50),(10,10,10)],
    "solver":['sgd','adam']
}

MLP_grid = GridSearchCV(
    b_MLP,
    param_grid=param_dict_MLP,
    cv=10,
    verbose=1,
    n_jobs=1

)
MLP_grid.fit(X_train,Y_train)
print(MLP_grid.best_params_)

top_MLP = MLPClassifier(hidden_layer_sizes=(30,50),activation='tanh',solver ='adam')
top_MLP.fit(X_train,Y_train)

predicted_top_MLP = top_MLP.predict(X_test)

#step 7
#Step 7
with open(output_performance_fullpath, 'w') as output_performance_file:
        
        #7. (a)Header
        output_performance_file.writelines([
            "(a)\n",
            "*********************************************\n",
            "**** Gaussian NB, try 1 ****\n",
            "*********************************************\n"])
        #7. (b) Confusion Matrix
        cm_GNB = confusion_matrix(Y_test, predicted_NB)
        print(cm_GNB)
        output_performance_file.write("(b)\n")
        np.savetxt(output_performance_file, cm_GNB, fmt="%3d")
        
        #7. (c) Precision, Recall, F1-measure
        output_performance_file.write("(c)\n")
        report = classification_report(Y_test, predicted_NB,target_names=target_names)
        #classification_report's output has excess lines, so remove those.
        truncated_report = ''.join(report.splitlines(keepends=True)[:-4])
        output_performance_file.write(report)
        output_performance_file.write(truncated_report)
        
        #7. (d) accuracy, macro-average F1 and weighted-average F1
        # Seems like these are some of the values truncated out in 7.(c). Specifications suggest to obtain the metrics through these methods though.
        output_performance_file.writelines(["(d)\n",
                                            "Accuracy   : ", str(accuracy_score(Y_test, predicted_NB)), "\n",
                                            "Macro    F1: ", str(f1_score(Y_test, predicted_NB, average="macro")), "\n",
                                            "Weighted F1: ", str(f1_score(Y_test, predicted_NB, average="weighted")), "\n"])

         #7. (a)Header
        output_performance_file.writelines([
            "\n\n(a)\n",
            "*********************************************\n",
            "**** Decision Tree , try 1 ****\n",
            "*********************************************\n"])
        #7. (b) Confusion Matrix
        cm_dt = confusion_matrix(Y_test, predicted)
        print(cm_dt)
        output_performance_file.write("(b)\n")
        np.savetxt(output_performance_file, cm_dt, fmt="%3d")
        
        #7. (c) Precision, Recall, F1-measure
        output_performance_file.write("(c)\n")
        report = classification_report(Y_test, predicted,target_names=target_names)
        #classification_report's output has excess lines, so remove those.
        truncated_report = ''.join(report.splitlines(keepends=True)[:-4])
        output_performance_file.write(report)
        output_performance_file.write(truncated_report)
        
        #7. (d) accuracy, macro-average F1 and weighted-average F1
        # Seems like these are some of the values truncated out in 7.(c). Specifications suggest to obtain the metrics through these methods though.
        output_performance_file.writelines(["(d)\n",
                                            "Accuracy   : ", str(accuracy_score(Y_test, predicted)), "\n",
                                            "Macro    F1: ", str(f1_score(Y_test, predicted, average="macro")), "\n",
                                            "Weighted F1: ", str(f1_score(Y_test, predicted, average="weighted")), "\n"])


         #7. (a)Header
        output_performance_file.writelines([
            "\n\n(a)\n",
            "*********************************************\n",
            "**** Top-dt , try 1 ****\n",
            "*********************************************\n"])
        #7. (b) Confusion Matrix
        cm_topDt = confusion_matrix(Y_test, predicted_topDt)
        print(cm_topDt )
        output_performance_file.write("(b)\n")
        np.savetxt(output_performance_file, cm_topDt , fmt="%3d")
        
        #7. (c) Precision, Recall, F1-measure
        output_performance_file.write("(c)\n")
        report = classification_report(Y_test, predicted_topDt,target_names=target_names)
        #classification_report's output has excess lines, so remove those.
        truncated_report = ''.join(report.splitlines(keepends=True)[:-4])
        output_performance_file.write(report)
        output_performance_file.write(truncated_report)
        
        #7. (d) accuracy, macro-average F1 and weighted-average F1
        # Seems like these are some of the values truncated out in 7.(c). Specifications suggest to obtain the metrics through these methods though.
        output_performance_file.writelines(["(d)\n",
                                            "Accuracy   : ", str(accuracy_score(Y_test, predicted_topDt)), "\n",
                                            "Macro    F1: ", str(f1_score(Y_test, predicted_topDt, average="macro")), "\n",
                                            "Weighted F1: ", str(f1_score(Y_test, predicted_topDt, average="weighted")), "\n"])

        #7. (a)Header
        output_performance_file.writelines([
            "\n\n(a)\n",
            "*********************************************\n",
            "**** Perceptron , try 1 ****\n",
            "*********************************************\n"])
        #7. (b) Confusion Matrix
        cm_per = confusion_matrix(Y_test, predicted_pt)
        print(cm_per )
        output_performance_file.write("(b)\n")
        np.savetxt(output_performance_file, cm_per , fmt="%3d")
        
        #7. (c) Precision, Recall, F1-measure
        output_performance_file.write("(c)\n")
        report = classification_report(Y_test, predicted_pt,target_names=target_names)
        #classification_report's output has excess lines, so remove those.
        truncated_report = ''.join(report.splitlines(keepends=True)[:-4])
        output_performance_file.write(report)
        output_performance_file.write(truncated_report)
        
        #7. (d) accuracy, macro-average F1 and weighted-average F1
        # Seems like these are some of the values truncated out in 7.(c). Specifications suggest to obtain the metrics through these methods though.
        output_performance_file.writelines(["(d)\n",
                                            "Accuracy   : ", str(accuracy_score(Y_test, predicted_pt)), "\n",
                                            "Macro    F1: ", str(f1_score(Y_test, predicted_pt, average="macro")), "\n",
                                            "Weighted F1: ", str(f1_score(Y_test, predicted_pt, average="weighted")), "\n"])

        
        #7. (a)Header
        output_performance_file.writelines([
            "\n\n(a)\n",
            "*********************************************\n",
            "**** Base_MLP , try 1 ****\n",
            "*********************************************\n"])
        #7. (b) Confusion Matrix
        cm_b_MLP = confusion_matrix(Y_test, predicted_MLP)
        print(cm_b_MLP)
        output_performance_file.write("(b)\n")
        np.savetxt(output_performance_file, cm_b_MLP , fmt="%3d")
        
        #7. (c) Precision, Recall, F1-measure
        output_performance_file.write("(c)\n")
        report = classification_report(Y_test, predicted_MLP,target_names=target_names)
        #classification_report's output has excess lines, so remove those.
        truncated_report = ''.join(report.splitlines(keepends=True)[:-4])
        output_performance_file.write(report)
        output_performance_file.write(truncated_report)
        
        #7. (d) accuracy, macro-average F1 and weighted-average F1
        # Seems like these are some of the values truncated out in 7.(c). Specifications suggest to obtain the metrics through these methods though.
        output_performance_file.writelines(["(d)\n",
                                            "Accuracy   : ", str(accuracy_score(Y_test, predicted_MLP)), "\n",
                                            "Macro    F1: ", str(f1_score(Y_test, predicted_MLP, average="macro")), "\n",
                                            "Weighted F1: ", str(f1_score(Y_test, predicted_MLP, average="weighted")), "\n"])


          #7. (a)Header
        output_performance_file.writelines([
            "\n\n(a)\n",
            "*********************************************\n",
            "**** Top_MLP , try 1 ****\n",
            "*********************************************\n"])
        #7. (b) Confusion Matrix
        cm_top_MLP = confusion_matrix(Y_test, predicted_top_MLP)
        print(cm_top_MLP)
        output_performance_file.write("(b)\n")
        np.savetxt(output_performance_file, cm_top_MLP , fmt="%3d")
        
        #7. (c) Precision, Recall, F1-measure
        output_performance_file.write("(c)\n")
        report = classification_report(Y_test, predicted_top_MLP,target_names=target_names)
        #classification_report's output has excess lines, so remove those.
        truncated_report = ''.join(report.splitlines(keepends=True)[:-4])
        output_performance_file.write(report)
        output_performance_file.write(truncated_report)
        
        #7. (d) accuracy, macro-average F1 and weighted-average F1
        # Seems like these are some of the values truncated out in 7.(c). Specifications suggest to obtain the metrics through these methods though.
        output_performance_file.writelines(["(d)\n",
                                            "Accuracy   : ", str(accuracy_score(Y_test, predicted_top_MLP)), "\n",
                                            "Macro    F1: ", str(f1_score(Y_test, predicted_top_MLP, average="macro")), "\n",
                                            "Weighted F1: ", str(f1_score(Y_test, predicted_top_MLP, average="weighted")), "\n"])
#Step 8
for i in range(1,11):
    
    #step 6 


    #a)Gaussian Naive bayes classifier
    NB_8=GaussianNB()
    NB_8.fit(X_train, Y_train)
    predicted_NB_8 = NB_8.predict(X_test)

    



    #B)Decision tree
    drugTree_8 = DecisionTreeClassifier(criterion="entropy", max_depth=4)

    drugTree_8.fit(X_train, Y_train)
    predicted_8 = drugTree_8.predict(X_test)

    



    #c) Top-DT

    param_dict={
        "criterion":['gini','entropy'],
        "max_depth":range(5,6),
        "min_samples_split":range(2,4)
    }

    dt_grid = GridSearchCV(
        drugTree ,
        param_grid=param_dict,
        cv=10,
        verbose=1,
        n_jobs=1

    )
    dt_grid.fit(X_train,Y_train)
    print(dt_grid.best_params_)

    drugTree_top_8 = DecisionTreeClassifier(criterion="gini", max_depth=5, min_samples_split=2)

    drugTree_top_8.fit(X_train, Y_train)
    predicted_topDt_8 = drugTree_top_8.predict(X_test)





    #d) Perceptron
    pt_8 = Perceptron()
    pt_8.fit(X_train,Y_train)

    predicted_pt_8 = pt_8.predict(X_test)

    
    #e)Base-MLP
    b_MLP_8 = MLPClassifier(hidden_layer_sizes=(100,),activation='logistic',solver ='sgd',max_iter=200 )
    b_MLP_8.fit(X_train,Y_train)

    predicted_MLP_8 = b_MLP_8.predict(X_test)

    

    #f)top-MLP

    param_dict_MLP={
        "activation":['logistic','tanh','relu','identity'],
        "hidden_layer_sizes":[(30,50),(10,10,10)],
        "solver":['sgd','adam']
    }

    MLP_grid = GridSearchCV(
        b_MLP,
        param_grid=param_dict_MLP,
        cv=10,
        verbose=1,
        n_jobs=1

    )
    MLP_grid.fit(X_train,Y_train)
    print(MLP_grid.best_params_)

    top_MLP_8 = MLPClassifier(hidden_layer_sizes=(30,50),activation='tanh',solver ='adam')
    top_MLP_8.fit(X_train,Y_train)

    predicted_top_MLP_8 = top_MLP_8.predict(X_test)

    #step 7
    with open(output_performance_fullpath, 'a') as output_performance_file:
    
            
     #7. (a)Header
        output_performance_file.writelines([
                "\n\n\n\n\nStep 8\n\n(a)\n",
                "*********************************************\n",
                "**** Gaussian NB, try ",str(i),"****\n" ,
                "*********************************************\n"])
            
            
            #8 accuracy, macro-average F1 and weighted-average F1
            # Seems like these are some of the values truncated out in 7.(c). Specifications suggest to obtain the metrics through these methods though.
        output_performance_file.writelines(["(d)\n",
                                                "Accuracy   : ", str(accuracy_score(Y_test, predicted_NB_8)), "\n",
                                                "Macro    F1: ", str(f1_score(Y_test, predicted_NB_8, average="macro")), "\n",
                                                "Weighted F1: ", str(f1_score(Y_test, predicted_NB_8, average="weighted")), "\n"])

            #7. (a)Header
        output_performance_file.writelines([
                "\n\n(a)\n",
                "*********************************************\n",
                "**** Decision Tree try ",str(i),"****\n",
                "*********************************************\n"])
            
            
            #7. (d) accuracy, macro-average F1 and weighted-average F1
            # Seems like these are some of the values truncated out in 7.(c). Specifications suggest to obtain the metrics through these methods though.
        output_performance_file.writelines(["(d)\n",
                                                "Accuracy   : ", str(accuracy_score(Y_test, predicted_8)), "\n",
                                                "Macro    F1: ", str(f1_score(Y_test, predicted_8, average="macro")), "\n",
                                                "Weighted F1: ", str(f1_score(Y_test, predicted_8, average="weighted")), "\n"])


            #7. (a)Header
        output_performance_file.writelines([
                "\n\n(a)\n",
                "*********************************************\n",
                "**** Top-dt try ",str(i),"****\n",
                "*********************************************\n"])
           
            
            #7. (d) accuracy, macro-average F1 and weighted-average F1
            # Seems like these are some of the values truncated out in 7.(c). Specifications suggest to obtain the metrics through these methods though.
        output_performance_file.writelines(["(d)\n",
                                                "Accuracy   : ", str(accuracy_score(Y_test, predicted_topDt_8)), "\n",
                                                "Macro    F1: ", str(f1_score(Y_test, predicted_topDt_8, average="macro")), "\n",
                                                "Weighted F1: ", str(f1_score(Y_test, predicted_topDt_8, average="weighted")), "\n"])

            #7. (a)Header
        output_performance_file.writelines([
                "\n\n(a)\n",
                "*********************************************\n",
                "**** Perceptron try ",str(i),"****\n",
                "*********************************************\n"])
            
            #7. (d) accuracy, macro-average F1 and weighted-average F1
            # Seems like these are some of the values truncated out in 7.(c). Specifications suggest to obtain the metrics through these methods though.
        output_performance_file.writelines(["(d)\n",
                                                "Accuracy   : ", str(accuracy_score(Y_test, predicted_pt_8)), "\n",
                                                "Macro    F1: ", str(f1_score(Y_test, predicted_pt_8, average="macro")), "\n",
                                                "Weighted F1: ", str(f1_score(Y_test, predicted_pt_8, average="weighted")), "\n"])

            
            #7. (a)Header
        output_performance_file.writelines([
                "\n\n(a)\n",
                "*********************************************\n",
                "**** Base_MLP try ",str(i),"****\n",
                "*********************************************\n"])
           
            
            #7. (d) accuracy, macro-average F1 and weighted-average F1
            # Seems like these are some of the values truncated out in 7.(c). Specifications suggest to obtain the metrics through these methods though.
        output_performance_file.writelines(["(d)\n",
                                                "Accuracy   : ", str(accuracy_score(Y_test, predicted_MLP_8)), "\n",
                                                "Macro    F1: ", str(f1_score(Y_test, predicted_MLP_8, average="macro")), "\n",
                                                "Weighted F1: ", str(f1_score(Y_test, predicted_MLP_8, average="weighted")), "\n"])


            #7. (a)Header
        output_performance_file.writelines([
                "\n\n(a)\n",
                "*********************************************\n",
                "**** Top_MLP try ",str(i),"****\n",
                "*********************************************\n"])
            
            
            #7. (d) accuracy, macro-average F1 and weighted-average F1
            # Seems like these are some of the values truncated out in 7.(c). Specifications suggest to obtain the metrics through these methods though.
        output_performance_file.writelines(["(d)\n",
                                                "Accuracy   : ", str(accuracy_score(Y_test, predicted_top_MLP_8)), "\n",
                                                "Macro    F1: ", str(f1_score(Y_test, predicted_top_MLP_8, average="macro")), "\n",
                                                "Weighted F1: ", str(f1_score(Y_test, predicted_top_MLP_8, average="weighted")), "\n"])


