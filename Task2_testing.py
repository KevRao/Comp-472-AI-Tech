#Suthan Sinnathurai
#40086318
#Task 2_testing

import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics
from sklearn import preprocessing
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import Perceptron
from sklearn.neural_network import MLPClassifier

datainput = pd.read_csv("drug200.csv", delimiter=",")

X = datainput[['Age', 'Sex', 'BP', 'Cholesterol', 'Na_to_K']].values

# Data Preprocessing


label_gender = preprocessing.LabelEncoder()
label_gender.fit(['F', 'M'])
X[:, 1] = label_gender.transform(X[:, 1])

label_BP = preprocessing.LabelEncoder()
label_BP.fit(['LOW', 'NORMAL', 'HIGH'])
X[:, 2] = label_BP.transform(X[:, 2])

label_Chol = preprocessing.LabelEncoder()
label_Chol.fit(['NORMAL', 'HIGH'])
X[:, 3] = label_Chol.transform(X[:, 3])

print(datainput)
y = datainput["Drug"]

# train_test_split
from sklearn.model_selection import train_test_split,GridSearchCV

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=3)

#step 6 

#a)Gaussian Naive bayes classifier
NB=GaussianNB()
NB.fit(X_train, y_train)
predicted_NB = NB.predict(X_test)

print(predicted_NB)

print("\nGaussian NB's Accuracy: ", metrics.accuracy_score(y_test, predicted_NB))

#B)Decision tree
drugTree = DecisionTreeClassifier(criterion="entropy", max_depth=4)

drugTree.fit(X_train, y_train)
predicted = drugTree.predict(X_test)

print(predicted)

print("\nDecisionTrees's Accuracy: ", metrics.accuracy_score(y_test, predicted))

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
dt_grid.fit(X_train,y_train)
print(dt_grid.best_params_)


#d) Perceptron
pt = Perceptron()
pt.fit(X_train,y_train)

predicted_pt = pt.predict(X_test)

print(predicted_pt)

print("\n Pereptron's Accuracy: ", metrics.accuracy_score(y_test, predicted_pt))


#e)Base-MLP
b_MLP = MLPClassifier(hidden_layer_sizes=(100,),activation='logistic',solver ='sgd',max_iter=200 )
b_MLP.fit(X_train,y_train)

predicted_MLP = b_MLP.predict(X_test)

print(predicted_MLP)

print("\n Base-MLP Pereptron's Accuracy: ", metrics.accuracy_score(y_test, predicted_MLP))

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
MLP_grid.fit(X_train,y_train)
print(MLP_grid.best_params_)
