#Suthan Sinnathurai
#40086318
#Task 2

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from collections import Counter
from sklearn.tree import DecisionTreeClassifier,plot_tree
from sklearn.metrics import accuracy_score,f1_score,precision_score,recall_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


drug_data = pd.read_csv('drug200.csv',delimiter=',')

#Step 3
counter =Counter(drug_data['Drug'])
drug_name = counter.keys()
drug_name_counts = counter.values()

indexes = np.arange(len(drug_name))
width = 0.7

plt.bar(indexes,drug_name_counts,width)
plt.xticks(indexes +width*0.5,drug_name)
plt.xlabel("Drug_type")
plt.ylabel("Count")
plt.title("Bar graph of the number of instances in each class of Drug_type")
plt.show()
plt.savefig("drug-distribution.pdf")


#step 4


#ordinal values 
#drug_data.Sex=pd.Categorical(drug_data.Sex,['F','M'],ordered = False)
drug_data.BP=pd.Categorical(drug_data.BP,['LOW','NORMAL','HIGH'],ordered = True)
drug_data.Cholesterol=pd.Categorical(drug_data.Cholesterol,['NORMAL','HIGH'],ordered = True)
#drug_data.Drug=pd.Categorical(drug_data.Drug,['drugY','drugC','drugX','drugA','drugB'],ordered = False)

drug_data.BP = drug_data.BP.cat.codes
drug_data.Cholesterol = drug_data.Cholesterol.cat.codes

drug_data =pd.get_dummies(drug_data)
print(drug_data)

#step 5
#not working yet
y=drug_data['Drug']
x=drug_data[['Age','Sex','BP','Cholesterol','Na_to_K']].values

X_train,X_test,Y_train,Y_test= train_test_split(x,y,random_state=1)
print(X_train.shape)
print(X_test.shape)
print(Y_train.shape)
print(Y_test.shape)




#Step 6

#b)Decision tree
dt = DecisionTreeClassifier(
    criterion ='entropy',
    max_depth =10
)

dt.fit(X_train,Y_train)

plot_tree(dt)

