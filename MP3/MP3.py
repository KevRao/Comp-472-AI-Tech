import gensim.downloader as api
from gensim.models import KeyedVectors
from gensim.models import Word2Vec
import pandas as pd
import os
import random
import csv

#task 1 part 1

model = api.load('word2vec-google-news-300')

file = pd.read_csv("MP3\synonyms.csv")

local_directory = os.path.dirname(__file__)
output_directory = os.path.join(local_directory, 'output')

output_fullname = 'word2vec-google-news-300-details.csv'
output_fullpath = os.path.join(output_directory, output_fullname)


with open(output_fullpath, 'w',newline="") as f:
    writer = csv.writer(f)
    header =['question','answer','model_guess','label']
    writer.writerow(header)


    for question,answer,*choices in file.values:
        choices = [choice for choice in choices if model.has_index_for(choice)]
    

        if(model.has_index_for(question) and choices):
            model_guess = model.most_similar_to_given(question,choices)
            if(answer==model_guess):
                label = 'correct'
                

                print(question,answer,model_guess,'correct')
            else:
                label = 'wrong'
                
                print(question,answer,model_guess,'wrong')
        else:
            label = 'guess'
            model_guess = (random.choice(choices))
            
            
            
            print(question,answer,model_guess,'guess')
        data =[question,answer,model_guess,label]
        writer.writerow(data)


#task 1 part 2
#reading the output file produced in task 1 part 1
file_analysis = pd.read_csv("MP3\output\word2vec-google-news-300-details.csv")

output_analysis = 'analysis.csv'
output_analysispath = os.path.join(output_directory, output_analysis)

with open(output_analysispath, 'w',newline="") as f:
    writer = csv.writer(f)
    header =['model name','size of vocab','number of correct labels','number of answer without guessing','accuracy']
    writer.writerow(header)

    #initializing the counter variables
    count_right =0
    count_guess =0

    for question,awnser,model_guess,label in file_analysis.values:
        if(label=='correct'):
            count_right+=1
        if(label =='guess'):
            count_guess+=1

    c = count_right
    v = 80-count_guess
    accuracy = c/v

    data =['word2vec-google-news-300',len(model.index_to_key),c,v,accuracy]
    writer.writerow(data)


    

