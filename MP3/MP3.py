import gensim.downloader as api
from gensim.models import KeyedVectors
from gensim.models import Word2Vec
import pandas as pd
import os
import random
import csv
import h5py


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


wv_12 = api.load('glove-wiki-gigaword-300')

output_fullname_12 = 'glove-wiki-gigaword-300.csv'
output_fullpath_12 = os.path.join(output_directory, output_fullname_12)


with open(output_fullpath_12, 'w',newline="") as f:
    writer = csv.writer(f)
    header =['question','answer','model_guess','label']
    writer.writerow(header)

    for question,answer,*choices in file.values:
        choices = [choice for choice in choices if wv_12.has_index_for(choice)]
        

        if(wv_12.has_index_for(question) and choices):
            model_guess = wv_12.most_similar_to_given(question,choices)
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


wv_gigaword_50 = api.load('glove-wiki-gigaword-50')

output_gigaword_50  = 'glove-wiki-gigaword-50.csv'
output_fullpath_gigaword_50  = os.path.join(output_directory, output_gigaword_50)


with open(output_fullpath_gigaword_50, 'w',newline="") as f:
    writer = csv.writer(f)
    header =['question','answer','model_guess','label']
    writer.writerow(header)

    for question,answer,*choices in file.values:
        choices = [choice for choice in choices if wv_gigaword_50.has_index_for(choice)]
        

        if(wv_gigaword_50.has_index_for(question) and choices):
            model_guess = wv_gigaword_50.most_similar_to_given(question,choices)
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

wv_gigaword_200 = api.load('glove-wiki-gigaword-200')

output_gigaword_200  = 'glove-wiki-gigaword-200.csv'
output_fullpath_gigaword_200  = os.path.join(output_directory, output_gigaword_200)


with open(output_fullpath_gigaword_200, 'w',newline="") as f:
    writer = csv.writer(f)
    header =['question','answer','model_guess','label']
    writer.writerow(header)

    for question,answer,*choices in file.values:
        choices = [choice for choice in choices if wv_gigaword_200.has_index_for(choice)]
        

        if(wv_gigaword_200.has_index_for(question) and choices):
            model_guess = wv_gigaword_200.most_similar_to_given(question,choices)
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

wv_twitter_200 = api.load('glove-twitter-200')

output_fullname_200 = 'glove-twitter-200.csv'
output_fullpath_200 = os.path.join(output_directory, output_fullname_200)


with open(output_fullpath_200, 'w',newline="") as f:
    writer = csv.writer(f)
    header =['question','answer','model_guess','label']
    writer.writerow(header)

    for question,answer,*options in file.values:
        choices = [choice for choice in options  if wv_twitter_200.has_index_for(choice)]
        
        

        if(wv_twitter_200.has_index_for(question) and choices):
            model_guess = wv_twitter_200.most_similar_to_given(question,choices)
            if(answer==model_guess):                
                label = 'correct'
                print(question,answer,model_guess,'correct')

            else:
                label = 'wrong'  
                print(question,answer,model_guess,'wrong')
        else:
            label = 'guess'
            model_guess = (random.choice(choices or options))
            print(question,answer,model_guess,'guess')
        data =[question,answer,model_guess,label]
        writer.writerow(data)


file_analysis_gigaword_50 = pd.read_csv("MP3\output\glove-wiki-gigaword-50.csv")
file_analysis_gigaword_200 = pd.read_csv("MP3\output\glove-wiki-gigaword-200.csv")
file_analysis_gigaword_300 = pd.read_csv("MP3\output\glove-wiki-gigaword-300.csv")
file_analysis_twitter_200  = pd.read_csv("MP3\output\glove-twitter-200.csv")

with open(output_analysispath,'a',newline="") as f:
    writer = csv.writer(f)
    
    #initializing the counter variables
    count_right =0
    count_guess =0

    for question,awnser,model_guess,label in file_analysis_gigaword_50.values:
        if(label=='correct'):
            count_right+=1
        if(label =='guess'):
            count_guess+=1

    c = count_right
    v = 80-count_guess
    accuracy = c/v

    data =['glove-wiki-gigaword-50',len(wv_gigaword_50.index_to_key),c,v,accuracy]
    writer.writerow(data)

    #initializing the counter variables
    count_right =0
    count_guess =0

    for question,awnser,model_guess,label in file_analysis_gigaword_200:
        if(label=='correct'):
            count_right+=1
        if(label =='guess'):
            count_guess+=1

    c = count_right
    v = 80-count_guess
    accuracy = c/v

    data =['glove-wiki-gigaword-200',len(wv_gigaword_200.index_to_key),c,v,accuracy]
    writer.writerow(data)

    #initializing the counter variables
    count_right =0
    count_guess =0

    for question,awnser,model_guess,label in file_analysis_gigaword_300:
        if(label=='correct'):
            count_right+=1
        if(label =='guess'):
            count_guess+=1

    c = count_right
    v = 80-count_guess
    accuracy = c/v

    data =['glove-wiki-gigaword-300',len(wv_12.index_to_key),c,v,accuracy]
    writer.writerow(data)

    #initializing the counter variables
    count_right =0
    count_guess =0

    for question,awnser,model_guess,label in file_analysis_twitter_200.values:
        if(label=='correct'):
            count_right+=1
        if(label =='guess'):
            count_guess+=1

    c = count_right
    v = 80-count_guess
    accuracy = c/v

    data =['glove-twitter-200',len(wv_twitter_200.index_to_key),c,v,accuracy]
    writer.writerow(data)

