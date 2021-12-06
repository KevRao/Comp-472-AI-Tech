import pandas as pd
import random
import csv
import gensim.downloader as api
from gensim.models import Word2Vec
from gensim.models import KeyedVectors
import os

##### TASK 1 & TASK 2 ######
def modelPerformance(modelName):
    #Load model
    model = api.load(modelName)
    #Load the question-word file
    corpus = pd.read_csv("MP3_lydia\synonyms.csv")
    #Create the details file of the model response
    local_directory = os.path.dirname(__file__)
    output_directory = os.path.join(local_directory, 'output')
    output_analysis = f"{modelName}-details.csv"
    output_analysispath = os.path.join(output_directory, output_analysis)
    with open(output_analysispath, 'w',newline="") as modelFile:
        modelWriter = csv.writer(modelFile)
        modelWriter.writerow(['question', 'answer', 'modelAnswer', 'label'])
        print('question', 'answer', 'modelAnswer', 'label')
    #Create analysis file
    output_analysis_2 = "analysis.csv"
    output_analysispath_2 = os.path.join(output_directory, output_analysis_2)
    
    print('ModelName', 'VocabularySize', 'correctLabels', 'NumOfNoGuess', 'ModelAccuracy')
    guessCount, correctCount, wrongCount = 0, 0, 0
    for question,answer,*options in corpus.values:
        
        
        list_Option = [ opt for opt in options if model.has_index_for(opt)]
        
        
        if(model.has_index_for(question) and list_Option):
            modelAnswer = model.most_similar_to_given(question, list_Option)
            print(modelAnswer)
            label = 'correct' if modelAnswer == answer else 'wrong' 
            if label == 'correct':
                correctCount += 1
            elif label == 'wrong':
                wrongCount += 1
        else:
            modelAnswer = random.choice(list_Option or options)
            label = 'guess'
            guessCount += 1 
        print(label)  
        with open(output_analysispath, 'a',newline="") as modelFile:   
            modelWriter = csv.writer(modelFile) 
            data =[question, answer, modelAnswer, label]
            modelWriter.writerows(data)
            print(question, answer, modelAnswer, label)
            
    with open(output_analysispath_2, 'a',newline="") as analysisFile:
        vocabularySize = len(model.index_to_key) 
        print(vocabularySize)     
        numGuess = 80 - guessCount
        print(numGuess)
        modelAccuracy = correctCount/numGuess
        print(modelAccuracy)
        analysisWriter = csv.writer(analysisFile)
        analysisWriter.writerow([f"{modelName}", vocabularySize, correctCount, numGuess, modelAccuracy])    
    
    print("done")
    

def main():
    modelList = ['word2vec-google-news-300', 'glove-twitter-200', 'glove-wiki-gigaword-200', 'glove-wiki-gigaword-50', 'glove-wiki-gigaword-300']
    local_directory = os.path.dirname(__file__)
    output_directory = os.path.join(local_directory, 'output')
    output_analysis_2 = "analysis.csv"
    output_analysispath_2 = os.path.join(output_directory, output_analysis_2)
    with open(output_analysispath_2, 'w',newline="") as analysisFile:
        analysisWriter = csv.writer(analysisFile)
        analysisWriter.writerow(['ModelName', 'VocabularySize', 'correctLabels', 'NumOfNoGuess', 'ModelAccuracy'])
    for models in modelList:
        modelPerformance(models)

if __name__ == "__main__":
	main()      