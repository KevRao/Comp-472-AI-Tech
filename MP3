import pandas as pd
import random
import csv
import gensim.downloader as api
from gensim.models import Word2Vec
from gensim.models import KeyedVectors

##### TASK 1 & TASK 2 ######
def modelPerformance(modelName):
    #Load model
    model = api.load(modelName)
    #Load the question-word file
    corpus = pd.read_csv("synonyms.csv")
    #Create the details file of the model response
    modelFile = open(f"{modelName}-details.csv", 'w')
    modelWriter = csv.writer(modelFile)
    modelWriter.writerow(['question', 'answer', 'modelAnswer', 'label'])
    print('question', 'answer', 'modelAnswer', 'label')
    #Create analysis file
    analysisFile = open('analysis.csv', 'w')
    analysisWriter = csv.writer(analysisFile)
    analysisWriter.writerow(['ModelName', 'VocabularySize', 'correctLabels', 'NumOfNoGuess', 'ModelAccuracy'])
    print('ModelName', 'VocabularySize', 'correctLabels', 'NumOfNoGuess', 'ModelAccuracy')
    guessCount, correctCount, wrongCount = 0, 0, 0
    for question,answer,*options in corpus.values:
        quest = model.has_index_for(question)
        print(quest)
        list_Option = [ opt for opt in options if model.has_index_for(opt)]
        modelAnswer = model.most_similar_to_given(quest, list_Option)
        print(modelAnswer)
        if(quest and list_Option):
            label = 'correct' if modelAnswer == answer else 'wrong' 
            if label == 'correct':
                correctCount += 1
            elif label == 'wrong':
                wrongCount += 1
        else:
            modelAnswer = random.choice(list_Option)
            label = 'guess'
            guessCount += 1 
        print(label)      
        modelWriter.writerows([question, answer, modelAnswer, label]) 
        print(question, answer, modelAnswer, label)
    vocabularySize = len(model.index_to_key) 
    print(vocabularySize)     
    numGuess = 80 - guessCount
    print(numGuess)
    modelAccuracy = correctCount/numGuess
    print(modelAccuracy)

    analysisWriter.writerow([f"{modelName}", vocabularySize, correctCount, numGuess, modelAccuracy])    
    modelFile.close()
    print("done")
    analysisFile.close()
    print("done")

def main():
    modelList = ['word2vec-google-news-300', 'glove-twitter-200', 'glove-wiki-gigaword-200', 'glove-wiki-gigaword-50', 'glove-wiki-gigaword-300']
    for models in modelList:
        modelPerformance(models)

if __name__ == "__main__":
	main()        
