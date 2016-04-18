import re
import csv
import sys
import string
import pickle
import os.path
import numpy as np
import time
start_time = time.time()

#initialization
uniqueWords={}
totalWordsInPosClass=[]
totalWordsInNegClass=[]
vocab=[]

def initialize(K):
    for k in xrange(K):
	vocab.append(0)
	totalWordsInPosClass.append(0)
	totalWordsInNegClass.append(0)

def initializeunigrams(K):
    a=[]
    for k in xrange(K):
	a.append([0,0])
    return a

# process the reviews
def processReview(review):

    #Convert to lower case
    review = review.lower()
    #Remove additional white spaces
    review = re.sub('[\s]+', ' ', review)
    #trim
    review = review.strip('\'"')
    return review
#end

#start replaceTwoOrMore
def replaceTwoOrMore(s):
    #look for 2 or more repetitions of character and replace with the character itself
    pattern = re.compile(r"(.)\1{1,}", re.DOTALL)
    return pattern.sub(r"\1\1", s)
#end

#start getStopWordList
def getStopWordList(stopWordListFileName):
    #read the stopwords file and build a list
    stopWords = []
    fp = open(stopWordListFileName, 'r')
    line = fp.readline()
    while line:
        word = line.strip()
        stopWords.append(word)
        line = fp.readline()
    fp.close()
    return stopWords
#end

#start getfeatureVector
def getFeatureVector(review,stopWords):
    featureVector = []
    #split review into words
    words = review.split()
    for w in words:
        #replace two or more with two occurrences
        w = replaceTwoOrMore(w)
        #strip punctuation
        #w = w.strip('\'"?,.')
        #check if the word starts with an alphabet
        val = re.search(r"^[a-zA-Z][a-zA-Z0-9]*$", w)
        #ignore if it is a stop word
        if(w in stopWords or val is None):
            continue
        else:
            featureVector.append(w.lower())
    return featureVector
#end

#creates stopwordslist
stopWords = getStopWordList('stopwords.txt')

#word extraction
def train(K):
	global uniqueWords
	global totalWordsInPosClass
	global totalWordsInNegClass
	global vocab
	fp = open("data.txt", "r")
	for i, row in enumerate(fp):
		line = row.split("\t")
		review=line[1]
        	clas=line[0]
        	review=processReview(review)
        	replace_punctuation = string.maketrans(string.punctuation, ' '*len(string.punctuation))
        	review=review.translate(replace_punctuation)
        	reviewf=getFeatureVector(review,stopWords)
		for k in xrange(K):
			if i%10!=k:	
        			if clas=='-':
        	    			totalWordsInNegClass[k]+=len(reviewf)
        			else:
        	    			totalWordsInPosClass[k]+=len(reviewf)
        			for word in reviewf:
        	    			if word in uniqueWords:
        	    				if clas=='-':
							uniqueWords[word][k][1]+=1
						else:
			 		       		uniqueWords[word][k][0]+=1
		    			else:
       		    				uniqueWords[word]=initializeunigrams(K)
		print "\r "+str(i)+" lines finished" 
	print "Finished Word Extraction"
	#constructing vocabulary having count greater than 2
	for word in uniqueWords:
		for k in xrange(K):
			if (uniqueWords[word][k][0]+uniqueWords[word][k][1])>=1:
				vocab[k]+=1
		for k in xrange(K):
			uniqueWords[word][k][0]=np.log10((uniqueWords[word][k][0]+1.0)/(totalWordsInPosClass[k]+vocab[k]))
			uniqueWords[word][k][1]=np.log10((uniqueWords[word][k][1]+1.0)/(totalWordsInNegClass[k]+vocab[k]))
	print len(uniqueWords)
#end

#saving into pickle
def savepickle():
	global uniqueWords
	global totalWordsInPosClass
	global totalWordsInNegClass
	global vocab
	print "Saving into Pickle"
	words=open('words.pickle','wb')
	negwords=open('negwords.pickle','wb')
	poswords=open('poswords.pickle','wb')
	vocabulary=open('vocab.pickle','wb')
	pickle.dump(uniqueWords,words,pickle.HIGHEST_PROTOCOL)
	pickle.dump(totalWordsInPosClass,poswords,pickle.HIGHEST_PROTOCOL)
	pickle.dump(totalWordsInNegClass,negwords,pickle.HIGHEST_PROTOCOL)
	pickle.dump(vocab,vocabulary,pickle.HIGHEST_PROTOCOL)
	words.close()
	negwords.close()
	poswords.close()
	vocabulary.close()
#end

#loading from pickle
def loadpickle():
	global uniqueWords
	global totalWordsInPosClass
	global totalWordsInNegClass
	global vocab
	f1=open('words.pickle','rb')
	f2=open('negwords.pickle','rb')
	f3=open('poswords.pickle','rb')
	f4=open('vocab.pickle','rb')
	uniqueWords=pickle.load(f1)		
	totalWordsInPosClass=pickle.load(f2)
	totalWordsInPosClass=pickle.load(f3)
	vocab=pickle.load(f4)
	f1.close()
	f2.close()
	f3.close()
	f4.close()
#end

def accuracyFinder(test,k):
	global uniqueWords
	global totalWordsInPosClass
	global totalWordsInNegClass
	global vocab
	givenNegreviews=0
	givenPosreviews=0
	obtainedNegreviews=0
	obtainedPosreviews=0
	wrongCount=0
	correctCount=0
	correctNeg=0
	correctPos=0
	posSmoothing=np.log10(1.0/(totalWordsInPosClass[k]+vocab[k]))
	negSmoothing=np.log10(1.0/(totalWordsInNegClass[k]+vocab[k]))
    	for i in xrange(len(test)):
        	review=test[i][0]
		clas=test[i][1]
        	review=processReview(review)
        	replace_punctuation = string.maketrans(string.punctuation, ' '*len(string.punctuation))
        	review=review.translate(replace_punctuation)
        	reviewf=getFeatureVector(review,stopWords)
		pos_prob=0
		neg_prob=0	
        	for word in reviewf:
            		if word in uniqueWords:
				neg_prob+=uniqueWords[word][k][1]
				pos_prob+=uniqueWords[word][k][0]
	    		else:
       	    			neg_prob+=negSmoothing
				pos_prob+=posSmoothing
		#evaluating for accuracy
        	if clas=='-':
        		givenNegreviews+=1
        	else:
        		givenPosreviews+=1
        	if(pos_prob>neg_prob):
         		obtainedPosreviews+=1
           		if clas=='-':
                		wrongCount+=1
            		else:
                		correctCount+=1
                		correctPos+=1
        	else:
            		obtainedNegreviews+=1
            		if clas=='-':
                		correctCount+=1
                		correctNeg+=1
            		else:
                		wrongCount+=1			
	print "=============================================================="
	print('GivenNegreviews'+": "+str(givenNegreviews))
	print('ObtainedNegreviews'+": "+str(obtainedNegreviews))
	print('GivenPosreviews'+": "+str(givenPosreviews))
	print('ObtainedPosreviews'+": "+str(obtainedPosreviews))
	print('CorrectCount is :'+str(correctCount))
	print('WrongCount is :'+str(wrongCount))
	print('CorrectPoCount is :'+str(correctPos))
	print('CorrectNegCount is :'+str(correctNeg))
	print('Accuracy is : {0}%'.format(float(correctCount)*100/(correctCount+wrongCount)))
	print "=============================================================="
	return float(correctCount)*100/(correctCount+wrongCount)
#end

#converting data into list of reviews and tags
train_samples=[]
l = open("data.txt", "r").readlines()
for row in l:
	line = row.split("\t")
	review=line[1]
        clas=line[0]	
	if (clas=='+'):
		train_samples.append([review,"+"])
	else:
		train_samples.append([review,"-"])

#k-fold cross validation divsion of data
def k_fold_cross_validation(X, K, randomise = False):
	if randomise: from random import shuffle; X=list(X); shuffle(X)
	avgacc=0
	fold=1
	for k in xrange(K):
		print("fold"+str(fold)+" started")
		test = [x for i, x in enumerate(X) if i % K == k]	
		avgacc = float(avgacc + accuracyFinder(test,k))
		fold+=1
	return float(avgacc/K)			
#end

#main
fold=10
initialize(fold)
train(fold)
savepickle()
loadpickle()
print('Avverage accuracy is : {0}%'.format(k_fold_cross_validation(train_samples, fold)))
print "=============================================================="
print("--- %s seconds ---" % (time.time() - start_time))
