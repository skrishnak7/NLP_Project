import re
import csv
import sys
import string
import pickle
import os.path
import bisect
import numpy as np
import time
start_time = time.time()

#initialization
uniqueWords={}
uniqueBigrams={}
vocab=[]
totalWordsInPosClass=[]
totalWordsInNegClass=[]

def initialize(K):
    for k in xrange(K):
	vocab.append(0)	
	totalWordsInPosClass.append(0)
	totalWordsInNegClass.append(0)

def initializeNgrams(K):
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
    review = re.sub('[0-9]th', '', review)
    review = re.sub('[~`^=!@#$,\.\)\(\:\;?\-\+%&*\/_\{\}\[\]<>\"]', ' ', review)
    review = string.replace(review,'\'',' ')
    review =string.replace(review,'/',' ')
    review =string.replace(review,'@',' ')
    review = review.strip()
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
	global uniqueBigrams
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
		reviewf=['<s>']+reviewf+['<e>']
		for k in xrange(K):
			if i%K!=k:	
				if clas=='-':
        	    			totalWordsInNegClass[k]+=len(reviewf)
        			else:
        	    			totalWordsInPosClass[k]+=len(reviewf)
        			for n in range(0,len(reviewf)):
					word=reviewf[n]
					if(n<len(reviewf)-1):
						bigram=reviewf[n]+" "+reviewf[n+1]
        	    			if word in uniqueWords:
        	    				if clas=='-':
							uniqueWords[word][k][1]+=1
						else:
			 		       		uniqueWords[word][k][0]+=1
		    			else:
       		    				uniqueWords[word]=initializeNgrams(K)
					if bigram in uniqueBigrams:
        	    				if clas=='-':
							uniqueBigrams[bigram][k][1]+=1
						else:
							uniqueBigrams[bigram][k][0]+=1	
		    			else:
						uniqueBigrams[bigram]=initializeNgrams(K)
		print "\r "+str(i)+" lines finished" 

	print "Finished Word Extraction"
	#removing bigrams having count less than 3
	delList=[]
	for bigram in uniqueBigrams:
		count=0
		for k in xrange(K):
			if (uniqueBigrams[bigram][k][0]+uniqueBigrams[bigram][k][1]<3):
				count+=1
		if(count==K):
			delList.append(bigram)
	for i in delList:
		del uniqueBigrams[i]
	print len(uniqueBigrams)	
	
	#constructing vocabulary having count greater than 2
	for word in uniqueWords:
		for k in xrange(K):
			if (uniqueWords[word][k][0]+uniqueWords[word][k][1])>2:
				vocab[k]+=1
	
	#calculating bigram probabilities
	for bigram in uniqueBigrams:
		words=bigram.split(" ")
		for k in xrange(K):
			uniqueBigrams[bigram][k][0]=np.log10((uniqueBigrams[bigram][k][0]+1.0)/((uniqueWords[words[0]][k][0]-uniqueBigrams[bigram][k][0])+vocab[k]))
			uniqueBigrams[bigram][k][1]=np.log10((uniqueBigrams[bigram][k][1]+1.0)/((uniqueWords[words[0]][k][1]-uniqueBigrams[bigram][k][1])+vocab[k]))
#end

#saving into pickle
def savepickle():
	global uniqueWords
	global uniqueBigrams
	global totalWordsInPosClass
	global totalWordsInNegClass
	global vocab
	print "Saving into Pickle"
	words=open('Unigrams.pickle','wb')
	bigrams=open('Bigrams.pickle','wb')	
	negwords=open('negativewords.pickle','wb')
	poswords=open('positivewords.pickle','wb')	
	vocabulary=open('vocabulary.pickle','wb')
	pickle.dump(uniqueWords,words,pickle.HIGHEST_PROTOCOL)	
	pickle.dump(uniqueBigrams,bigrams,pickle.HIGHEST_PROTOCOL)
	pickle.dump(totalWordsInPosClass,poswords,pickle.HIGHEST_PROTOCOL)
	pickle.dump(totalWordsInNegClass,negwords,pickle.HIGHEST_PROTOCOL)
	pickle.dump(vocab,vocabulary,pickle.HIGHEST_PROTOCOL)
	words.close()
	bigrams.close()	
	negwords.close()
	poswords.close()
	vocabulary.close()
#end

#loading from pickle
def loadpickle():
	global uniqueWords
	global uniqueBigrams
	global totalWordsInPosClass
	global totalWordsInNegClass
	global vocab
	f1=open('Unigrams.pickle','rb')
	f2=open('Bigrams.pickle','rb')
	f3=open('negativewords.pickle','rb')
	f4=open('positivewords.pickle','rb')
	f5=open('vocabulary.pickle','rb')
	uniqueWords=pickle.load(f1)		
	uniqueBigrams=pickle.load(f2)
	totalWordsInPosClass=pickle.load(f4)
	totalWordsInNegClass=pickle.load(f3)
	vocab=pickle.load(f5)
	f1.close()
	f2.close()
	f3.close()
	f4.close()
	f5.close()
#end

def accuracyFinder(test,k):
	global uniqueWords
	global uniqueBigrams
	global totalWordsInPosClass
	global totalWordsInNegClass
	global vocab
	print "Evaluating accuracy"
	givenNegreviews=0
	givenPosreviews=0
	obtainedNegreviews=0
	obtainedPosreviews=0
	wrongCount=0
	correctCount=0
	correctNeg=0
	correctPos=0
    	for i in xrange(len(test)):
        	review=test[i][0]
		clas=test[i][1]
        	review=processReview(review)
        	replace_punctuation = string.maketrans(string.punctuation, ' '*len(string.punctuation))
        	review=review.translate(replace_punctuation)
        	reviewf=getFeatureVector(review,stopWords)
		reviewf=['<s>']+reviewf+['<e>']
		pos_prob=0
		neg_prob=0	
        	for n in range(1,len(reviewf)):
			bigram =reviewf[n-1]+" "+reviewf[n]
			posSmoothing=np.log10((uniqueWords[reviewf[n]][k][0]+1.0)/(totalWordsInPosClass[k]+vocab[k]))
			negSmoothing=np.log10((uniqueWords[reviewf[n]][k][1]+1.0)/(totalWordsInNegClass[k]+vocab[k]))
            		if bigram in uniqueBigrams:
				neg_prob+=uniqueBigrams[bigram][k][1]
				pos_prob+=uniqueBigrams[bigram][k][0]
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
