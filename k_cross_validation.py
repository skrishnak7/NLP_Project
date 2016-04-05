import csv
import re
import sys
import string
import numpy as np
import nltk
import pickle
from sklearn import cross_validation

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

#training 9/10 of data for 10-fold validation
def train(training):
	posWords=[]
	negWords=[]
	uniqueWords={}
	posreviewCount=0
	negreviewCount=0
	for row in training:
		review=row[0]
        	clas=row[1]
        	replace_punctuation = string.maketrans(string.punctuation, ' '*len(string.punctuation))
		reviewf=processReview(review)
        	reviewf=reviewf.translate(replace_punctuation)
        	reviewf=getFeatureVector(review,stopWords)
		if clas=="Neg":
        	    negreviewCount+=1
        	else:
        	    posreviewCount+=1
        	for word in reviewf:
		    if word not in uniqueWords:
			uniqueWords[word]=1
		    else:
			uniqueWords[word]+=1
		    if clas=="Neg":
			    	negWords.append(word)
        	    else:
				posWords.append(word) 
	#removing words that have occurences less than 2
	for word,count in uniqueWords.items():
		if count<2:
			del uniqueWords[word]
	sys.stdout.write("\r total+ve : {0},total-ve : {1},total+veWords: {2},totalUnique+ve: {3},total-veWords: {4},totalUnique-ve: {5},totalUnique: {6}\n".format(posreviewCount,negreviewCount,len(posWords),len(set(posWords)),len(negWords),len(set(negWords)),len(uniqueWords)))
        sys.stdout.flush()
	#dictionaries having word and their positive and negative probabilities
	pos_prob_dict={}
	neg_prob_dict={}
	for word in uniqueWords:	
		pos_prob_dict[word]=np.log10((posWords.count(word)+1.0)/(len(posWords)+len(set(posWords))))
		neg_prob_dict[word]=np.log10((negWords.count(word)+1.0)/(len(negWords)+len(set(negWords))))
	#saving into pickle 
	pos=open('pos.pickle','wb')
	neg=open('neg.pickle','wb')
	pickle.dump(pos_prob_dict,pos,pickle.HIGHEST_PROTOCOL)
	pickle.dump(neg_prob_dict,neg,pickle.HIGHEST_PROTOCOL)
	pos.close()
	neg.close()
	posSmoothing=np.log10(1.0/(len(posWords)+len(set(posWords))))
	negSmoothing=np.log10(1.0/(len(negWords)+len(set(negWords))))	
	return posSmoothing,negSmoothing
#end

#testing 1/10 of data and evaluating accuracy
def accuracyFinder(training,test):
	posSmoothing,negSmoothing=train(training)
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
		#loading from pickle
		f1=open('pos.pickle','rb')
		f2=open('neg.pickle','rb')		
		pos_dict=pickle.load(f1)		
		neg_dict=pickle.load(f2)
		f1.close()
		f2.close()
		#calculating log probabilities for each review
		pos_prob=0
		neg_prob=0
		for word in reviewf:
			if word in pos_dict:
				pos_prob=pos_prob+pos_dict[word]
			else:
				pos_prob+=posSmoothing
			if word in neg_dict:
				neg_prob=neg_prob+neg_dict[word]
			else:
				neg_prob+=negSmoothing
		#evaluating for accuracy
        	if clas=='Neg':
        		givenNegreviews+=1
        	else:
        		givenPosreviews+=1
        	if(pos_prob>neg_prob):
         		obtainedPosreviews+=1
           		if clas=='Neg':
                		wrongCount+=1
            		else:
                		correctCount+=1
                		correctPos+=1
        	else:
            		obtainedNegreviews+=1
            		if clas=='Neg':
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
	line = row.split(",")
	review=line[1]
        clas=line[0]	
	if (clas=='Pos'):
		train_samples.append([review,"Pos"])
	else:
		train_samples.append([review,"Neg"])

#k-fold cross validation divsion of data
def k_fold_cross_validation(X, K, randomise = False):
	if randomise: from random import shuffle; X=list(X); shuffle(X)
	avgacc=0
	fold=1
	for k in xrange(K):
		print("fold"+str(fold)+" started")
		training = [x for i, x in enumerate(X) if i % K != k]
		test = [x for i, x in enumerate(X) if i % K == k]		
		avgacc = float(avgacc + accuracyFinder(training,test))
		fold+=1
	return float(avgacc/10)			
#end

print('Avverage accuracy is : {0}%'.format(k_fold_cross_validation(train_samples, 10)))
print "=============================================================="
