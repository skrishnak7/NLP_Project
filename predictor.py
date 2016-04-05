import csv
import re
import sys
import string
import pickle
import numpy as np

def processReview(review):
    # process the reviews

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
       # w = w.strip('\'"?,.')
        #check if the word stats with an alphabet
        val = re.search(r"^[a-zA-Z][a-zA-Z0-9]*$", w)
        #ignore if it is a stop word
        if(w in stopWords or val is None):
            continue
        else:
            featureVector.append(w.lower())
    return featureVector
#end

stopWords = getStopWordList('stopwords.txt')

'''
posWords=[]
negWords=[]
uniqueWords={}
#training given data 
l = open("data.txt", "r").readlines()
for row in l:
	line = row.split(",")
	review=line[1]
       	clas=line[0]
       	replace_punctuation = string.maketrans(string.punctuation, ' '*len(string.punctuation))
	reviewf=processReview(review)
       	reviewf=reviewf.translate(replace_punctuation)
       	reviewf=getFeatureVector(review,stopWords)
       	for word in reviewf:
	    if word not in uniqueWords:
		uniqueWords[word]=1
	    else:
		uniqueWords[word]+=1
	    if clas=="Neg":
		    	negWords.append(word)
       	    else:
			posWords.append(word) 
	#removing words having occurences less than 2
	for word,count in uniqueWords.items():
		if count<2:
			del uniqueWords[word]
#dictionaries having words and their positive and negative probabilities
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
print("Fininished Training the data \n")
posSmoothing=np.log10(1.0/(len(posWords)+len(set(posWords))))
negSmoothing=np.log10(1.0/(len(negWords)+len(set(negWords))))
'''

#testing for a given review
review="its a very awesome movie."
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
#calculating log probabilities for given review 
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
if(pos_prob>neg_prob):
	print "Positive"
else:
	print "Negative"
