import re
import csv
import sys
import string
import nltk

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

stopWords = getStopWordList('stopwords.txt')
uniqueWords={}
l = open("data.txt", "r").readlines()
for row in l:
	line = row.split("\t")
	review=line[1]
	clas=line[0]
	replace_punctuation = string.maketrans(string.punctuation, ' '*len(string.punctuation))
	reviewf=processReview(review)
	reviewf=reviewf.translate(replace_punctuation)
	reviewf=getFeatureVector(review,stopWords)
	for word in reviewf:
		if word not in uniqueWords: 
			if len(word)!=1:
				uniqueWords[word]=1
		else:
				uniqueWords[word]+=1			
sys.stdout.write("finsihed words extraction \n")
f=open('vocabulary.txt','w')
for i,count in uniqueWords.items():
	if count>=2:
		f.write(i+"\n")
sys.stdout.write("vocabulary.txt file created.\n")
