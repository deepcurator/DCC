# -*- coding: utf-8 -*-
"""
Created on Tue Jan 15 19:39:03 2019

@author: Amar Viswanathan
"""

from xml.dom import minidom
import glob
import pandas as pd
import requests
from SPARQLWrapper import SPARQLWrapper, JSON
from IPython.display import display
import pandas as pd
import json
import numpy as np
from pandas.io.json import json_normalize
import nltk
from nltk.tokenize import WhitespaceTokenizer
from nltk.tokenize import sent_tokenize
import html



url = 'http://localhost:2222/rest/annotate'

def sentenceSplitter(text):
    return sent_tokenize(text)

def sentenceCreator(sentences):
    return [sentence for sentence in sentences]
    
def filterUnicode(text):
    text = text.encode('ascii', 'ignore').decode('unicode_escape')
    return text

def cleanText(text):
    newString = text.strip('\n')
    newString = newString.strip('\r')

    newString = newString.replace('\n'," ")
    newString = newString.replace('- ','')
    newString = newString.replace(r'&#12;','')
    return newString

def annotate(text):
    userdata = {"text": text, "confidence": "0.6", "support": "0"}
    resp = requests.post(url, data=userdata, headers = {'Accept' :'text/xml', 'Content-type': 'application/x-www-form-urlencoded'})
#  print(resp)
    return resp.text

def annotateToConll(text):
    mydoc = minidom.parseString(text)
    annotations = mydoc.getElementsByTagName('Resource')
    resourceURI=''
    for annotation in annotations:
        resourceURI = annotation.attributes['URI'].value
        surfaceForm = annotation.attributes['surfaceForm'].value
    if(resourceURI == ''):
        return 'O'
    else:
        return 'B-' + resourceURI
    

def offsets(text):
#    dictionary  = defaultdict(String)
#    print(text['Annotated'])
    print('\n\n')
    dictionary = {}
    mydoc = minidom.parseString(text['Annotated'])
#
    annotations = mydoc.getElementsByTagName('Resource')
    for annotation in annotations:
        resourceURI = annotation.attributes['URI'].value
        surfaceForm = annotation.attributes['surfaceForm'].value
        dictionary[surfaceForm] = 'B-' + resourceURI
        
       
    print(dictionary)
#    annotation = annotations[0].attributes['text'].value
#    print(annotation)
#    print(items[1].attributes['surfaceForm'].value)
#    print(items[1].attributes['offset'].value)
#    print(items[1].attributes['surfaceForm'].value + " isA "  +  items[1].attributes['URI'].value )
#    returnList = []
#    offset = 0
#    for index, word in enumerate(nltk.word_tokenize(text)):
#        returnList.append((offset, word))
#        offset = offset + len(word) + 1
    spangenerator = WhitespaceTokenizer().span_tokenize(text['Abstract'])
    tokenized = nltk.word_tokenize(text['Abstract'])
    offsets = [span[0] for span in spangenerator]
#    tokens = [nltk.pos_tag(nltk.word_tokenize(word)) for word in tokenized]
    tokens = nltk.pos_tag(tokenized)
    POS = [word[1] for word in tokens]
    
#    lstval = [dictionary[token] for token in tokenized]
    tag = []
    for token in tokenized:
        if(token in dictionary):
            tag.append(dictionary[token])
        else:
            tag.append('O')
    
    print(tag)
        
#    print(lstval)
#    print(POS)
#    print(type(tokens))
#    print(type(offsets))
#    test = list(zip(tokenized,POS,offsets))
#    print(type(test))
    
    return list(zip(tokenized,POS,tag))

    

root_dir = 'C:\\Users\\z003z47y\\Documents\\git\\darpa_aske_dcc\\src\\paperswithcode\\txt\\papersv2'
textIterator = glob.glob(root_dir + '**/*.txt', recursive=True)
#print(textIterator)

abstracts = []

start  = 'abstract'
end = 'introduction'

titles = []
for file in textIterator:
   with open(file,encoding='utf8') as f:
       title = f.readline()
       titles.append(str(title.strip()))
       lines = f.readlines()
       papertxt = ''.join(str(line) for line in lines)
       papertxt = papertxt.lower()
       result = papertxt.split(start)[1].split(end)[0]
       result  = result.rstrip('\r\n')
       abstracts.append(result)
       
#dictionary = dict(zip(titles,abstracts))

sentence_list = [sentenceSplitter(cleanText(abstract)) for abstract in abstracts]
sentences = [sentence for sublist in sentence_list for sentence in sublist]

sentence_indices =[]
words  = []
pos = []
for i, sentence in enumerate(sentences):
    tokenized = nltk.word_tokenize(sentence)
#    print(sentence + ":" + str(tokenized))
    for tokens in tokenized:
        sentence_indices.append('Sentence: '+ str(i))
#        print('Sentence: '+ str(i) + "," + tokens)
        words.append(tokens)
        word_list = [tokens]
        pos.append(nltk.pos_tag(word_list))

pos = [tup[0][1] for tup in pos]
df = pd.DataFrame(list(zip(sentence_indices,words,pos)),columns = ['Sentence#','Word','POS'])
df['Annotated'] = df['Word'].apply(annotate)
df.to_pickle('../../paperswithcode/txt/annotated/annotated_v2.pkl')
df['Tag'] = df['Annotated'].apply(annotateToConll)

tagged_df = df[['Sentence#','Word','POS','Tag']]
#tagged_df = tagged_df.rename(columns={'Sentence#':'Sentence #','POS':'Tag'})
tagged_df.to_pickle('../../paperswithcode/txt/annotated/annotated_v3.pkl')
tagged_df.to_csv('../../paperswithcode/txt/annotated/annotated_v3.csv')               

#colList = ["Title", "Abstract"]

#df = pd.DataFrame(list(dictionary.items()), columns = ['Title', 'Abstract'])
#df['Abstract'] = df['Abstract'].apply(filterUnicode)
#df['Abstract'] = df['Abstract'].apply(cleanText)
#
#df['Annotated'] = df['Abstract'].apply(annotate)
#
#results = []
#df.to_pickle('../../paperswithcode/txt/annotated/annotated.pkl')
#df.to_csv('../../paperswithcode/txt/annotated/annotated.csv')
#
#
#
#newdf2 = pd.read_csv('../../paperswithcode/txt/annotated/annotated.csv')
#
##newdf = pd.read_pickle('../../paperswithcode/txt/annotated/annotated.pkl')
##newdf2['Annotated'] = newdf2['Annotated'].apply(filterUnicode)
##newdf2['Offsets'] = newdf2.apply(offsets,axis=1)
#
#newdf2['Sentences'] = newdf2['Abstract'].apply(sentenceSplitter)
#a = []
#a.append(sentenceCreator())



