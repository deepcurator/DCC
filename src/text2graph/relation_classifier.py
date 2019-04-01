#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 11 12:07:15 2019

@author: Amar Viswanathan
"""

def prepare_dataset(raw):
    sentences, relations = [], []
    to_replace = [("\"", ""), ("\n", ""), ("<", " <"), (">", "> ")]
    last_was_sentence = False
    for line in raw:
        sl = line.split("\t")
        if last_was_sentence:
            relations.append(sl[0].split("(")[0].replace("\n", ""))
            last_was_sentence = False
        if sl[0].isdigit():
            sent = sl[1]
            for rp in to_replace:
                sent = sent.replace(rp[0], rp[1])
            sentences.append(sent)
            last_was_sentence = True
    print("Found {} sentences".format(len(sentences)))
    return sentences, relations

with open("/home/amar/git/DCC/src/text2graph/Data/Abstracts-annotated/semeval/semeval_train_new.txt") as f:
    train_file = f.readlines()
    
sentences,relations = prepare_dataset(train_file)

temp = [bool(not s or s.isspace()) for s in sentences]
sentence1 = []
relation1 = []

for (sentence,relation,flag) in zip(sentences,relations,temp):
    if(not flag):
        sentence1.append(sentence)
        relation1.append(relation)

temp_rel = [bool(s.isdigit()) for s in relation1]

sentence2 = []
relation2 = []

for (sentence,relation,flag) in zip(sentence1,relation1,temp_rel):
    if(not flag):
        sentence2.append(sentence)
        relation2.append(relation)

sentences = sentence2
relations = relation2

n_relations = len(set(relations))
print("Found {} relations\n".format(n_relations))
print("Relations:\n{}".format(list(set(relations))))

print(sentences[3])
print(relations[3])

from models import KerasTextClassifier
import numpy as np
from sklearn.model_selection import train_test_split

clf = KerasTextClassifier(input_length = 50,n_classes=n_relations,max_words=15000)
tr_sent, te_sent, tr_rel, te_rel = train_test_split(sentences, relations, test_size=0.1)
clf.fit(X=tr_sent, y=tr_rel, X_val=te_sent, y_val=te_rel,
         batch_size=10, lr=0.001, epochs=20)

## Save model to file
clf.save("/home/amar/git/DCC/src/text2graph/model")

## Print the metrics

from sklearn.metrics import f1_score, classification_report, accuracy_score
y_test_pred = newclf.predict(te_sent)
label_dict = {}
for i,c in enumerate(list(clf.encoder.classes_)):
    print(str(i) + ": " + c)
    label_dict[i] = c

## Load a new model
newclf = KerasTextClassifier()
newclf.load("/home/amar/git/DCC/src/text2graph/model")

## Show predictions side by side
for i,sentence in enumerate(te_sent):
    print(sentence + ":\n" +  label_dict.get(y_test_pred[i]) + "\n")
