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

with open("SemEval2010_task8_all_data/SemEval2010_task8_training/TRAIN_FILE.TXT") as f:
    train_file = f.readlines()
    
sentences,relations = prepare_dataset(train_file)

n_relations = len(set(relations))
print("Found {} relations\n".format(n_relations))
print("Relations:\n{}".format(list(set(relations))))

from models import KerasTextClassifier
import numpy as np
from sklearn.model_selection import train_test_split

clf = KerasTextClassifier(input_length = 50,n_classes=n_relations,max_words=15000)
tr_sent, te_sent, tr_rel, te_rel = train_test_split(sentences, relations, test_size=0.1)
clf.fit(X=tr_sent, y=tr_rel, X_val=te_sent, y_val=te_rel,
         batch_size=10, lr=0.001, epochs=20)