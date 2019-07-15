#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 26 11:58:33 2019

@author: Amar Viswanathan
"""

def annToSemEval(annFile,txtfile):
    
    ids =  []
    NER = []
    text = []
    sentences = []
    replacementText = []
    entityCount = 1
    relationship = ""
    with open(annFile, "r", encoding = 'utf8') as a, open(txtfile, "r", encoding = 'utf8') as b:
        lines = a.readlines()
        sentence = b.readlines()
        sentence = ''.join(sentence)
        print(sentence)
        sentences.append(sentence)
        relString = ""
        entityCount = 1
        for line in lines:
            if line.startswith('T'):
    #            print(line)
                tempString = line.split('\t')
                ids.append(tempString[0])
                NER.append(tempString[1])
                textString = tempString[2].replace('\n'," ")
                text.append(textString)
                replacementText.append('<T' +  str(entityCount) + '>' + textString + '</T' +  str(entityCount) + '> ')
                entityCount +=1
            elif line.startswith('R'):
                temprelString = line.split('\t')
                relString = temprelString[1].split(' ')
                print(len(relString))
                relationship += relString[0]
                print(relString[0])
                print(relString[1].split(":")[1])
                print(relString[2].split(":")[1])
                stringToBeAdded = '(' + relString[1].split(":")[1] + ',' + relString[2].split(":")[1] + ')'
                relationship += stringToBeAdded

    for (t,r) in zip(text,replacementText):
        newString1 = sentence.replace(t,r)
        sentence = newString1
       
    print(newString1)
    print(relationship)
    returnText = newString1 + "\n" + relationship
    return returnText


import glob
print(glob.glob("/Data/*.ann"))