# -*- coding: utf-8 -*-
"""
Created on Wed Mar 20 07:58:09 2019

@author: Amar Viswanathan
"""
import pandas as pd

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
#        print(sentence)
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
                replacementText.append('<' +  tempString[0] + '>' + textString + '</' +  tempString[0] + '> ')
                entityCount +=1
            elif line.startswith('R'):
                temprelString = line.split('\t')
                relString = temprelString[1].split(' ')
#                print(len(relString))
                relationship += relString[0]
#                print(relString[0])
#                print(relString[1].split(":")[1])
#                print(relString[2].split(":")[1])
                stringToBeAdded = '(' + relString[1].split(":")[1] + ',' + relString[2].split(":")[1] + ')'
                relationship += stringToBeAdded + "\n"
    newString1 = ""
    for (t,r) in zip(text,replacementText):
        newString1 = sentence.replace(t,r)
        sentence = newString1
       
#    print(newString1)
#    print(relationship)
    returnText = newString1 + "\n" + relationship
    return returnText

import glob
annFiles = glob.glob("training-ioannis\*.ann")
txtFiles = [file.split('.')[0] + '.txt' for file in annFiles]

count = 1
for (annFile,txtFile) in zip(annFiles,txtFiles):
    text = annToSemEval(annFile,txtFile)
    print(str(count) + "\t" + text)
    print("\n")
    count = count+1
#testString = "we present foundations for using model predictive control (mpc) as a differentiable policy class for reinforcement learning in continuous state and action spaces."
#newS = testString.replace('model predictive control (mpc)', '<e1>model predictive control (mpc)</e1>')
#print(text[0])
#print(replacementText[0])

#zipped_list = list(zip(ids,NER,text))
#print(zipped_list)
#
#df = pd.DataFrame(zipped_list,columns = ['ID','NER','text',replacementText])
#df.head()


    
        