import os
import glob
import tokenize

import pandas as pd
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules



# Ignore comments, strings, and encodings
# (COMMENT) 57, (STRING) 3, (ENCODING) 59
def allowed_types(t):
    if t == 57 or t == 3 or t == 59:
        return False
    return True


def extract_tf_type(five_tuple):
    t = five_tuple.type
    l = five_tuple.line
    start = five_tuple.start[1]
    end = five_tuple.end[1]
    if l[start:end+1] == 'tf.':
        pos = end+1
        while l[pos]:
            if l[pos] == '(':
                break
            if l[pos] == ' ':
                break
            if l[pos] == ',':
                break
            if l[pos] == ')':
                break
            pos = pos+1
        return l[start:pos]


def main():
    baskets = []
    tf_dict = set()
    for filename in glob.iglob('./paperswithcode/data/**/*.py', recursive=True):
        with open(filename, 'rb') as f:
            itemset = []
            try:
                for five_tuple in tokenize.tokenize(f.readline):
                    mystring = five_tuple.string
                    mystring = mystring.strip()
                    if mystring and 'tf' in mystring:
                        if allowed_types(five_tuple.type):
                            tf_type = extract_tf_type(five_tuple)
                            tf_dict.add(tf_type)
                            itemset.append(mystring)
                            #print(five_tuple)
                baskets.append(itemset)
            except:
                continue
        #print('%s has %s items' % (filename.split('/')[-1], len(itemset)))
    print(len(tf_dict))
    
    for i in tf_dict:
        print(i)

 

if __name__ == "__main__":
    main()
