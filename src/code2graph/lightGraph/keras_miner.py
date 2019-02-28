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


def extract_keras_type(five_tuple):
    t = five_tuple.type
    l = five_tuple.line
    start = five_tuple.start[1]
    end = five_tuple.end[1]
    if l[start:end+1] == 'keras.':
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


def extract_keras_import(five_tuple):
    t = five_tuple.type
    l = five_tuple.line
    
    if 'from keras' in l:
        import_ix = l.find('import')
        import_ix = import_ix + len('import ')
        import_list = l[import_ix:].split(',')
        for i in range(len(import_list)):
            import_list[i] = import_list[i].strip()
        return import_list



def main():
    keras_dict = set()
    for filename in glob.iglob('./paperswithcode/data/**/*.py', recursive=True):
        with open(filename, 'rb') as f:
            itemset = []
            try:
                for five_tuple in tokenize.tokenize(f.readline):
                    mystring = five_tuple.string
                    mystring = mystring.strip()
                    if mystring and 'keras' in mystring:
                        if allowed_types(five_tuple.type):
                            keras_type = extract_keras_type(five_tuple)
                            keras_dict.add(keras_type)

                            keras_imports = extract_keras_import(five_tuple)
                            for imp in keras_imports:
                                keras_dict.add(imp)
            except:
                continue

    print(len(keras_dict))
    for i in keras_dict:
        print(i)

 

if __name__ == "__main__":
    main()
