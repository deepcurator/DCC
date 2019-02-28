import os
import glob
import tokenize

import pandas as pd
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules


def main():

    baskets = []
    for filename in glob.iglob('./paperswithcode/data_test/**/*.py', recursive=True):

        with open(filename, 'rb') as f:
            itemset = []
            try:
                for five_tuple in tokenize.tokenize(f.readline):
                    mystring = five_tuple.string
                    mystring = mystring.strip()
                    if mystring:
                        itemset.append(mystring)
                baskets.append(itemset)
            except:
                continue
        print('%s has %s items' % (filename.split('/')[-1], len(itemset)))


    te = TransactionEncoder()
    te_ary = te.fit(baskets).transform(baskets)
    df = pd.DataFrame(te_ary, columns=te.columns_)
    frequent_itemsets = apriori(df, min_support=0.6, use_colnames=True)
    print(frequent_itemsets)


    rules = association_rules(frequent_itemsets, metric="lift", min_threshold=1.2)
    print(rules)


if __name__ == "__main__":
    main()
