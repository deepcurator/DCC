# Whole Graph Convlutions

## Requirements
1. gcn https://github.com/tkipf/gcn/tree/master/gcn
2. gae https://github.com/tkipf/gae

## Dataset
1. Uncompress UCI_TF_Papers/rdf_triples to the DCC/src/code2graph/core/gcn directory.
2. I have created labels.csv for the UCI_TF_Papers/rtf_triples consisting of 10 labels (ML, NLP, GAN, ...).
3. I have created labels2.csv for the same dataset consisting of 4 labels.

## Embedding
Using plain GCN use: python gcn_train.py

Using GAE use: python gae_train.py

## TODO
For GCN we need to get to the embeddings. For GAE we need to do whole graph embedding instead of link prediction.

## Example output from GAE

