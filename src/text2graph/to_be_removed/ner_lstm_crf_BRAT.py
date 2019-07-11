# Perform Named Entity Recognition  using LSTM and Conditional Random Fields
# Keras with TF backend is used to construct the network architectures.
#
#
#Parts of this code were taken from https://www.depends-on-the-definition.com/sequence-tagging-lstm-crf/
#

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from keras.models import Model, Input
from keras.layers import LSTM, Embedding, Dense, TimeDistributed, Dropout, Bidirectional
from keras_contrib.layers import CRF
from seqeval.metrics import precision_score, recall_score, f1_score, classification_report

fpath='C:/Users/dfradkin/Desktop/abstracts/'
datalist=[]
for n in range(95):
    f=os.path.join(fpath,str(n)+'.conll')    
    if os.path.exists(f):
        data = pd.read_csv(f, encoding="latin1", delimiter='\t', header=None)
        data = data.fillna(method="ffill")
        data.columns=['Tag','Start','End','Word']
        datalist.append(data)
data=pd.concat(datalist)    
data['Sentence #']=np.cumsum(data['Word'] == '.')

print(data.tail(10))

words = list(set(data["Word"].values))
words.append("ENDPAD")
n_words = len(words);
print("number of words: ", n_words)

tags = list(set(data["Tag"].values))
n_tags = len(tags);
print("number of tags: ", n_tags)


class SentenceGetter(object):
  def __init__(self, data):
    self.n_sent = 1
    self.data = data
    self.empty = False
    agg_func = lambda s: [(w, p, t) for w, p, t in zip(s["Word"].values.tolist(),
                                                       s["Start"].values.tolist(),
                                                       s["Tag"].values.tolist())]
    self.grouped = self.data.groupby("Sentence #").apply(agg_func)
    self.sentences = [s for s in self.grouped]

  def get_next(self):
    try:
      s = self.grouped["Sentence: {}".format(self.n_sent)]
      self.n_sent += 1
      return s
    except:
      return None


getter = SentenceGetter(data)
sent = getter.get_next()
print(sent)

# get all sentences
sentences = getter.sentences

# create dictionaries of words and tags
max_len = 75
word2idx = {w: i + 1 for i, w in enumerate(words)}
tag2idx = {t: i for i, t in enumerate(tags)}

# check the index for specific words and tags
#print(word2idx["Obama"])
#print(tag2idx["B-geo"])

X = [[word2idx[w[0]] for w in s] for s in sentences]

X = pad_sequences(maxlen=max_len, sequences=X, padding="post", value=n_words-1)


y = [[tag2idx[w[2]] for w in s] for s in sentences]

y = pad_sequences(maxlen=max_len, sequences=y, padding="post", value=tag2idx["O"])


# change the labels y to categorical
y = [to_categorical(i, num_classes=n_tags) for i in y]

# split in train and test sets
X_tr, X_te, y_tr, y_te = train_test_split(X, y)


# setup the CRF-LSTM
input = Input(shape=(max_len,))
model = Embedding(input_dim=n_words + 1, output_dim=20,
                  input_length=max_len, mask_zero=True)(input)  # 20-dim embedding
model = Bidirectional(LSTM(units=50, return_sequences=True,
                           recurrent_dropout=0.1))(model)  # variational biLSTM
model = TimeDistributed(Dense(50, activation="relu"))(model)  # a dense layer as suggested by neuralNer
crf = CRF(n_tags)  # CRF layer
out = crf(model)  # output

model = Model(input, out)
model.compile(optimizer="rmsprop", loss=crf.loss_function, metrics=[crf.accuracy])
model.summary()

history = model.fit(X_tr, np.array(y_tr), batch_size=64, epochs=4,
                    validation_split=0.1, verbose=1)

hist = pd.DataFrame(history.history)

print(hist.columns.values)
plt.style.use("ggplot")
plt.figure(figsize=(12,12))
plt.plot(hist["crf_viterbi_accuracy"])
plt.plot(hist["val_crf_viterbi_accuracy"])
plt.show()


# evaluation of the model
test_pred = model.predict(X_te, verbose=1)

idx2tag = {i: w for w, i in tag2idx.items()}


def pred2label(pred):
  out = []
  for pred_i in pred:
    out_i = []
    for p in pred_i:
      p_i = np.argmax(p)
      out_i.append(idx2tag[p_i].replace("PAD", "O"))
    out.append(out_i)
  return out


pred_labels = pred2label(test_pred)
test_labels = pred2label(y_te)

print("F1-score: {:.1%}".format(f1_score(test_labels, pred_labels)))

print(classification_report(test_labels, pred_labels))



# perform some predictions
#i = 35
#p = model.predict(np.array([X_te[i]]))
#p = np.argmax(p, axis=-1)
#true = np.argmax(y_te[i], -1)
#print("{:15}||{:5}||{}".format("Word", "True", "Predicted"))
#print(30 * "=")
#for w, t, pred in zip(X_te[i], true, p[0]):
#    if w != 0:
#        print("{:15}: {:5} {}".format(words[w-1], tags[t], tags[pred]))
#
#
#
## inference with lstm-crf using new data
## (this can be a new abstract or a new deep learning paper)
#test_sentence = ["attention"]
#
## Transform every word to its integer index
#x_test_sent = pad_sequences(sequences=[[word2idx.get(w, 0) for w in test_sentence]],
#                            padding="post", value=0, maxlen=max_len)
#
#p = model.predict(np.array([x_test_sent[0]]))
#p = np.argmax(p, axis=-1)
#print("{:15}||{}".format("Word", "Prediction"))
#print(30 * "=")
#for w, pred in zip(test_sentence, p[0]):
#    print("{:15}: {:5}".format(w, tags[pred]))

