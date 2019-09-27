
# We train a model for Relation Extraction from text that contains both entities
# and relations. The input to the model is a dataset of sentences where the entities
# have been identified using the SemEval standard. The dataset has been generated from
# the collected papers that have been annotated using Brat (http://brat.nlplab.org/).
# The training is done by using a Bidirectional LSTM model that uses an attention mechanism.
#
# This material is based upon work supported by Defense Advanced Research
# Projects Agency (DARPA) under Agreement No. HR00111990010


from __future__ import unicode_literals, print_function

import os
import yaml
from models import KerasTextClassifier
import numpy as np
from sklearn.model_selection import train_test_split


config = yaml.safe_load(open('../../conf/conf.yaml'))
input_dir = config['SENTENCE_ANNOTATED_TEXT_PATH_SEMEVAL']
model_dir = config['MODEL_PATH']
# test_dir = config['TEST_DATA_PATH']
# output_dir = config['TEXT_OUTPUT_PATH']


# Prepare the dataset of sentences in a format that can be used by a BiLSTM model developed
# by using TF/Keras framework
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



# Load the annotated text/sentences (derived from the DL papers).
input_filename = "SemEval_Output.txt"
input_path_and_file=os.path.join(input_dir, input_filename)
with open(input_path_and_file, encoding="utf8") as f:
    train_file = f.readlines()


# Collect the sentences where the entities have been marked, and the corresponding relations
sentences, relations = prepare_dataset(train_file)

# check how many relations exist in the dataset
# n_relations = len(set(relations))
# print("found {} relations \n ".format(n_relations))
# print("relations: {}".format(list(set(relations))))


# Define the BiLSTM model for RE
classif = KerasTextClassifier(input_length=50, n_classes=n_relations, max_words=15000)

# Define the training and testing sets for both the sentences/entities and relationships
tr_sent, te_sent, tr_rel, te_rel = train_test_split(sentences, relations, test_size=0.1)

# Fit the BiLSTM model
classif.fit(X=tr_sent, y=tr_rel, X_val=te_sent, y_val=te_rel,
            batch_size=10, lr=0.001, epochs=50)

# Save the model for future use
# re_model_filename = "re_model_raw_data"
re_model_filename = "re_model_latest_auto_ann_run_2019_25_09"
re_model_path_and_file=os.path.join(model_dir, re_model_filename)
classif.save(re_model_path_and_file)


# evaluate the re model
from sklearn.metrics import f1_score, classification_report, accuracy_score
y_test_pred = classif.predict(te_sent)

label_idx_to_use = [i for i, c in enumerate(list(classif.encoder.classes_)) if c != "Other"]
label_to_use = list(classif.encoder.classes_)

# print("F1 score: [:.1%]".format(f1_score(classif.encoder.transform(te_rel), y_test_pred,
#                                          average="macro", labels=label_idx_to_use)))


print(classification_report(classif.encoder.transform(te_rel), y_test_pred,
                            target_names=label_to_use,
                            labels=label_idx_to_use))

