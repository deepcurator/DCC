"""
code from: https://www.depends-on-the-definition.com/attention-lstm-relation-classification/
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
sentences, relations = prepare_dataset(train_file)

sentences[156]
relations[156]

n_relations = len(set(relations))
print("Found {} relations\n".format(n_relations))
print("Relations:\n{}".format(list(set(relations))))

################# Model Training

from keras_attention_models import KerasTextClassifier
import numpy as np
from sklearn.model_selection import train_test_split

kclf = KerasTextClassifier(input_length=50, n_classes=n_relations, max_words=15000)
tr_sent, te_sent, tr_rel, te_rel = train_test_split(sentences, relations, test_size=0.1)
kclf.fit(X=tr_sent, y=tr_rel, X_val=te_sent, y_val=te_rel,
         batch_size=10, lr=0.001, epochs=20)


############ Investigate Attention:
import matplotlib.pyplot as plt
#plt.style.use("ggplot")
#%matplotlib inline

y_pred = kclf.predict(te_sent)
y_attn = kclf._get_attention_map(te_sent)

i = 354
activation_map = np.expand_dims(y_attn[i][:len(te_sent[i].split())], axis=1)

f = plt.figure(figsize=(8, 8))
ax = f.add_subplot(1, 1, 1)

img = ax.imshow(activation_map, interpolation='none', cmap='gray')

plt.xlim([0,0.5])
ax.set_aspect(0.1)
ax.set_yticks(range(len(te_sent[i].split())))
ax.set_yticklabels(te_sent[i].split());
ax.grid()
plt.title("Attention map of sample {}\nTrue relation: {}\nPredicted relation: {}"
          .format(i, te_rel[i], kclf.encoder.classes_[y_pred[i]]));

# add colorbar
cbaxes = f.add_axes([0.2, 0, 0.6, 0.03]);
cbar = f.colorbar(img, cax=cbaxes, orientation='horizontal');
cbar.ax.set_xlabel('Probability', labelpad=2);

############ Evaluation:
from sklearn.metrics import f1_score, classification_report, accuracy_score

y_test_pred = kclf.predict(te_sent)

label_idx_to_use = [i for i, c in enumerate(list(kclf.encoder.classes_)) if  c !="Other"]
label_to_use = list(kclf.encoder.classes_)
label_to_use.remove("Other")

print("F1-Score: {:.1%}"
      .format(f1_score(kclf.encoder.transform(te_rel), y_test_pred,
                       average="macro", labels=label_idx_to_use)))

print(classification_report(kclf.encoder.transform(te_rel), y_test_pred,
                            target_names=label_to_use,
                            labels=label_idx_to_use))

