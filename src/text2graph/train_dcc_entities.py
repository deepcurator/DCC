

#!/usr/bin/env python
# coding: utf8

from __future__ import unicode_literals, print_function

from brat2spacy import *
import time

import plac
import random
from pathlib import Path
import spacy
from spacy.util import minibatch, compounding
from spacy.gold import GoldParse
from spacy.scorer import Scorer

from ner_model_eval import *

# new entity label
ENTITIES = ['Method', 'Generic', 'Task', 'Material', 'Eval', 'Other']


def evaluate(ner_model, examples):
    scorer = Scorer()
    for input_, annot in examples:
        doc_gold_text = ner_model.make_doc(input_)
        gold = GoldParse(doc_gold_text, entities=annot)
        pred_value = ner_model(input_)
        scorer.score(pred_value, gold)
    return scorer.scores


#data_directory = 'DATA/abstract-sentences-test/'
data_directory = 'Output/BreakBrat/Abstracts-annotated30/'
output_dir = 'Models/'
TRAIN_DATA = create_training_data(data_directory)

@plac.annotations(
    model=("Model name. Defaults to blank 'en' model.", "option", "m", str),
    new_model_name=("New model name for model meta.", "option", "nm", str),
    output_dir=("Optional output directory", "option", "o", Path),
    n_iter=("Number of training iterations", "option", "n", int))


def main(model=None, new_model_name='DCC', output_dir=output_dir, n_iter=50):
    """Set up the pipeline and entity recognizer, and train the new entity."""
    if model is not None:
        nlp = spacy.load(model)  # load existing spaCy model
        print("Loaded model '%s'" % model)
    else:
        nlp = spacy.blank('en')  # create blank Language class
        print("Created blank 'en' model")
    # Add entity recognizer to model if it's not in the pipeline
    # nlp.create_pipe works for built-ins that are registered with spaCy
    if 'ner' not in nlp.pipe_names:
        ner = nlp.create_pipe('ner')
        nlp.add_pipe(ner)
    # otherwise, get it, so we can add labels to it
    else:
        ner = nlp.get_pipe('ner')

    for i in range(len(ENTITIES)):
      ner.add_label(ENTITIES[i])   # add new entity label to entity recognizer
    if model is None:
        optimizer = nlp.begin_training()
    else:
        # Note that 'begin_training' initializes the models, so it'll zero out
        # existing entity types.
        optimizer = nlp.entity.create_optimizer()

    training_start_time = time.time()
    # get names of other pipes to disable them during training
    other_pipes = [pipe for pipe in nlp.pipe_names if pipe != 'ner']
    with nlp.disable_pipes(*other_pipes):  # only train NER
        for itn in range(n_iter):
            random.shuffle(TRAIN_DATA)
            losses = {}
            # batch up the examples using spaCy's minibatch
            batches = minibatch(TRAIN_DATA, size=compounding(4., 32., 1.001))
            for batch in batches:
                texts, annotations = zip(*batch)
                nlp.update(texts, annotations, sgd=optimizer, drop=0.35,
                           losses=losses)
            print('iter:', itn)
            print('Losses', losses)

    training_end_time = time.time()
    print("training time: ", training_end_time-training_start_time)

    ############################
    # test the trained model
    #
    # define a sentence (or group of sentences) with no annotations.
    # the model will find the entities in it.
    test_text = 'non-local methods exploiting the self-similarity of natural signals have been well studied, ' \
                'for example in image analysis and restoration. existing approaches, however, rely on k-nearest ' \
                'neighbors (knn) matching in a ﬁxed feature space. the main hurdle in optimizing this feature space ' \
                'w. r. t. application performance is the non-differentiability of the knn selection rule. to overcome ' \
                'this, we propose a continuous deterministic relaxation of knn selection that maintains ' \
                'differentiability w. r. t. pairwise distances, but retains the original knn as the limit of a ' \
                'temperature parameter approaching zero. to exploit our relaxation, we propose the neural nearest ' \
                'neighbors block (n3 block), a novel non-local processing layer that leverages the principle of ' \
                'self-similarity and can be used as building block in modern neural network architectures.1 we show ' \
                'its effectiveness for the set reasoning task of correspondence classiﬁcation as well as for image ' \
                'restoration, including image denoising and single image super-resolution, where we outperform strong ' \
                'convolutional neural network (cnn) baselines and recent non-local models that rely on knn selection ' \
                'in hand-chosen features spaces.  1'
    doc = nlp(test_text)

    print("\nEntities in: '%s'" % test_text)
    for ent in doc.ents:
        print(ent.label_, ent.text, ent.start_char, ent.end_char)

    # write the results in a txt file
    ent_file = open("Output/entities.txt", "w+")
    for ent in doc.ents:
        ent_file.write("%s %s %d %d\n" % (ent.label_.encode("utf-8"), ent.text.encode("utf-8"), ent.start_char, ent.end_char))
    ent_file.close()


    ##########################
    # model evaluation



    examples = [
        ('Deep learning is applied in many every day application with great success in object recognition.',
         [(0, 13, 'Method'), (77, 95, 'Task')]),
        ('Recurrent neural networks are used for forecasting and natural language processing.',
         [(0, 25, 'Method'), (39, 50, 'Task'), (55, 82, 'Task')]),
        ('Convolutional neural networks are frequently used in object recognition and medical image processing.',
         [(0, 25, 'Method'), (39, 50, 'Task'), (55, 82, 'Task')])
    ]
    res = ner_eval(nlp, examples)
    print("\nModel evaluation results:")
    print(res)



    ############################################
    # save trained model
    if output_dir is not None:
        output_dir = Path(output_dir)
        if not output_dir.exists():
            output_dir.mkdir()
        nlp.meta['name'] = new_model_name  # rename model
        nlp.to_disk(output_dir)
        print("The model was saved to the directory: ", output_dir)

        # test the saved model
        #print("Loading from", output_dir)
        #nlp2 = spacy.load(output_dir)
        #doc2 = nlp2(test_text)
        #for ent in doc2.ents:
        #    print(ent.label_, ent.text)


if __name__ == '__main__':
    plac.call(main)