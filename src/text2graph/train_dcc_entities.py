

# A new NER model is trained that contains new entities. The new entities are
# defined in the list: ['Method', 'Generic', 'Task', 'Material', 'Eval', 'Other']
# The new model takes as input annotated sentences extracted from pdf files,
# describing methods, architectures, and applications of Deep Learning. The
# sentences have been annotated using BRAT (http://brat.nlplab.org/).
# The training is done by using the statistical models provided by spaCy
# (https://spacy.io/). The trained model can be saved in a user defined folder
# for future use.
#
# This material is based upon work supported by Defense Advanced Research
# Projects Agency (DARPA) under Agreement No. HR00111990010


from __future__ import unicode_literals, print_function

import yaml
import time
import plac
import random
from pathlib import Path
import spacy
from spacy.util import minibatch, compounding
from spacy.util import decaying
from spacy.pipeline import EntityRuler

from brat2spacy import create_training_data
from ner_utils import ner_eval, test_ner_model

# new entity labels
new_entities_list = ['Method', 'Generic', 'Task', 'Material', 'Eval', 'Other']
#new_entities_list = ['Method', 'Generic', 'Task', 'Material', 'Metric', 'OtherScientificTerm']


config = yaml.safe_load(open('../../conf/conf.yaml'))
input_dir = config['SENTENCE_ANNOTATED_TEXT_PATH']
model_dir = config['MODEL_PATH']
test_dir = config['TEST_DATA_PATH']
output_dir = config['TEXT_OUTPUT_PATH']

#input_dir = 'C:/Home02/src02/DCCdev_921/grobid/Text_Files_In_Sentences_V3_Partial/'
#input_dir = 'C:/Home02/src02/DCCdev_921/All_Data_For_NER/'
#input_dir = 'C:/Home02/src02/DCCdev_921/extracted_captions/'

n_iter = 50

# passing command line arguments using plac
@plac.annotations(
    model=("Model name. Defaults to blank 'en' model.", "option", "m", str),
    new_model_name=("New model name for model meta.", "option", "nm", str),
    input_dir=("Input directory containing the Brat data files", "option", "i", str),
    saved_model_dir=("Directory where the trained model is saved", "option", "sm", str),
    output_dir=("Optional output directory", "option", "o", Path),
    test_dir=("optional directory containing test data", "option", "t", Path),
    n_iter=("Number of training iterations", "option", "n", int))

# The main function that sets up the SpaCy pipeline and entity recognizer. The new entities are defined as a list of strings.
# Input -
#   model: the name of an existing trained model
#   new_model_name: the name of the new entity model
#   output_dir: the path of the directory where the new trained model will be saved.
#   n_iter: number of training iterations (epochs)
# Output -
#   The trained entity model stored in the output_dir
def main(model=None, new_model_name='DCC_ent', input_dir=input_dir, saved_model_dir=model_dir, output_dir=output_dir, test_dir=test_dir, n_iter=n_iter):
    random.seed(1234)

    # create the training from annotated data produced by using Brat
    data_reading_start_time = time.time()
    training_data = create_training_data(input_dir)
    data_reading_end_time = time.time()
    data_reading_time = data_reading_end_time - data_reading_start_time
    print("--->data reading time: ", data_reading_time)

    # check if the user provides an existing language model
    if model is not None:
        nlp = spacy.load(model)  # load existing spaCy model
        print("Loaded existing model '%s'" % model)
    else:
        nlp = spacy.blank('en')  # create blank Language class
        print("No model provided, created blank 'en' model")

    # Add entity recognizer to model if it's not in the pipeline
    # nlp.create_pipe works for built-ins that are registered with spaCy
    if 'ner' not in nlp.pipe_names:
        ner = nlp.create_pipe('ner')
        nlp.add_pipe(ner)
    else:
        # otherwise, get it, so we can add labels to it
        ner = nlp.get_pipe('ner')

    # add all new entities to the recognizer
    for i in range(len(new_entities_list)):
      ner.add_label(new_entities_list[i])

    if model is None:
        optimizer = nlp.begin_training()
    else:
        # Note that 'begin_training' initializes the models, so it'll zero out
        # existing entity types.
        optimizer = nlp.entity.create_optimizer()

    # start the training of the recognizer (and the time)
    training_start_time = time.time()
    for itn in range(n_iter):
        iter_start_time = time.time()
        dropout = decaying(0.4, 0.2, 1.0e-2)
        random.shuffle(training_data)
        losses = {}
        # batch up the examples using spaCy's minibatch
        batches = minibatch(training_data, size=compounding(4., 32., 1.001))
        for ib, batch in enumerate(batches):
            # print("     batch ", ib)
            ignore_batch = False
            for bl in range(len(batch)):
                # print(batch[bl])
                # print(len(batch[bl]))
                if len(batch[bl]) < 2:
                    ignore_batch = True
            if ignore_batch == True:
                continue
            texts, annotations = zip(*batch)
            # print(texts)
            # print(annotations)
            nlp.update(texts, annotations, sgd=optimizer, drop=0.35,
                       losses=losses)
        iter_end_time = time.time()
        iter_elapsed_time = iter_end_time - iter_start_time
        print('     iter:', itn)
        print('     Losses', losses)
        print('     iter elapsed time:', iter_elapsed_time)

    training_end_time = time.time()
    print("training time: ", training_end_time-training_start_time)

    ############################
    # test the ner model on a set of text data taken from papers
    # (if the user does not provide text data, no testing will be performed)
    if test_dir is not None:
        # test_ner_model(nlp, test_dir)
        test_ner_model(nlp, test_dir, output_dir,out_tag='_ents_from_existing_model')

    ##########################
    # model evaluation
    #
    # define a set of examples that will be used as ground truth
    examples = [
        ('Deep learning is applied in many every day application with great success in object recognition.',
         [(0, 13, 'Method'), (77, 95, 'Task')]),
        ('Recurrent neural networks are used for forecasting and natural language processing.',
         [(0, 25, 'Method'), (39, 50, 'Task'), (55, 82, 'Task')]),
        ('Convolutional neural networks are frequently used in object recognition and medical image processing.',
         [(0, 29, 'Method'), (53, 72, 'Task'), (84, 101, 'Task')])
    ]
    res = ner_eval(nlp, examples)
    print("\nModel evaluation results:")
    print(res)



    ############################################
    # save trained model
    # (if the user does not provide a directory, the trained model will not be saved)
    if saved_model_dir is not None:
        saved_model_dir = Path(saved_model_dir)
        if not saved_model_dir.exists():
            saved_model_dir.mkdir()
        nlp.meta['name'] = new_model_name  # rename model
        nlp.to_disk(saved_model_dir)
        print("The model was saved to the directory: ", saved_model_dir)

        # test the saved model
        #print("Loading from", output_dir)
        #nlp2 = spacy.load(output_dir)
        #doc2 = nlp2(test_text)
        #for ent in doc2.ents:
        #    print(ent.label_, ent.text)


if __name__ == '__main__':
    plac.call(main)
