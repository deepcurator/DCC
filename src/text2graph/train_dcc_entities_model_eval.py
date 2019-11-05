

# Use of a trained NER model to evaluate a set of manually labeled sentences
#
# This material is based upon work supported by Defense Advanced Research
# Projects Agency (DARPA) under Agreement No. HR00111990010



import yaml
import time
import plac
import random
from pathlib import Path
import spacy

from ner_utils import ner_eval, test_ner_model

# new entity label
new_entities_list = ['Method', 'Generic', 'Task', 'Material', 'Eval', 'Other']
#new_entities_list = ['Method', 'Generic', 'Task', 'Material', 'Metric', 'OtherScientificTerm']


config = yaml.safe_load(open('../../conf/conf.yaml'))
model_dir = config['MODEL_PATH']
test_dir = config['TEST_DATA_PATH']

# passing command line arguments using plac
@plac.annotations(
    model=("Model name. Defaults to blank 'en' model.", "option", "m", str),
    new_model_name=("New model name for model meta.", "option", "nm", str),
    test_dir=("optional directory containing test data", "option", "t", Path))

# The main function that sets up the SpaCy pipeline and entity recognizer. The new entities are defined as a list of strings.
# Input -
#   model: the name of an existing trained model
#   new_model_name: the name of the new entity model
#   output_dir: the path of the directory where the new trained model will be saved.
#   n_iter: number of training iterations (epochs)
# Output -
#   The trained entity model stored in the output_dir
#def main(model=None, new_model_name='DCC_ent', input_dir=input_dir, saved_model_dir=model_dir, output_dir=output_dir, test_dir=test_dir, n_iter=n_iter):
def main(model=None, new_model_name='DCC_ent', test_dir=test_dir):
    random.seed(1234)

    nlp = spacy.load(model_dir)

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
         [(0, 29, 'Method'), (53, 72, 'Task'), (84, 101, 'Task')],
         )
    ]
    res = ner_eval(nlp, examples)
    print("\nModel evaluation results:")
    print(res)

    print("f1 = ", res['ents_f'])
    print("r  = ", res['ents_r'])
    print("p  = ", res['ents_p'])


if __name__ == '__main__':
    plac.call(main)

