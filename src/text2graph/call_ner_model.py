
# An already trained and saved NER model is used to annotate new text.
# The new text does not contain any annotations and may have been extracted
# from a pdf paper describing a DL method, architecture or application.
# The entites we consider are: ['Method', 'Generic', 'Task', 'Material', 'Eval', 'Other']
#
# This material is based upon work supported by Defense Advanced Research
# Projects Agency (DARPA) under Agreement No. HR00111990010


from __future__ import unicode_literals, print_function


import os
import plac
from pathlib import Path
import spacy
import yaml

config = yaml.safe_load(open('../../conf/conf.yaml'))
model_dir = config['MODEL_PATH']
test_dir = config['TEST_DATA_PATH']
output_dir = config['TEXT_OUTPUT_PATH']

# passing command line arguments using plac
@plac.annotations(
    saved_model_dir=("Directory where the trained model is saved", "option", "sm", str),
    output_dir=("Optional output directory", "option", "o", Path),
    test_dir=("optional directory containing test data", "option", "t", Path),
)



# It uses a saved NER model to annotate new text.
# Input -
#   saved_model_dir: the path of the directory where the saved (trained) model is
#   test_dir: the path of the directory where the text exists (the text may be an abstract, a section of a DL paper, or the entire paper)
#   output_dir: the path of the directory where the entities will be saved
# Output -
#   The new entities predicted by the NER model for each input text file are stored in individual files in the output_dir directory

def main(saved_model_dir=model_dir, test_dir=test_dir, output_dir=output_dir):
    # first collect all file names in the test directory
    files_list = []
    for root, dirs, files in os.walk(test_dir):
        for file1 in files:
          files_list.append(file1)

    # next go through all files in the test directory, read the text they contain,
    # and predict the entities that exist in them.
    test_files = os.scandir(test_dir)
    file_counter = 0
    for test_file in test_files:
        curr_file = os.path.join(test_file)
        with open(curr_file, "r+", encoding="utf8") as test_file:
           test_text = test_file.read()
           nlp = spacy.load(saved_model_dir)
           doc = nlp(test_text)

           file_name = str(files_list[file_counter])
           file_name = file_name.replace('.txt', '')
           ents_file = os.path.join(output_dir, file_name + "_ents_from_existing_model" + ".txt")
           with open(ents_file, "w+", encoding="utf8") as ef:
             for ent in doc.ents:
               ef.write("%s %s %d %d\n" % (ent.label_.encode("utf-8"), ent.text.encode("utf-8"), ent.start_char, ent.end_char))

           file_counter = + 1
        
    print("The predicted entities we stored in ", output_dir)



if __name__ == '__main__':
    plac.call(main)
