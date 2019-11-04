

'''
 function that produces various scores of the trained NER model
 Input: the trained NER model,
        a set of examples with annotations incorporated within;
           the structure is as follows:
           [('sentence01', [(start01, end01, 'entity type 01')]),
            ('sentence02', [(start02-1, end02-1, 'entity type 02-1'),
                            (start02-2, end02-2, 'entity type 02-2')])
           ]
 Output: various evaluation scores, including precision, f1, ...
'''

from spacy.gold import GoldParse
from spacy.scorer import Scorer

def ner_eval(model, examples):
  scorer = Scorer()
  for input_, annot in examples:
    gold_text = model.make_doc(input_)
    gold = GoldParse(gold_text, entities=annot)
    pred_val = model(input_)
    scorer.score(pred_val, gold)
  return scorer.scores

'''
Perform testing
'''
import os


def collect_files(test_dir):
  # first collect all file names in the test directory
  files_list = []
  for root, dirs, files in os.walk(test_dir):
    for file1 in files:
      files_list.append(file1)
  return(files_list)
    
'''
 It uses a saved NER model to annotate new text.
 Input -
   nlp: saved (trained) model (can be loaded from a dir: nlp = spacy.load(saved_model_dir))
   test_dir: the path of the directory where the text exists (the text may be an abstract, a section of a DL paper, or the entire paper)
   output_dir: the path of the directory where the entities will be saved
 Output -
   The new entities predicted by the NER model for each input text file are stored in individual files in the output_dir directory
'''

def test_ner_model_df(nlp, test_dir, output_dir, out_tag='_ents'):
  # next go through all files in the test directory, read the text they contain,
  # and predict the entities that exist in them.
  test_files = os.scandir(test_dir)
  for (file_counter, test_file) in enumerate(test_files):
    curr_file = os.path.join(test_file)
    with open(curr_file, "r+", encoding="utf8") as test_file:
      test_text = test_file.read()
      doc = nlp(test_text)

      file_name = test_file.replace('.txt', '')
      ents_file = os.path.join(output_dir, file_name + out_tag + ".txt")
      with open(ents_file, "w+", encoding="utf8") as ef:
        for ent in doc.ents:
          ef.write("%s %s %d %d\n" % (ent.label_.encode("utf-8"), ent.text.encode("utf-8"), ent.start_char, ent.end_char))



def test_ner_model(nlp, test_dir, output_dir, out_tag='_ents'):
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
      doc = nlp(test_text)

      file_name = str(files_list[file_counter])
      file_name = file_name.replace('.txt', '')
      ents_file = os.path.join(output_dir, file_name + "_ents" + ".txt")
      with open(ents_file, "w+", encoding="utf8") as ef:
        for ent in doc.ents:
          ef.write("%s %s %d %d\n" % (ent.label_.encode("utf-8"), ent.text.encode("utf-8"), ent.start_char, ent.end_char))

      file_counter =+ 1
