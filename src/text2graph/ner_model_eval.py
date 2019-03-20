
# function that produces various scores of the trained NER model
# Input: the trained NER model,
#        a set of examples with annotations incorporated within;
#           the structure is as follows:
#           [('sentence01', [(start01, end01, 'entity type 01')]),
#            ('sentence02', [(start02-1, end02-1, 'entity type 02-1'),
#                            (start02-2, end02-2, 'entity type 02-2')])
#           ]
# Output: various evaluation scores, including precision, f1, ...


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

