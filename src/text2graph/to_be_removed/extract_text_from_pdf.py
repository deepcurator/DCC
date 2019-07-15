
#############################################
## Extract text from PDF files.
## The 3rd party software PDFMiner is used.
#############################################

from io import StringIO
from pdfminer.pdfinterp import PDFResourceManager, PDFPageInterpreter
from pdfminer.converter import TextConverter
from pdfminer.layout import LAParams
from pdfminer.pdfpage import PDFPage

import spacy
nlp = spacy.load("en")


##########################################################################################
# retrieve a pdf file, and extract text from it. there is an option to retrieve text from
# a certain pages. if pages=[p1,p2] the function will retrieve text from page p1 to p2.
def convert(fname, pages=None):
  if not pages:
    pagenums = set()
  else:
    pagenums = set(pages)

  output = StringIO()
  manager = PDFResourceManager()
  converter = TextConverter(manager, output, laparams=LAParams())
  interpreter = PDFPageInterpreter(manager, converter)

  infile = open(fname, 'rb')
  for page in PDFPage.get_pages(infile, pagenums):
    interpreter.process_page(page)
  infile.close()
  converter.close()
  text = output.getvalue()
  output.close
  return text



# define the path where the pdf lives.
#paper_path = 'C:\\Home\\src\\darpa_aske_dcc\\src\\paperswithcode\\data\\0\\1810.13409v1.pdf'
paper_path = 'C:\\Home\\src\\darpa_aske_dcc\\src\\paperswithcode\\NatureDeepReview.pdf'

# define the start and end pages to be extracted
start_page = 0
end_page = 0
ppr2txt = convert(paper_path, pages=[start_page, end_page])
print(ppr2txt)
print('\n==================\n')
print(type(ppr2txt))





#############################################
# use spacy to extract info from a pdf paper

# assign the paper as str (in text) to the nlp model for processing
doc = nlp(ppr2txt)

for token in doc:
  print("{0}\t{1}\t{2}\t{3}\t{4}\t{5}\t{6}\t{7}".format(
        token.text,
        token.idx,
        token.lemma_,
        token.is_punct,
        token.is_space,
        token.shape_,
        token.pos_,
        token.tag_))

### sentence detection
for sent in doc.sents:
  print(sent)

### POS tagging
pos_list = [(token.text, token.tag_) for token in doc]
print(pos_list)


### NER
for ent in doc.ents:
  print(ent.text, ent.label_)

# we can also view the IOB style tagging of the sentences
from nltk.chunk import conlltags2tree
iob_tagged = [
  (
    token.text,
    token.tag_,
    "{0}-{1}".format(token.ent_iob_, token.ent_type_) if token.ent_iob_ != '0' else token.ent_iob_
  ) for token in doc
]
print(iob_tagged)

# in nltk.Tree format
#print(conlltags2tree(iob_tagged))

for ent in doc.ents:
  print(ent.text, ent.label_)

for chunk in doc.noun_chunks:
    print(chunk.text, chunk.label_, chunk.root.text)

### dependency parsing
for token in doc:
  print("{0}/{1} <--{2}-- {3}/{4}".format(
        token.text, token.tag_, token.dep_, token.head.text, token.head.tag_))
