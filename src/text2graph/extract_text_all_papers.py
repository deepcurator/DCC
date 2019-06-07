import yaml
import glob
import os
import pandas as pd

from io import StringIO
from pdfminer.pdfinterp import PDFResourceManager, PDFPageInterpreter
from pdfminer.converter import TextConverter
from pdfminer.layout import LAParams
from pdfminer.pdfpage import PDFPage

##########################################################################################
# retrieve a pdf file, and extract text from it. there is an option to retrieve text from
# a certain pages. if pages=[p1,p2] the function will retrieve text from page p1 to p2.
def convert(fname, pages=None):
  if pages is None:
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

#def get_page_count(fname):
#    parser = PDFParser(fname)
#    document = PDFDocument(parser)
#    pn=resolve1(document.catalog['Pages'])['Count'])
#    return(pn)
##########################################################################################

start_page = 0
end_page = None

config = yaml.safe_load(open('../../conf/conf.yaml'))
root_dir = config['RAW_DATA_FOLDER']
out_dir = config['EXTRACT_TEXT_PATH']

if not os.path.exists(out_dir):
    os.makedirs(out_dir)

for (i,filename) in enumerate(glob.iglob(root_dir+'/*/*.pdf', recursive=True)):
    ppr2txt = convert(filename) # pages=[start_page, end_page])
    head, tail = os.path.split(filename)
    print(str(i) +' '+ filename)
    #out = os.path.join(head,tail.replace('.pdf','.txt'))
    out=os.path.join(out_dir,tail.replace('.pdf','.txt'))
    file = open(out,'w', encoding = 'utf8') 
    file.write(ppr2txt) 
    file.close() 


