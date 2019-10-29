# Scientific papers are stored in PDF format. We have used Grobid to extract the text from the PDF files.
# This file extracts the title, abstract and various sections of the papers and stores them in txt files.
# The extracted text is stored in separate files, where each file stores one sentence of the extracted
# text from each paper. This is required because these files will be used as input to the auto_annotate.py
# to genrate annotations that follow the Brat format.

import yaml
import time
import os
from bs4 import BeautifulSoup

# maximum number of section in the paper (in xml format) you want to consider.
# note: the abstract is always the first section that is considered.
# if you want to include only the abstract, then set MAX_NUM_SECTIONS = 1
# if you want to include the abstract, then 1st and 2nd sections, then set MAX_NUM_SECTIONS = 3
# all the text in the abstract, 1st and 2nd sections will be included in one file.
MAX_NUM_SECTIONS = 2


#config = yaml.safe_load(open('../../conf/conf.yaml'))
#file_path = config['PAPERS_IN_XML_PATH']
#output_dir = config['SENTENCE_ANNOTATED_TEXT_PATH']
#output_dir = "C:/Home02/src02/DCCdemoBeta/DCC-development/DCC-development/src/text2graph/Data/XML2TXTresutls/"

class TEIFile(object):
    def __init__(self, filename):
        self.filename = filename
        self.soup = read_tei(filename)
        self._text = None
        self._title = ''
        self._abstract = ''

    @property
    def doi(self):
        idno_elem = self.soup.find('idno', type='DOI')
        if not idno_elem:
            return ''
        else:
            return idno_elem.getText()

    @property
    def title(self):
        if not self._title:
            self._title = self.soup.title.getText()
        return self._title

    @property
    def abstract(self):
        if not self._abstract:
            abstract = self.soup.abstract.getText(separator=' ', strip=True)
            self._abstract = abstract
        return self._abstract

    @property
    def authors(self):
        authors_in_header = self.soup.analytic.find_all('author')

        result = []
        for author in authors_in_header:
            persname = author.persname
            if not persname:
                continue
            firstname = elem_to_text(persname.find("forename", type="first"))
            middlename = elem_to_text(persname.find("forename", type="middle"))
            surname = elem_to_text(persname.surname)
            person = Person(firstname, middlename, surname)
            result.append(person)
        return result

    @property
    def text(self):
        if not self._text:
            divs_text = []
            for div in self.soup.body.find_all("div"):
                # div is neither an appendix nor references, just plain text.
                if not div.get("type"):
                    div_text = div.get_text(separator=' ', strip=True)
                    divs_text.append(div_text)

            plain_text = " ".join(divs_text)
            self._text = plain_text
        return self._text

    @property
    def partial_text(self):
        if not self._text:
            divs_text = []
            #max_sections = 2
            sec_cnt = 0
            div_list = self.soup.body.find_all("div")
            for div in div_list:
                # div is neither an appendix nor references, just plain text.
                if not div.get("type"):
                    div_text = div.get_text(separator=' ', strip=True)
                    divs_text.append(div_text)
                sec_cnt += 1
                if sec_cnt >= MAX_NUM_SECTIONS or sec_cnt > len(div_list)+1:
                    break

            plain_text = " ".join(divs_text)
            self._text = plain_text
        return self._text


def read_tei(tei_file):
    with open(tei_file, 'rb') as tei:
        soup = BeautifulSoup(tei, 'lxml')
        return soup
    raise RuntimeError('cannot generate a soup from the input')



#start_time = time.time()
#
##file_path = 'C:/Home02/src02/DCCdev/grobid/'
#files_xml = [x for x in os.listdir(file_path) if os.path.splitext(x)[1]=='.xml']
#
#
#empty_abstracts = []
#sent_cnt = 0
#for i, f in enumerate(files_xml):
#    print('Processing paper ', i)
#    tei_file = os.path.join(file_path, f)
#    paper = TEIFile(tei_file)
#
#
#    paper_body = paper.partial_text
#    paper_body = paper_body.lower()
#    paper_body = paper_body.replace('et al.', 'et al')
#    new_f = f.replace('.tei', '')
#    new_f = new_f.replace('.xml', '')
#
#
#    outfile = os.path.join(output_dir, new_f + '.txt')
#    with open(outfile, 'w', encoding='utf8') as of:
#        of.write(paper_body)
#
#print("\n Total number of sentences (and files): ", sent_cnt)
#end_time = time.time()
#elapsed_time = end_time - start_time
#print("\n *** Elapsed time: ", elapsed_time)