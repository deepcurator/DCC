from selenium import webdriver
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.by import By
from selenium.common.exceptions import TimeoutException
from bs4 import BeautifulSoup
import os
import time
import wget



def write_metadata(path, meta_file, data):
    with open(path + "/" + meta_file + ".txt", 'w') as myfile:
        myfile.write(data)


def fetch_paper(paper_link, path):
    decomposed = paper_link.split('/')
    if 'arxiv' in decomposed[2]:
        decomposed[3] = 'pdf'
        decomposed[4] += '.pdf'
        reconstructed = decomposed[0] + "//" + decomposed[2] + "/" + decomposed[3] + "/" + decomposed[4]
        filename = wget.download(reconstructed, out=path)
        #print(reconstructed)
    else:
        print("Don't know how to fetch %s" % paper_link)


def fetch_code(code_link, path):
    decomposed = code_link.split('/')
    assert len(decomposed) == 5
    if 'github' in decomposed[2]:
        decomposed.append('archive/master.zip')
        reconstructed = decomposed[0] + "//" + decomposed[2] + "/" + decomposed[3] + "/" + decomposed[4] + "/" + decomposed[5]
        filename = wget.download(reconstructed, out=path)
        #print(reconstructed)
    else:
        print("Don't know how to fetch %s" % paper_link)


def fetch_metadata(browser, url):
    browser.get(url)
    delay = 2
    try:
        myElem = WebDriverWait(browser, delay).until(EC.presence_of_element_located((By.ID, 'div')))
        pass
    except TimeoutException:
        pass
    html_source = browser.page_source
    soup = BeautifulSoup(html_source, "html5lib")
    paper_list = soup.findAll('div', {'class':'container list-container'})
    print(type(paper_list), len(paper_list))



    for paper_num, paper in enumerate(paper_list):
        # create directory
        paper_directory = "./data/" + str(paper_num)
        os.makedirs(paper_directory)

        # Process paper
        title = paper.find('h5')
        write_metadata(paper_directory, "title", title.text)
        #print(title.text)

        abstract = paper.find('small')
        write_metadata(paper_directory, "abstract", abstract.text)
        #print(abstract.text)

        stars = paper.find('div', {'class':'stars'})
        mystar = stars.text.strip().split('\n')[0]
        write_metadata(paper_directory, "stars", mystar)
        #print(mystar)

        date = paper.find('div', {'class':'stars-accumulated text-center'})
        write_metadata(paper_directory, "date", date.text.strip())
        #print(date.text.strip())

        paper_link = paper.find('a', {'class':'btn btn-primary'})
        write_metadata(paper_directory, "paper", paper_link['href'])
        #print(paper_link['href'])

        code_link = paper.find('a', {'class':'btn btn-success'})
        write_metadata(paper_directory, "code", code_link['href'])
        #print(code_link['href'])


        metadata = paper.find('div', {'class':'metadata'})
        mydatasets = []
        if metadata:
            datasets_banner = metadata.find('b')
            if "Datasets" in datasets_banner.text:
                dsets = metadata.findAll('a')
                for d in range(len(dsets)):
                    mydatasets.append(dsets[d].text)
                    #print(dsets[d].text, dsets[d]['href'])
            else:
                # Conference
                mydatasets = None
                #print('None')
        else:
            #print('None')
            mydatasets = None

        if isinstance(mydatasets, list): 
            write_metadata(paper_directory, "datasets", ''.join(mydatasets))
        else:
            write_metadata(paper_directory, "datasets", 'None')


        framework = paper.find('div', {'class':'framework-img'})
        if framework:
            img = framework.find('img')
            if img:
                myframework = img['src'].split('/')[2].split('.')[0]
                write_metadata(paper_directory, "framework", myframework)
                #print(myframework)
            else:
                write_metadata(paper_directory, "framework", 'None')
        else:
            write_metadata(paper_directory, "framework", 'None')


        # Fetch paper from arxiv
        pdf = fetch_paper(paper_link['href'], paper_directory)

        # Fetch zip file from github
        zipfile = fetch_code(code_link['href'], paper_directory)
        

        print('- paper %s -' % paper_directory)


if __name__ == "__main__":
    browser = webdriver.Chrome('./chromedriver')
    fetch_metadata(browser, "https://paperswithcode.com/latest")

