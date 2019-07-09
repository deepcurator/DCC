from selenium import webdriver
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.by import By
from selenium.common.exceptions import TimeoutException
from bs4 import BeautifulSoup
from pathlib import Path
import os
import time
import wget


def write_metadata(path, meta_file, data):
    filename = meta_file+".txt"
    file_path = Path(path) / filename
    with open(file_path, 'w') as myfile:
        myfile.write(data)


def fetch_paper(paper_link, path):
    if 'arxiv' in paper_link and '.pdf' in paper_link:
        filename = wget.download(paper_link, out=str(path))
        # print(reconstructed)
    else:
        print("Don't know how to fetch %s" % paper_link)


def fetch_code(code_link, path):
    decomposed = code_link.split('/')
    assert len(decomposed) == 5
    if 'github' in decomposed[2]:
        decomposed.append('archive/master.zip')
        reconstructed = decomposed[0] + "//" + decomposed[2] + "/" + \
            decomposed[3] + "/" + decomposed[4] + "/" + decomposed[5]
        filename = wget.download(reconstructed, out=str(path))
        # print(reconstructed)
    else:
        print("Don't know how to fetch %s" % code_link)


def fetch_metadata(path_to_chromedriver, url, limit: int = -1, save_directory=Path("./data").resolve(), condition: dict={}) -> bool:
    chrome_options = webdriver.ChromeOptions()
    chrome_options.add_argument('--headless')
    chrome_options.add_argument('--no-sandbox')
    browser = webdriver.Chrome(str(path_to_chromedriver), options=chrome_options)
    browser.get(url)
    delay = 2
    try:
        myElem = WebDriverWait(browser, delay).until(
            EC.presence_of_element_located((By.ID, 'div')))
        pass
    except TimeoutException:
        pass
    html_source = browser.page_source
    soup = BeautifulSoup(html_source, "lxml")
    paper_list = soup.find_all('div', {'class': 'col-lg-9 item-col'})
    
    limit = len(paper_list) if (limit == -1) else limit

    print("Retrieving %d out of %d papers..." % (limit, len(paper_list)))

    for paper_num, paper in enumerate(paper_list):
        
        if paper_num == limit:
            break
        try:
            stop = fetch_one(paper_num, paper, browser, delay, save_directory, condition)
            if stop:
                return stop
        except Exception as e:
            print(e)
            continue
    
    browser.close()
    return False

def fetch_one(paper_num, paper, browser, delay, save_directory, condition) -> bool:
    stop = False
    # Process paper
    title = paper.find('h1')
    # print(title.text)
    abstract = paper.find('p', {'class': 'item-strip-abstract'})
    # print(abstract.text)
    stars = paper.find('div', {'class': 'entity-stars'})
    mystar = stars.text.strip().split('\n')[0]
    # print(mystar)
    date = paper.find('div', {'class': 'stars-accumulated text-center'})
    # print(date.text.strip())
    tags = paper.findAll('span', {'class': 'badge badge-primary'})
    # print(tags[0].string)

    # url where paper and code links are located
    links_url = paper.find('a')
    links_url = links_url['href']

    dir_name = links_url.split('/')[-1]
    print("\n%d. %s" % (paper_num+1, dir_name))
    
    # create directory
    paper_directory = save_directory / str(dir_name)

    if os.path.isdir(paper_directory):
        print("\tAlready downloaded.")
        return stop
    os.makedirs(paper_directory, exist_ok=True)

    paper_link = None
    code_link = None
    myframework = None
    if "http" not in links_url:
        browser.get("https://paperswithcode.com"+links_url)
        try:
            myElem = WebDriverWait(browser, delay).until(
                EC.presence_of_element_located((By.ID, 'div')))
            pass
        except TimeoutException:
            pass
        links_html_source = browser.page_source
        links_soup = BeautifulSoup(links_html_source, "lxml")

        paper_link = links_soup.find('a', {'class': 'badge badge-light'})
        # print(paper_link['href'])
        code_link = links_soup.find('a', {'class': 'code-table-link'})
        # print(code_link['href'])

        framework = links_soup.find('div', {'class': 'col-md-2'})
        if framework:
            img = framework.find('img')
            if img:
                myframework = img['src'].split('/')[3].split('.')[0]

    if title:
        write_metadata(paper_directory, "title", title.text)
    if abstract:
        write_metadata(paper_directory, "abstract", abstract.text)
    if mystar:
        write_metadata(paper_directory, "stars", mystar)
    if date:
        date_text = date.string.strip()
        print(date_text)
        if condition:
            if "year" in condition:
                if date_text.split(" ")[-1] == condition['year']:
                    stop = True
        write_metadata(paper_directory, "date", date_text)
    if tags:
        # comma seperated tags
        cs_tags = ','.join(x.string for x in tags)
        write_metadata(paper_directory, "tags", cs_tags)
    else:
        write_metadata(paper_directory, "tags", "None")

    # TODO: currently grabbing dataset not working, have to use queries
    metadata = paper.find('div', {'class': 'metadata'})
    mydatasets = []
    if metadata:
        datasets_banner = metadata.find('b')
        if "Datasets" in datasets_banner.text:
            dsets = metadata.findAll('a')
            for d in range(len(dsets)):
                mydatasets.append(dsets[d].text)
                # print(dsets[d].text, dsets[d]['href'])
        else:
            # Conference
            mydatasets = None
            # print('None')
    else:
        # print('None')
        mydatasets = None

    if isinstance(mydatasets, list):
        write_metadata(paper_directory, "datasets", ''.join(mydatasets))
    else:
        write_metadata(paper_directory, "datasets", 'None')

    if paper_link:
        # Fetch paper from arxiv
        fetch_paper(paper_link['href'], paper_directory)
        write_metadata(paper_directory, "paper", paper_link['href'])
    
    if "github" in links_url:
        # get framework from original soup
        framework = paper.find('img')
        if framework:
            myframework = framework['src'].split('/')[3].split('.')[0]
        
        fetch_code(links_url, paper_directory)
        write_metadata(paper_directory, "code", links_url)

    elif code_link:
        # Fetch zip file from github
        fetch_code(code_link['href'], paper_directory)
        write_metadata(paper_directory, "code", code_link['href'])

    if myframework:
        write_metadata(paper_directory, "framework", myframework)
    else:
        write_metadata(paper_directory, "framework", 'None')

    # print('- paper %s -' % paper_directory)
    return stop

if __name__ == "__main__":
    fetch_metadata('./chromedriver', "https://paperswithcode.com/latest", 1)
