from selenium import webdriver
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.by import By
from selenium.common.exceptions import TimeoutException
from bs4 import BeautifulSoup
import wget
import smtplib
from pprint import pprint, pformat


class PWCReporter:
    ''' Mail utilities for Paperswithcode service '''
    
    def __init__(self):

        # TODO: Write a script to read those from .cfg file. Do NOT reveal your personal info at any time! 
        self.email_address = "paperswithcode.bot@gmail.com"
        self.password = "N8f4$o36" 
        self.recipients = ["shihyuay@uci.edu"]

    def send_email(self, subject="No Title", body="No Content"):

        recipient = self.recipients

        headers = [
            "From: " + self.email_address,
            "Subject: " + subject,
            "To: " + ", ".join(recipient),
            "MIME-Version: 1.0",
            "Content-Type: text/plain"]
        headers = "\r\n".join(headers)

        message = headers + "\r\n\r\n" + body
        
        try:
            server = smtplib.SMTP("smtp.gmail.com", 587)
            server.ehlo()
            server.starttls()
            server.login(self.email_address, self.password)
            server.sendmail(self.email_address, recipient, message)
            server.close()

            print("\tSent notification email.")
        
        except Exception as e:
            print(e)

            print("\tFailed to send email.")
            


class PWCScraper:
    ''' Main class for paperswithcode service '''

    def __init__(self, config):
        self.config = config 

        self.paperswithcode_base_url = "https://paperswithcode.com"
        self.paperswithcode_url = "https://paperswithcode.com/latest"

        self.path_to_chromedriver = self.config.chrome_driver_path
        self.chrome_options = webdriver.ChromeOptions()
        self.chrome_options.add_argument('--headless')
        self.chrome_options.add_argument('--no-sandbox')

        self.reporter = PWCReporter()

        self.papers = []
        
        self.delay = 2
    
    def scrape_papers_from_index(self, condition: dict = {}):
        '''
            scrape the papers from the index page and related metadata.  
        '''
        self.browser.get(self.paperswithcode_url)        
        try:
            WebDriverWait(self.browser, self.delay).until(EC.presence_of_element_located((By.ID, 'div')))
        except TimeoutException as e:
            print(e)

        soup = BeautifulSoup(self.browser.page_source, "lxml")

        paper_list = soup.find_all('div', {'class': 'col-lg-9 item-col'})

        tot_paper_to_get = self.config.tot_paper_to_scrape_per_shot
        limit = len(paper_list) if (tot_paper_to_get == -1) else tot_paper_to_get

        for paper_num, paper in enumerate(paper_list[:limit]):
            print("============")
            print("Processing the %dth paper (in total %d papers) ..." % (paper_num, limit))
            try:
                paper_dict = {} 
                paper_dict['title']    = paper.find('h1').text.strip()
                paper_dict["abstract"] = paper.find('p', {'class': 'item-strip-abstract'}).text.strip()
                paper_dict["stars"]    = paper.find('div', {'class': 'entity-stars'}).text.strip()
                paper_dict["date"]     = paper.find('div', {'class': 'stars-accumulated text-center'}).text.strip()
                
                
                paper_dict["url"]      = paper.find('a')['href'] # url where paper and code links are located

                paper_dict["stored_dir_name"] = paper_dict["url"].split('/')[-1] # store using the hash tag of this paper.
                paper_dict["stored_dir_path"] = self.config.storage_path / paper_dict["stored_dir_name"]

                paper_dict["tags"]     = []
                tags_list              = paper.findAll('span', {'class': 'badge badge-primary'})

                for tag in tags_list:
                    paper_dict["tags"].append(tag.text.strip())

                pprint(paper_dict)
                self.papers.append(paper_dict)

            except Exception as e:
                print(e)

            print("============")

    def scrape_papers_from_profile(self):
        '''
            scrape the code link and framework and paper link from paper profile page.  
        '''
        for paper_idx, paper in enumerate(self.papers):
            
            if not paper['url'].startswith('http'):
                self.browser.get(self.paperswithcode_base_url+paper['url'])
                
                try:
                    WebDriverWait(self.browser, self.delay).until(EC.presence_of_element_located((By.ID, 'div')))
                except TimeoutException as e:
                    print(e)

                links_html_source = self.browser.page_source
                links_soup = BeautifulSoup(links_html_source, "lxml")
                
                paper['paper_link'] = links_soup.find('a', {'class': 'badge badge-light'})['href']
                paper['code_link']  = links_soup.find('a', {'class': 'code-table-link'})['href'] # might be multiple code_link, what to do? 
                paper['framework'] = None

                # scrape the filename of framework to judge which framework is adopted.
                framework = links_soup.find('div', {'class': 'col-md-2'})
                if framework:
                    img = framework.find('img')
                    if img:
                        paper['framework'] = img['src'].split('/')[3].split('.')[0]
        
    def scrape(self):
        
        self.browser = webdriver.Chrome(self.path_to_chromedriver, options=self.chrome_options)

        self.scrape_papers_from_index()
        self.scrape_papers_from_profile()

        self.browser.close()

        self.store_result()

    def store_result(self):
        
        self.config.storage_path.mkdir(exist_ok=True)

        for paper_index, paper in enumerate(self.papers):
            paper_directory = paper['stored_dir_path'].resolve()
            if paper_directory.resolve().exists():
                print("Already downloaded.")
                continue
            paper_directory.mkdir(exist_ok=True) # create directory
            
            self.write_to_file(paper['title'], paper_directory / "title.txt")
            self.write_to_file(paper['abstract'], paper_directory / "abstract.txt")
            self.write_to_file(paper['stars'], paper_directory / "stars.txt")

            # if date:
            #     date_text = date.string.strip()
            #     # print(date_text)
            #     if condition:
            #         if "year" in condition:
            #             if date_text.split(" ")[-1] == condition['year']:
            #                 stop = True
            #     self.write_metadata_(paper_directory, "date", date_text)
                
            if len(paper['tags']) == 0:
                self.write_to_file("None", paper_directory / "tags.txt")
            else:
                self.write_to_file(','.join(paper['tags']), paper_directory / "tags.txt")
            self.write_to_file(paper['paper_link'], paper_directory / "paper.txt")

            wget.download(paper['paper_link'], out=str(paper['stored_dir_path']/(paper['stored_dir_name']+'.pdf')))

            self.write_to_file(paper['code_link'], paper_directory / "code.txt")
            self.fetch_code(paper['code_link'], paper_directory)

            self.write_to_file(paper['framework'], paper_directory / "framework.txt")

            if 'tf' in paper['framework']: 
                message = ("New TensorFlow paper scraped from Paperswithcode.com.\r\n" + pformat(paper))
                title = "Paperswithcode: New TensorFlow paper:%s!"%paper['title']

                self.reporter.send_email(subject=title, body=message)
    
    def write_to_file(self, data, path):
        with open(str(path), 'w') as myfile:
            myfile.write(data)

    def fetch_code(self, code_link, path):
        decomposed = code_link.split('/')
        assert len(decomposed) == 5
        if 'github' in decomposed[2]:
            decomposed.append('archive/master.zip')
            reconstructed = decomposed[0] + "//" + decomposed[2] + "/" + \
                decomposed[3] + "/" + decomposed[4] + "/" + decomposed[5]
            wget.download(reconstructed, out=str(path))
            # print(reconstructed)
        else:
            print("Don't know how to fetch %s" % code_link)