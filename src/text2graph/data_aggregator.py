import pandas as pd 
from bs4 import BeautifulSoup
import urllib.request
from urllib.parse import urljoin
from pathlib import Path
import requests
import os

print(os.getcwd())

def getpdflink(urllink):
    # print(urllink)
    soup = BeautifulSoup(urllib.request.urlopen(urllink),'html.parser')
    # print(soup.find_all('a'))
    links = soup.find_all("a")
    for link in links:
        fullurl = link.get("href")
        fullurl = str(fullurl)
        if(fullurl.endswith('.pdf')):
            # print(urljoin(str(urllink),fullurl))
            return(urljoin(str(urllink),fullurl))
    


def downloadpdf(index, df):
    link  = df['paper_link']
    # print("AT index " + str(index))
    downloadLink = link
    if("arxiv" in link):
        downloadLink = "http://arxiv.org/pdf/" + link.split('/')[-1] + ".pdf"
        # print("http://arxiv.org/pdf/" + link.split('/')[-1] + ".pdf")
    elif (not link.endswith('.pdf') and (not "arxiv" in link)):
        downloadLink = getpdflink(str(link))
    
            # getpdflink(row['paper_link'])
    
    folderpath = str(df['conference']) + "/" + str(df['year']) + "/"
    filename = os.path.join(os.getcwd(),'pdf/' + folderpath + downloadLink.split('/')[-1])
    # print("Link for paper {} is {}".format(index,downloadLink))
    # print("File will be sorted into : " + filename)

    # filename = os.path.join(filename)
    with open(filename,"wb") as file:
        response = requests.get(str(downloadLink), stream = True)
        file.write(response.content)
        print("Downloaded {} into {}".format(downloadLink, filename))
    print("success!")

df = pd.read_csv('../pwc_edited_plt.csv')
df = df.drop("Framework",axis=1)

list_of_values = ['pytorch','tensorflow']
edited_df = df[(df['Platform'] == 'tensorflow') | (df['Platform'] == 'pytorch')]
# edited_df['fname'] = edited_df['paper_link'].apply(downloadpdf)


for index, row in edited_df.iterrows():
    downloadpdf(index,row)

edited_df.head()
edited = edited_df.reset_index(drop=True)
edited.info()
edited = edited.drop('Unnamed: 0',axis=1)
edited.to_csv('pwc_edited_tensorflow_pytorch.csv')