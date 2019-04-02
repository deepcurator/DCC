import json
import os
import urllib.request
from bs4 import BeautifulSoup

def has(key, obj):
    if key in obj:
        return True
    else:
        return False

def getKey(item):
    return item[0]



def fetch_abstract(url):
    contents = urllib.request.urlopen(url).read()
    soup = BeautifulSoup(contents, features='html.parser')

    if 'arxiv.org' in url:
        abstract = soup.find('blockquote', {'class': 'abstract mathjax'})
        return abstract.text.strip().replace('Abstract:  ', '')
    elif 'papers.nips.cc' in url:
        abstract = soup.find('p', {'class':'abstract'})
        return abstract.text.strip()
    elif 'uni-saarland.de' in url:
        abstract = soup.find('div', {'class':'card-body acl-abstract'})
        if abstract:
            return abstract.text.strip().replace('Abstract', '')
    return None

if __name__ == "__main__":
    with open('./evaluation-tables.json', 'r') as f:
        myjson = json.load(f)

    try:
        os.mkdir("./data")
    except:
        pass

    # Json data is organized in this hierarhcy task->datasets->sota->sota_rows

    paper_seen = set()
    papers = []

    for i in myjson:
        category = i['categories']
        task = i['task']

        for dataset in i['datasets']:
            if has('sota', dataset):
                #print(dataset['sota']['sota_rows'])
                sota = dataset['sota']
                for sota_row in sota['sota_rows']:
                    if sota_row['paper_url'] not in paper_seen:
                        paper_seen.add(sota_row['paper_url'])
                        papers.append((category[0], sota_row['paper_url']))


    for paper_id, paper in enumerate(sorted(papers, key=getKey)):
        abstract = fetch_abstract(paper[1])
        if abstract:
            # Write abstract
            with open('./data/'+str(paper_id)+'.txt', 'w') as f:
                f.write(abstract)
            # Write meta-data
            with open('./data/'+str(paper_id)+'.cat', 'w') as f:
                f.write(paper[0])
            with open('./data/'+str(paper_id)+'.url', 'w') as f:
                f.write(paper[1])
            with open('./data/toc.csv', 'a') as f:
                f.write('%s, %s, %s\n' % (paper_id, paper[0], paper[1]))
            print('%s, %s, %s' % (paper_id, paper[0], paper[1]))
