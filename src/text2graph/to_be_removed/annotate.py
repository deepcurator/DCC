import glob
import pandas as pd
import requests
url = 'http://localhost:2222/rest/annotate'





#root_dir = 'C:\\Home\\src\\darpa_aske_dcc\\src\\paperswithcode\\data\\'
#paperPath = []
#abslist = []
#for i in range(100):
#  text = str(i) + '\\abstract.txt'
#  dir = root_dir + text
#  #print(dir)
#  paperPath.append(dir)
#
#for file in paperPath:
#  abstract = open(file, 'r')
#  abslist.append(abstract)

root_dir = 'C:\\Users\\z003z47y\\Documents\\git\\darpa_aske_dcc\\src\\paperswithcode\\txt\\papers'
textIterator = glob.glob(root_dir + '**/*.txt', recursive=True)
#print(textIterator)

abstracts = []

start  = 'abstract'
end = 'introduction'


for file in textIterator:
   with open(file,encoding='utf8') as f:
#       print("Reading file : " + str(file) + "\n")
       lines = f.readlines()
       papertxt = ''.join(str(line) for line in lines)
       papertxt = papertxt.lower()
       result = papertxt.split(start)[1].split(end)[0]
       abstracts.append(result)
       

results = []

for i, text in enumerate(abstracts):
#  print(i)
  #print(text)
  userdata = {"text": text, "confidence": "0.5", "support": "0"}
  resp = requests.post(url, data=userdata, headers = {'Accept' :'text/xml'})
#  print(resp)
  results.append(resp.text)

# print(z)
# txtIterator = glob.glob(root_dir + '**/*.csv', recursive=True)

for i, result in enumerate(results):
    with open('C:\\Users\\z003z47y\\Documents\\git\\darpa_aske_dcc\\src\\paperswithcode\\txt\\papers\\annotated.txt', "a",encoding='utf8') as f1:
        f1.write('************Paper' + str(i) + '*******************\n' +result + "***********************************\n")
# df = pd.concat([pd.read_csv(f,header=None) for f in glob.glob(root_dir + '**/*.csv', recursive=True)],ignore_index=True)
# files = csvIterator[0]
# print("Reading file " + file)
# print("File name is " + csvIterator[0])
# df = pd.read_csv(csvIterator[0], header=None)
# df.columns = ["Subject", "Predicate", "Object"]
# df.head()
# df['Subject'] = df['Subject'].apply(replace)
# df['Object'] = df['Object'].apply(replace)
# df.head(100)

