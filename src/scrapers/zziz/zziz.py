import csv
import wget
import os

# Data source: https://github.com/zziz/pwc/


def write_metadata(path, meta_file, data):
    with open(path + "/" + meta_file + ".txt", 'w') as myfile:
        myfile.write(data)


def fetch_paper(paper_link, path):
    decomposed = paper_link.split('/')
    if 'arxiv' in decomposed[2]:
        decomposed[3] = 'pdf'
        decomposed[4] += '.pdf'
        reconstructed = decomposed[0] + "//" + decomposed[2] + "/" + decomposed[3] + "/" + decomposed[4]

        try:
            filename = wget.download(reconstructed, out=path)
            return filename
        except:
            return None
        #print(reconstructed)
    else:
        print("Don't know how to fetch %s" % paper_link)


def fetch_code(code_link, path):
    decomposed = code_link.split('/')
    assert len(decomposed) == 5
    if 'github' in decomposed[2]:
        decomposed.append('archive/master.zip')
        reconstructed = decomposed[0] + "//" + decomposed[2] + "/" + decomposed[3] + "/" + decomposed[4] + "/" + decomposed[5]

        try:
            filename = wget.download(reconstructed, out=path)
            return filename
        except:
            return None
        #print(reconstructed)
    else:
        print("Don't know how to fetch %s" % paper_link)



if __name__ == "__main__":
    with open('pwc-2018.11.13.csv') as csvfile:
        csv_reader = csv.reader(csvfile, delimiter=',')
        try:
            os.stat("./data")
        except:
            os.makedirs("./data")

        for i, row in enumerate(csv_reader):
            title = row[0]
            paper_link = row[1]
            conference = row[2]
            year = row[3]
            code_link = row[4]

            paper_directory = "./data/" + str(i)

            if 'arxiv' in paper_link and 'github' in code_link:
                if title:
                    write_metadata(paper_directory, "title", title)
                if paper_link:
                    write_metadata(paper_directory, "paper", paper_link)
                if conference:
                    write_metadata(paper_directory, "conference", conference)
                if year:
                    write_metadata(paper_directory, "year", year)
                if code_link:
                    write_metadata(paper_directory, "code", code_link)

                os.makedirs(paper_directory)
                pdf = fetch_paper(paper_link, paper_directory)
                zipfile = fetch_code(code_link, paper_directory)

                if not pdf and not zipfile:
                    print('---- removing %s ----' % (paper_directory))
                    os.remove(paper_directory + "/*")
                    os.rmdir(paper_directory)

                print('- paper %s -' % (paper_directory))
