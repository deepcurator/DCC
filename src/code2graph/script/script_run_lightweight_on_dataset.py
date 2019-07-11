from pathlib import Path
import os
from zipfile import ZipFile
from argparse import Namespace
import shutil
import glob

import sys
sys.path.append('../')

from core.graphlightweight import TFTokenExplorer
from config.config import LightWeightMethodConfig

original_100_dataset = False

def process(data_path, stats_path):
    if not Path(stats_path).exists():
        with open(stats_path, 'w') as f:
            #TODO: date bug
            f.write("Title,Framework,Lightweight,Error Msg,Date,Tags,Stars,Code Link,Paper Link\n")

    for (dirpath, dirnames, filenames) in os.walk(data_path):
        title = None
        date = " "
        framework = " "
        tags = "None"
        stars = "None"
        paper_link = " "
        code_link = " "
        success = "N/A"
        error_msg = "N/A"
        for filename in filenames:
            if filename == "title.txt":
                title_path = Path(dirpath)/filename
                title_path = title_path.resolve()
                with open(title_path, 'r') as f:
                    title = f.read()
                    if ',' in title:
                        title = '"'+title+'"'
                        title = title.replace('\n', ' ')

            if filename == "date.txt":
                date_path = Path(dirpath)/filename
                date_path = date_path.resolve()
                with open(date_path, 'r') as f:
                    date = f.read().strip()

            if filename == "tags.txt":
                tags_path = Path(dirpath)/filename
                tags_path = tags_path.resolve()
                with open(tags_path, 'r') as file:
                    tags = file.read()
                    if "," in tags:
                        tags = '"'+tags+'"'
            
            if filename == "stars.txt":
                stars_path = Path(dirpath)/filename
                stars_path = stars_path.resolve()
                with open(stars_path, 'r') as file:
                    stars = file.read()
                    if "," in stars:
                        stars = '"'+stars+'"'

            if filename == "paper.txt":
                paper_path = Path(dirpath)/filename
                paper_path = paper_path.resolve()
                with open(paper_path, 'r') as file:
                    paper_link = file.read()

            if filename == "code.txt":
                code_link_path = Path(dirpath)/filename
                code_link_path = code_link_path.resolve()
                with open(code_link_path, 'r') as file:
                    code_link = file.read()

            if filename.endswith('.zip'):
                is_tf = False
                try:
                    framework_path = Path(dirpath)/"framework.txt"
                    framework_path = framework_path.resolve()
                    with open(framework_path, 'r') as file:
                        framework = file.read()
                        if 'tf' in framework:
                            is_tf = True
                except FileNotFoundError:
                    print("\tframework.txt file not found.")
                    continue
        
                if is_tf:
                    ext_dir_name = filename.split(".")[0]
                    if not original_100_dataset:
                        zip_dir = Path(dirpath)/filename
                        zip_dir = zip_dir.resolve()
                        ext_dir = Path(dirpath)/ext_dir_name
                        ext_dir = ext_dir.resolve()
                        # remove directory if it already exists
                        if os.path.exists(ext_dir):
                            shutil.rmtree(ext_dir)
                        with ZipFile(zip_dir, "r") as zip_ref:
                            zip_ref.extractall(ext_dir)
                    code_path = Path(dirpath)/ext_dir_name/ext_dir_name
                    code_path = code_path.resolve()
                    # print(glob.glob("%s/**/*.py" % str(code_path), recursive=True))
                    args = Namespace(code_path=code_path, output_types=[5], show_arg=True, show_url=True)
                    config = LightWeightMethodConfig(args)
                    try:
                        explorer = TFTokenExplorer(config)
                        explorer.explore_code_repository()
                        success = "Success"
                    except Exception as e:
                        print("\t",e)
                        success = "Error"
                        error_msg = str(e).replace(",",";")
                        pass

        if title:
            with open(stats_path, 'a') as f:
                stats = ','.join([title, framework, success, error_msg, date, tags, stars, code_link, paper_link])
                f.write(stats+"\n")
                

def move_triples(data_path, dest_path):
    name_index = 5
    if original_100_dataset:
        name_index = 9
    for path in Path(data_path).rglob("*.triples"):
        repo_name = str(path).split('/')[name_index]
        repo_path = Path(dest_path) / repo_name
        if not repo_path.is_dir():
            repo_path.mkdir(exist_ok=True)
        shutil.copy(path, repo_path)

    
if __name__ == "__main__":
    data_path = Path("../data_tf/")
    dest_path = Path("../rdf_triples/").resolve()
    stat_file_path = dest_path/"stats.csv"
    process(data_path, stat_file_path)
    Path(dest_path).mkdir(exist_ok=True)
    move_triples(data_path, dest_path)