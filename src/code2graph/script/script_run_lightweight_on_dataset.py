from pathlib import Path
import os
from zipfile import ZipFile
from argparse import Namespace
import shutil
import glob
import csv

import sys
sys.path.append('../')

from core.graphlightweight import TFTokenExplorer
from config.config import LightWeightMethodConfig

cols = ['Title','Framework','Lightweight','Error Msg','Date','Tags','Stars','Code Link','Paper Link']
metas = ['title', 'framework', 'date', 'tags', 'stars', 'code', 'paper']

def process(data_path: Path, stats_path: Path, options: list):
    if not Path(stats_path).exists():
        with open(stats_path, 'w') as file:
            writer = csv.writer(file)
            writer.writerow(cols)

    subdirs = [x for x in data_path.iterdir() if x.is_dir()]
    # import pdb; pdb.set_trace()

    dataset = [] 

    for subdir in subdirs:
        
        repo = {}
        for meta_prefix in metas:
            try:
                with open(str(subdir / (meta_prefix + '.txt')), 'r') as f:
                    if meta_prefix == 'title':
                        repo[meta_prefix] = ' '.join([line.strip() for line in f.readlines()])
                    else:
                        repo[meta_prefix] = f.read().strip()
            except:
                repo[meta_prefix] = ""
        repo['zip_path'] = list(subdir.glob('*.zip'))[0]
        dataset.append(repo)

        # import pdb; pdb.set_trace()

    for repo in dataset:
        title = repo['title']
        date = repo['date']
        framework = repo['framework']
        tags = repo['tags']
        stars = repo['stars']
        paper_link = repo['paper']
        code_link = repo['code']
        success = "N/A"
        error_msg = "N/A"
    
        if 'tf' in framework:
            extract_name = (repo['zip_path'].name).split('.')[0]
            extract_path = repo['zip_path'].parent / extract_name
            # remove directory if it already exists
            if extract_path.exists:
                shutil.rmtree(extract_path)
            # unzip file
            with ZipFile(repo['zip_path'], "r") as zip_ref:
                zip_ref.extractall(extract_path)
            
            args = Namespace(code_path=extract_path, output_types=options, show_arg=True, show_url=True)
            config = LightWeightMethodConfig(args)
            try:
                explorer = TFTokenExplorer(config)
                explorer.explore_code_repository()
                success = "Success"
            except Exception as e:
                print("\t",e)
                success = "Error"
                error_msg = str(e).strip()
                pass

        with open(stats_path, 'a') as file:
            writer = csv.writer(file)
            writer.writerow([title, framework, success, error_msg, date, tags, stars, code_link, paper_link])
                
def move_triples(data_path, dest_path, filetype, name_index=7):
    for path in Path(data_path).rglob(filetype):
        path = Path(path)
        repo_name = str(path).split('/')[name_index]
        repo_path = Path(dest_path) / repo_name
        if not repo_path.is_dir():
            repo_path.mkdir(exist_ok=True)
        shutil.copy(path, repo_path)

    
if __name__ == "__main__":
    data_path = Path("../raw_data_tf/").resolve()
    dest_path = Path("../rdf_triples/").resolve()
    dest_path.mkdir(exist_ok=True)
    rdf_path = Path("../rdf/").resolve()
    stat_file_path = dest_path/"stats.csv"
    process(data_path, stat_file_path, options=[5])
    move_triples(data_path, dest_path, "*.triples")
    move_triples(data_path, rdf_path, "combined_triples.triples")