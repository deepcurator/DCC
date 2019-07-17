from pathlib import Path
from zipfile import ZipFile
from argparse import Namespace
import shutil
import glob
import csv

import sys
sys.path.append('../')

from core.graphlightweight import TFTokenExplorer
from config.config import LightWeightMethodArgParser, LightWeightMethodConfig

cols = ['Title','Framework','Lightweight','Error Msg','Date','Tags','Stars','Code Link','Paper Link']
metas = ['title', 'framework', 'date', 'tags', 'stars', 'code', 'paper']

def process(data_path: Path, stats_path: Path, options: list):
    
    with open(stats_path, 'w') as file:
        writer = csv.writer(file)
        writer.writerow(cols)

    subdirs = [x for x in data_path.iterdir() if x.is_dir()]

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

    for repo in dataset:
        success = "N/A"
        error_msg = "N/A"
    
        if 'tf' in repo['framework']:
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
            writer.writerow([repo['title'], repo['framework'], success, error_msg, 
                             repo['date'], repo['tags'], repo['stars'], repo['code_link'], 
                             repo['paper_link']])
                
def move_files(data_path, dest_path, filetype, name_index=7):
    for path in Path(data_path).rglob(filetype):
        path = Path(path)
        repo_name = str(path).split('/')[name_index]
        repo_path = Path(dest_path) / repo_name
        if not repo_path.is_dir():
            repo_path.mkdir(exist_ok=True)
        shutil.copy(path, repo_path)

def run_lightweight_method(args):
    config = LightWeightMethodConfig(LightWeightMethodArgParser().get_args(args))
    
    if args.is_dataset:
        config.dest_path.mkdir(exist_ok=True)
        stat_file_path = config.dest_path / "stats.csv"
        if 5 in config.output_types:
            process(config.code_path, stat_file_path, options=[5])
            if config.combined_triples_only:
                move_files(config.code_path, config.dest_path, "combined_triples.triples")
            else:
                move_files(config.code_path, config.dest_path, "*.triples")
        if 3 in config.output_types:
            process(config.code_path, stat_file_path, options=[3])
            move_files(config.code_path, config.dest_path, "*.html")
    else:
        explorer = TFTokenExplorer(config)
        explorer.explore_code_repository()


if __name__ == "__main__":

    # append argument -h to see more options
    # ex1: python script_run_lightweight_method.py -ipt ../test/fashion_mnist
    # ex2: python script_run_lightweight_method.py -ipt ../test/VGG16 -opt 3 (get RDF graphs)
    # ex3: python script_run_lightweight_method.py -ipt ../test/Alexnet -opt 2 (get call trees)
    # ex4: python script_run_lightweight_method.py -ipt ../test/Xception
    # ex5: python script_run_lightweight_method.py -ipt ../test/NeuralStyle
    # ex6: python script_run_lightweight_method.py -ipt=../raw_data_tf --ds -dp=../rdf -opt=5
    # ex6: python script_run_lightweight_method.py -ipt=../raw_data_tf --ds -dp=../rdf -opt=3 --arg --url
    run_lightweight_method(sys.argv[1:])
