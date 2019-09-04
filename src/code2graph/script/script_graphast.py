from pathlib import Path
from zipfile import ZipFile
from argparse import Namespace
import shutil
import glob
from dateutil import parser

import sys
sys.path.append('../')

from core.graphast import ASTExplorer
from config.config import GraphASTArgParser, GraphASTConfig

metas = ['title', 'framework', 'date', 'tags', 'stars', 'code', 'paper']

def extract_data(data_path: Path) -> list:

    subdirs = [x for x in data_path.iterdir() if x.is_dir()]
    dataset = []

    for subdir in subdirs:
        repo = {}
        for meta_prefix in metas:
            try:
                with open(str(subdir / (meta_prefix + '.txt')), 'r') as f:
                    if meta_prefix == 'title':
                        repo[meta_prefix] = ' '.join(
                            [line.strip() for line in f.readlines()])
                    else:
                        repo[meta_prefix] = f.read().strip()
                    if repo[meta_prefix] == "" or repo[meta_prefix] == "None":
                        repo[meta_prefix] = None
            except:
                repo[meta_prefix] = None
        if repo['stars']:
            repo['stars'] = int(repo['stars'].replace(",", ""))
        repo['folder_name'] = subdir.name

        if repo['date']:
            try:
                # check if date is valid
                parser.parse(repo['date'])
            except ValueError:
                repo['date'] = None

        repo['code_path'] = None
        if repo['framework'] and 'tf' in repo['framework']:
            try:
                zip_path = list(subdir.glob('*.zip'))[0]
                extract_name = zip_path.name.split('.')[0]
                extract_path = zip_path.parent / extract_name
                # remove directory if it already exists
                if extract_path.exists():
                    shutil.rmtree(extract_path)
                # unzip file
                with ZipFile(zip_path, "r") as zip_ref:
                    zip_ref.extractall(extract_path)
                repo['code_path'] = extract_path
            except:
                pass

        dataset.append(repo)

    return dataset

def recursive(data_path):
   
    dataset = extract_data(data_path)
    tasks = []

    for idx, repo in enumerate(dataset):
        if repo['framework'] and 'tf' in repo['framework']:

            if repo['code_path'] is not None:
                tasks.append(repo['code_path'])

    return tasks

def copy_files(data_path, dest_path, filetype, name_index=-3):
    """ Copy files in data_path that matches filetype to dest_path """
    for path in Path(data_path).rglob(filetype):
        path = Path(path)
        repo_name = str(path).split('/')[name_index]
        repo_path = Path(dest_path) / repo_name
        if not repo_path.is_dir():
            repo_path.mkdir(exist_ok=True)
        shutil.copy(path, repo_path)


def move_output_files(config):
    config.dest_path.mkdir(exist_ok=True)
    copy_files(config.input_path, config.dest_path, "functions.txt")


def run_graphast_method(code_path, resolution):
    try:
        explorer = ASTExplorer(str(code_path), resolution)
        explorer.dump_functions_source_code()
    except Exception as e:
        print(e)


def graphast_pipeline(args):
    config = GraphASTConfig(GraphASTArgParser().get_args(args))
    tasks = []

    if config.recursive:
        tasks = recursive(config.input_path)

    else:
        tasks.append(config.input_path)

    for task in tasks:
        run_graphast_method(task, config.resolution)

    move_output_files(config)

if __name__ == "__main__":
    graphast_pipeline(sys.argv[1:])