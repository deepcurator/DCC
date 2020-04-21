from pathlib import Path
from zipfile import ZipFile
from argparse import Namespace
from compileall import compile_dir
import shutil
import glob
import csv
import subprocess
import traceback
from dateutil import parser

# import sys
# sys.path.append('../')

import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

from core.graphlightweight import TFTokenExplorer
from config.config import LightWeightMethodArgParser, LightWeightMethodConfig
from core.database import Database

cols = ['Folder Name', 'Title', 'Framework', 'Lightweight',
        'Error Msg', 'Date', 'Tags', 'Stars', 'Code Link', 'Paper Link']
metas = ['title', 'framework', 'date', 'tags', 'stars', 'code', 'paper']


def extract_data(data_path: Path) -> list:
    """ Preprocessing of raw data to make it ready for lightweight graph construction.
        The raw data crawled from the PWCscraper consist of zip file and some metadata stored in text format.

    Creates a dictionary with metadata for each paper and extracts the zip file.

    Arguments:
        data_path {Path} -- Path to directory with a collection of repositories with metadata files.

    Returns:
        list -- A list of dictonaries with paper metadata.
    """

    subdirs = [x for x in data_path.iterdir() if x.is_dir()]

    dataset = []

    for subdir in subdirs:

        repo = fetch_meta_info(subdir)

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


def fetch_meta_info(subdir: Path) -> dict:
    """Fetches meta information of the code repository.

    Arguments:
        subdir {pathlib.Path} -- Path to code repository.
    """

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
    
    repo['stored_dir_name'] = subdir.name

    if repo['date']:
        try:
            # check if date is valid
            parser.parse(repo['date'])
        except ValueError:
            repo['date'] = None
    
    return repo
        

def preprocess(code_path: str):
    """Preprocess a code repository by checking if the code is python3 compatible.
    If the code is not python3 compatible, run 2to3 and autopep8 to fix it.

    Arguments:
        code_path {str} -- [Path to code repository]
    """
    code_path = str(code_path)
    # check if the code is python3 compatible by compiling it
    result = compile_dir(code_path, force=True)

    if result is False:
        subprocess.run("2to3 -w -n %s" % code_path, shell=True)
        subprocess.run("autopep8 --in-place -r %s" % code_path, shell=True)


def recursive(data_path: Path, config: LightWeightMethodConfig) -> tuple:
    """Process all papers in data_path.
    Extract metadata and zip file by calling extract_data.
    Run lightweight method on all tensorflow papers.
    Return a tasks list with code_paths to process and metadata list with results.

    Arguments:
        data_path {Path} -- Path to directory with a collection of repositories with metadata files.
        stats_path {Path} -- Path to stats.csv file.
        config {LightWeightMethodConfig} -- Config for running Lightweight method.

    Returns:
        tuple -- A tuple of tasks and metadata
    """
    metadata = []
    tasks = []
    metadata.append(cols)
    dataset = extract_data(data_path)

    for idx, repo in enumerate(dataset):
        success = 'N/A'
        error_msg = 'N/A'
        if repo['framework'] and 'tf' in repo['framework']:

            if repo['code_path'] is not None:
                task = {'code_path': repo['code_path'], 'meta': repo, 'id': idx+1}
                tasks.append(task)
            else:
                success = "Error"
                error_msg = "There is no zip file."

        metadata.append([repo['stored_dir_name'], repo['title'], repo['framework'], success, error_msg,
                         repo['date'], repo['tags'], repo['stars'], repo['code'],
                         repo['paper']])

    return tasks, metadata


def run_lightweight_method(code_path: Path, meta: dict, config: LightWeightMethodConfig) -> tuple:
    """Runs lightweight method.
    Capture exception and return the error message.

    Arguments:
        code_path {Path} -- Path to code repository.
        config {LightWeightMethodConfig} -- Config for Lightweight method.

    Returns:
        tuple -- success: Ouput status, error_msg: Exception raised by Lightweight method.
    """

    try:
        explorer = TFTokenExplorer(code_path, config, meta)
        explorer.explore_code_repository()
        success = "Success"
        error_msg = "N/A"

    except:
        success = "Error"
        exc_type, exc_value, exc_traceback = sys.exc_info()
        error_msg = traceback.format_exception(
            exc_type, exc_value, exc_traceback)
        error_msg = ''.join(error_msg[-4:])
        print(error_msg)

    return (success, error_msg)


def move_output_files(config: LightWeightMethodConfig):
    """Move output files generated by lightweight method to destination path."""
    if 5 in config.output_types:
        if config.combined_triples_only:
            copy_files(config.input_path, config.dest_path,
                       "combined_triples.triples")
            copy_files(config.input_path, config.dest_path,
                       "combined_quads.quads")
        else:
            copy_files(config.input_path, config.dest_path, "*.triples")

    if 3 in config.output_types:
        copy_files(config.input_path, config.dest_path, "*.html")
    elif 1 in config.output_types:
        copy_files(config.input_path, config.dest_path, "call_graph.html")

    if 6 in config.output_types:
        copy_files(config.input_path, config.dest_path, "*.rdf")


def copy_files(data_path, dest_path, filetype, name_index=2):
    """ Copy files in data_path that matches filetype to dest_path """
    for path in Path(data_path).rglob(filetype):
        path = Path(path)
        code_dir_path = path
        for _ in range(name_index):
            code_dir_path = code_dir_path.parent

        repo_path = Path(dest_path) / code_dir_path.name
        if not repo_path.is_dir():
            repo_path.mkdir(exist_ok=True)
        shutil.copy(path, repo_path)


def save_metadata(metadata: list, stat_file_path: str):
    with open(str(stat_file_path), 'w') as file:
        writer = csv.writer(file)
        writer.writerows(metadata)


def export_data(metadata: list, tasks: list, config: LightWeightMethodConfig):

    if config.recursive:
        config.dest_path.mkdir(exist_ok=True)
        move_output_files(config)
        database = Database()
        for task in tasks:
            metadata[task['id']][3] = task['success']
            metadata[task['id']][4] = task['err_msg']
            paper = metadata[task['id']]
            try:
                database.update_query(paper[0], paper[3], paper[4])
            except Exception as e:
                continue
        save_metadata(metadata, str(config.dest_path / "stats.csv"))


def pipeline_the_lightweight_approach(args):

    config = LightWeightMethodConfig(
        LightWeightMethodArgParser().get_args(args))
    tasks = []
    metadata = []
    if config.recursive:
        tasks, metadata = recursive(config.input_path, config)

    else:
        task = {'code_path': config.input_path, 'meta': fetch_meta_info(config.input_path.parent)}
        tasks.append(task)
    
    for task in tasks:
        preprocess(task['code_path'])
        task['success'], task['err_msg'] = run_lightweight_method(
            task['code_path'], task['meta'], config)

    export_data(metadata, tasks, config)


if __name__ == "__main__":

    # append argument -h to see more options
    # ex1: python script_lightweight.py -ip ../test/fashion_mnist
    # ex2: python script_lightweight.py -ip ../test/VGG16 -opt 3 (get RDF graphs)
    # ex3: python script_lightweight.py -ip ../test/Alexnet -opt 2 (get call trees)
    # ex4: python script_lightweight.py -ip ../test/Xception
    # ex5: python script_lightweight.py -ip ../test/NeuralStyle
    # ex6: python script_lightweight.py -ip=../raw_data_tf -r -dp=../rdf -opt=5
    # ex7: python script_lightweight.py -ip=../raw_data_tf -r -dp=../rdf -opt=3 --arg --url

    pipeline_the_lightweight_approach(sys.argv[1:])
