from pathlib import Path
from zipfile import ZipFile
from argparse import Namespace
from compileall import compile_dir
import shutil
import glob
import csv
import subprocess
import traceback

import sys
sys.path.append('../')

from core.graphlightweight import TFTokenExplorer
from config.config import LightWeightMethodArgParser, LightWeightMethodConfig

cols = ['Title','Framework','Lightweight','Error Msg','Date','Tags','Stars','Code Link','Paper Link']
metas = ['title', 'framework', 'date', 'tags', 'stars', 'code', 'paper']

def extract_data(data_path: Path, stats_path: Path) -> list:
    """ Preprocessing of raw data to make it ready for lightweight graph construction. 
        The raw data crawled from the PWCscraper consist of zip file and some metadata stored in text format. 

    Creates a dictionary with metadata for each paper and extracts the zip file.
    Creates a stats.csv file and write the column headers.
    
    Arguments:
        data_path {Path} -- Path to directory with a collection of repositories with metadata files.
        stats_path {Path} -- Path to stats.csv file.
    
    Returns:
        list -- A list of dictonaries with paper metadata.
    """

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
        
        repo['code_path'] = None
        if 'tf' in repo['framework']:
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

def preprocess(code_path: str):
    """Preprocess a code repository by checking if the code is python3 compatible.
    If the code is not python3 compatible, run 2to3 and autopep8 to fix it.
    
    Arguments:
        code_path {str} -- [Path to code repository]
    """
    code_path = str(code_path)
    # check if the code is python3 compatible by compiling it
    result = compile_dir(code_path, force=True)
    
    if result == False:
        subprocess.run("2to3 -w -n %s" % code_path, shell=True)
        subprocess.run("autopep8 --in-place -r %s" % code_path, shell=True)

def recursive(data_path: Path, stats_path: Path, config: LightWeightMethodConfig) -> list:
    """Process all papers in data_path.
    Extract metadata and zip file by calling extract_data.
    Run lightweight method on all tensorflow papers.
    Saves metadata to stats.csv file.
    
    Arguments:
        data_path {Path} -- Path to directory with a collection of repositories with metadata files.
        stats_path {Path} -- Path to stats.csv file.
        options {list} -- Output options for Lightweight method.
    
    Returns:
        list -- A list of metadata with result of running lightweight method.
    """
    metadata = []
    metadata.append(cols)
    dataset = extract_data(data_path, stats_path)    

    for repo in dataset:
        success = 'N/A'
        error_msg = 'N/A'
        if 'tf' in repo['framework']:
            
            if repo['code_path'] is not None:
                preprocess(repo['code_path'])
                success, error_msg = run_lightweight_method(repo['code_path'], config)    
            
            else:
                success = "Error"
                error_msg = "There is no zip file."

        metadata.append([repo['title'], repo['framework'], success, error_msg, 
                             repo['date'], repo['tags'], repo['stars'], repo['code'], 
                             repo['paper']])

    return metadata

def lightweight_method(code_path, config: LightWeightMethodConfig):
    """ running the lightweight graph construction """
    explorer = TFTokenExplorer(code_path, config)
    explorer.explore_code_repository()

def run_lightweight_method(code_path, config: LightWeightMethodConfig) -> tuple:
    """Runs lightweight method.
    If exception occurs, try to fix the error by running 2to3 and autopep8.
    2to3 converts python2 code to python3 compatible code.
    autopep8 fixes inconsistent use of tabs and spaces error.
    
    Arguments:
        config {LightWeightMethodConfig} -- Config for Lightweight method.
    
    Returns:
        tuple -- success: Ouput status, error_msg: Exception raised by Lightweight method.
    """
    success = "N/A"
    error_msg = "N/A"
    try:
        lightweight_method(code_path, config)
        success = "Success"
    
    except:
        success = "Error"
        exc_type, exc_value, exc_traceback = sys.exc_info()
        error_msg = traceback.format_exception(exc_type, exc_value, exc_traceback)
        error_msg = ''.join(error_msg)
        print(''.join(error_msg))
        
    return (success, error_msg)

def move_output_files(config: LightWeightMethodConfig):
    """Move output files generated by lightweight method to destination path."""
    if 5 in config.output_types:
        if config.combined_triples_only:
            copy_files(config.input_path, config.dest_path, "combined_triples.triples")
        else:
            copy_files(config.input_path, config.dest_path, "*.triples")
            
    if 3 in config.output_types:
        copy_files(config.input_path, config.dest_path, "*.html")
    
    if 6 in config.output_types:
        copy_files(config.input_path, config.dest_path, "*.rdf")

def copy_files(data_path, dest_path, filetype, name_index=-3):
    for path in Path(data_path).rglob(filetype):
        path = Path(path)
        repo_name = str(path).split('/')[name_index]
        repo_path = Path(dest_path) / repo_name
        if not repo_path.is_dir():
            repo_path.mkdir(exist_ok=True)
        shutil.copy(path, repo_path)

def save_metadata(metadata: list, stat_file_path: str):
    with open(str(stat_file_path), 'w') as file:
        writer = csv.writer(file)
        writer.writerows(metadata)

def pipeline_the_lightweight_approach(args):

    config = LightWeightMethodConfig(LightWeightMethodArgParser().get_args(args))
    
    if config.recursive:
        config.dest_path.mkdir(exist_ok=True)
        stat_file_path = config.dest_path / "stats.csv"
        metadata = recursive(config.input_path, stat_file_path, config)
        save_metadata(metadata, str(stat_file_path))
        move_output_files(config)

    else:
        preprocess(config.input_path)
        run_lightweight_method(config.input_path, config)


if __name__ == "__main__":

    # append argument -h to see more options
    # ex1: python script_lightweight.py -ipt ../test/fashion_mnist
    # ex2: python script_lightweight.py -ipt ../test/VGG16 -opt 3 (get RDF graphs)
    # ex3: python script_lightweight.py -ipt ../test/Alexnet -opt 2 (get call trees)
    # ex4: python script_lightweight.py -ipt ../test/Xception
    # ex5: python script_lightweight.py -ipt ../test/NeuralStyle
    # ex6: python script_lightweight.py -ipt=../raw_data_tf --ds -dp=../rdf -opt=5
    # ex7: python script_lightweight.py -ipt=../raw_data_tf --ds -dp=../rdf -opt=3 --arg --url
    
    pipeline_the_lightweight_approach(sys.argv[1:])
