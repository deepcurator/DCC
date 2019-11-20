from pathlib import Path
from zipfile import ZipFile
import shutil
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

def retrieve_tasks_from_root_dir(data_path):
    
    dataset = extract_data(data_path)
    tasks = []

    for idx, repo in enumerate(dataset):
        if repo['framework'] and 'tf' in repo['framework']:

            if repo['code_path'] is not None:
                task = {'code_path': repo['code_path'],
                        'dir_name':  repo['folder_name']}
                tasks.append(task)

    return tasks


def extract_function_level_doc2vec(code_path, store_path, resolution='function'):
    # for doc2vec methods, generate text-based content for each function. 
    # TODO: to be modified to dump source code level & repository level.
    try:
        explorer = ASTExplorer(str(code_path), resolution=resolution)
        explorer.dump_functions_source_code(stored_path=store_path)
        
    except Exception as e:
        print(e)

def extract_function_level_ast(code_path, store_path, resolution='function'):
    try:
        explorer = ASTExplorer(str(code_path), resolution=resolution)
        explorer.process_all_nodes()
        # explorer.visualize_ast()
        explorer.export(stored_path=store_path)
    except Exception as e:
        print(e)


def graphast_pipeline(args):
    config = GraphASTConfig(GraphASTArgParser().get_args(args))
    config.dump() # sanity check for configurations. 
    
    # To make sure that both paths exist.
    root_path = config.input_path
    result_path = config.dest_path 
    result_path.mkdir(exist_ok=True) 
     
    # Task initialization.
    tasks = []

    if config.recursive:
        tasks = retrieve_tasks_from_root_dir(root_path)
    else:
        task = {'code_path': root_path, 
                'dir_name':  root_path.name}
        tasks.append(task)

    # Run task one-by-one. 
    for task in tasks:
        code_path = task['code_path']
        store_path = result_path / task['dir_name']
        store_path.mkdir(exist_ok=True) 
        resolution = config.resolution
        
        extract_function_level_doc2vec(code_path, store_path, resolution=resolution)
        extract_function_level_ast(code_path, store_path, resolution=resolution)
        # extract_function_level_ast()




if __name__ == "__main__":
    graphast_pipeline(sys.argv[1:])
    # test with "python script_graphast.py -ip ../test/ -r"
    # test with "python script_graphast.py"