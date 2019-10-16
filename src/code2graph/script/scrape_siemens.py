import sys
import pandas as pd
import threading
import time
import queue
from pathlib import Path
import csv
import shutil
from zipfile import ZipFile
from argparse import Namespace
import urllib
import wget

sys.path.append('../')
from config.config import LightWeightMethodConfig
from core.graphlightweight import TFTokenExplorer
from config.config import PWCConfigArgParser, PWCConfig
from core.paperswithcode import PWCScraper
from script.script_lightweight import preprocess, run_lightweight_method, copy_files, save_metadata

def fetch_code(code_link, path):
        decomposed = code_link.split('/')
        assert len(decomposed) == 5
        if 'github' in decomposed[2]:
            decomposed.append('archive/master.zip')
            reconstructed = decomposed[0] + "//" + decomposed[2] + "/" + \
                decomposed[3] + "/" + decomposed[4] + "/" + decomposed[5]
            wget.download(reconstructed, out=str(path))
            return path
        else:
            return None

def extract_code(dir_path):
    zip_path = list(Path(dir_path).glob('*.zip'))
    if zip_path:
        zip_path = zip_path[0]
    else:
        return None
    extract_name = (zip_path.name).split('.')[0]
    extract_path = Path(dir_path) / extract_name
    # remove directory if it already exists
    if extract_path.exists():
        shutil.rmtree(extract_path)
    # unzip file
    with ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(extract_path)
    return extract_path

def run_lightweight(paper: dict, out_path: Path, out_types: list):

    success = "N/A"
    error_msg = "N/A"

    code_path = extract_code(paper['stored_dir_path'])
    if not code_path:
        return "Error", "No code repository found."
    preprocess(code_path)

    args = Namespace(input_path=code_path, is_dataset=False, dest_path=".",
                     combined_triples_only=False, recursive=False,
                     output_types=[6], show_arg=True, show_url=True)
    config = LightWeightMethodConfig(args)

    success, error_msg = run_lightweight_method(code_path, config, paper)
    for filetype in out_types:
        copy_files(code_path, out_path, filetype)
        
    return success, error_msg

def scrape_from_csv_file(path):
    df = pd.read_csv(path)

    data_path = Path("./siemens_data").resolve()
    data_path.mkdir(exist_ok=True)
    output_path = Path("./siemens_output").resolve()
    output_path.mkdir(exist_ok=True)
    output_types = ['*.rdf']

    success_list = []

    for index, paper in df.iterrows():
        if paper['Platform'] == 'tensorflow':
            success = "Error"

            meta = {'code': paper['repo_link'], 'paper': paper['paper_link']}
            dir_name = paper['paper_link'].split('/')[-1].replace('.pdf','').replace('.html','')
            meta['stored_dir_name'] = dir_name
            
            dir_path = data_path / dir_name
            dir_path.mkdir(exist_ok=True)
            meta['stored_dir_path'] = dir_path

            try:
                fetch_code(meta['code'], dir_path)
            except Exception as e:
                print(e)
                success_list.append("Error")
                continue

            success, err_msg = run_lightweight(meta, output_path, output_types)
                
            success_list.append(success)
        
        else:
            success_list.append('N/A')
    
    df['Lightweight'] = success_list
    df.to_csv(r'pwc_out.csv')

if __name__ == "__main__":
    scrape_from_csv_file("pwc.csv")