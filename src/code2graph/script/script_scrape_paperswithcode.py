import sys, threading, time
import queue
from pathlib import Path
import csv
import shutil
from zipfile import ZipFile
import subprocess
import traceback
from argparse import Namespace

sys.path.append('../')
from core.paperswithcode import PWCScraper
from config.config import PWCConfigArgParser, PWCConfig
from core.graphlightweight import TFTokenExplorer
from config.config import LightWeightMethodConfig

triple_save_path = Path("./triples").resolve()
stats_file_path = triple_save_path / "stats.csv"

def process(paper: dict):

    if not stats_file_path.exists():
        with open(str(stats_file_path), 'w') as file:
            writer = csv.writer(file)
            writer.writerow(['Title','Framework','Lightweight','Error Msg',
                            'Date','Tags','Stars','Code Link','Paper Link'])
    
    success = "N/A"
    error_msg = "N/A"

    zip_path = list(Path(paper['stored_dir_path']).glob('*.zip'))[0]
    extract_name = (zip_path.name).split('.')[0]
    extract_path = Path(paper['stored_dir_path']) / extract_name
    # remove directory if it already exists
    if extract_path.exists():
        shutil.rmtree(extract_path)
    # unzip file
    with ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(extract_path)
    
    # convert python2 code to python3
    subprocess.run("2to3 -w -n %s" % extract_path, shell=True)
    # fix indent errors
    subprocess.run("autopep8 --in-place -r %s" % extract_path, shell=True)

    args = Namespace(code_path=extract_path, is_dataset=False, dest_path=".",
                        combined_triples_only=False,
                        output_types=[3,5,6], show_arg=True, show_url=True)
    config = LightWeightMethodConfig(args)
    try:
        explorer = TFTokenExplorer(config)
        explorer.explore_code_repository()
        success = "Success"
    except:
        success = "Error"
        exc_type, exc_value, exc_traceback = sys.exc_info()
        error_msg = traceback.format_exception(exc_type, exc_value, exc_traceback)
        error_msg = ''.join(error_msg)
        print(''.join(error_msg))
        pass

    with open(stats_file_path, 'a') as file:
        writer = csv.writer(file)
        writer.writerow([paper['title'], paper['framework'], success, error_msg, 
                            paper['date'], ','.join(paper['tags']), paper['stars'], paper['code_link'], 
                            paper['paper_link']])

def service(scraper: PWCScraper):
    scraper.scrape()
    while True:
        try:
            paper = scraper.tf_papers.get_nowait()
            process(paper)
        except queue.Empty:
            break

def service_scrape_papers(args):
    config = PWCConfig(PWCConfigArgParser().get_args(args))
    config.tot_paper_to_scrape_per_shot = -1
    scraper = PWCScraper(config)
    
    while True:
        print("Spawning new thread.")
        hourly_thread = threading.Thread(target=service, args=(scraper,))
        hourly_thread.start()
        time.sleep(3600)


if __name__ == "__main__":
    service_scrape_papers(sys.argv[1:])