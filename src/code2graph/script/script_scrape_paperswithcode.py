import sys
from selenium import webdriver
import threading
import time
sys.path.append('../')

from core.paperswithcode import fetch_metadata
from config.config import PaperswithcodeArgParser

def scrape_paperswithcode(args):
    while True:
        print("Spawning new thread.")
        hourly_thread = threading.Thread(target=fetch_metadata, args=(args.chromedriver, args.url, args.limit))
        hourly_thread.start()
        time.sleep(3600)

if __name__ == "__main__":
    args = PaperswithcodeArgParser().get_args()
    scrape_paperswithcode(args)