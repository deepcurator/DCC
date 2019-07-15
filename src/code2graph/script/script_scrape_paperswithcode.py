import sys
from selenium import webdriver
import threading
import time
sys.path.append('../')

from core.paperswithcode import PWCScraper
from config.config import PaperswithcodeArgParser

def scrape_paperswithcode(args):
    scraper = PWCScraper(args.chromedriver)
    while True:
        print("Spawning new thread.")
        hourly_thread = threading.Thread(target=scraper.fetch_metadata, args=(args.url, args.limit))
        hourly_thread.start()
        time.sleep(3600)

if __name__ == "__main__":
    args = PaperswithcodeArgParser().get_args()
    scrape_paperswithcode(args)