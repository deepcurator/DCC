import sys, threading, time
sys.path.append('../')


from core.paperswithcode import PWCScraper
from config.config import PWCConfigArgParser, PWCConfig


def service_scrape_papers(args):
    config = PWCConfig(PWCConfigArgParser().get_args(args))
    scraper = PWCScraper(config)
    
    while True:
        print("Spawning new thread.")
        hourly_thread = threading.Thread(target=scraper.scrape, args=())
        hourly_thread.start()
        time.sleep(3600)


if __name__ == "__main__":
    service_scrape_papers(sys.argv[1:])