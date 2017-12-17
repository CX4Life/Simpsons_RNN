"""Web scraper to retrieve Simpsons scripts from the site www.springfieldspringfield.co.uk
and save as text files.
"""
__author__ = 'Tim Woods'
__license__ = 'MIT'
__copyright__ = 'Copyright (c) 2017 Tim Woods'

from queue import Queue
import urllib
import argparse
import threading
from bs4 import BeautifulSoup

BASE_URL = 'https://www.springfieldspringfield.co.uk/'
ROOT = 'episode_scripts.php?tv-show=the-simpsons'


def parse_args():
    """Add support for excluding characters from the script by using the
    -exclude parameter, followed by a string of characters to exclude.
    """
    parser = argparse.ArgumentParser(description='Scrapper.py scrapes the Springfield, Springfield'
                                                 ' site for every Simpsons script ever! Optionally'
                                                 ' takes a string of characters to exclude from the'
                                                 ' raw strips as an argument.')
    parser.add_argument("-exclude",
                        type=str,
                        default='',
                        help='String of characters to COMPLETELY exclude from the script')
    return parser.parse_args()


def return_links_to_all_episodes():
    """Get the links to every page containing a script."""
    all_links_page = urllib.request.urlopen(BASE_URL + ROOT).read()
    soup = BeautifulSoup(all_links_page, "html5lib")
    links = soup.find_all("a", class_='season-episode-title')
    return links


def get_html_from_script_page(page_url):
    """Fetch the text from each episode page and return it as a stripped string."""
    script_page = urllib.request.urlopen(BASE_URL + page_url)
    soup = BeautifulSoup(script_page, "html5lib")
    script = soup.find("div", class_="scrolling-script-container").get_text()
    return script.lstrip().rstrip()


def filesystem_friendly_name(episode_name):
    """Change human-readable episode name to filename by removing number
    and replacing spaces with underscores."""
    chunks = episode_name.split(' ')
    return '_'.join(chunks[1:])


def clean_script(script_string, args):
    """Remove excessive punctuation from the script."""
    excluded_chars = args.exclude
    script = ''.join([x for x in script_string if x not in excluded_chars])
    return script


def write_script_to_file(queue, args):
    """Remove a link tag from the queue, get the episode name, retrieve the script,
    and write to a txt file.
    """
    while True:
        link_tag = queue.get()
        page_url = link_tag.get('href')
        episode_name = filesystem_friendly_name(link_tag.get_text())
        print(episode_name)
        script = get_html_from_script_page(page_url)
        output_file = open(episode_name + '.txt', 'w')
        output_file.write(clean_script(script, args))
        queue.task_done()


def main():
    """Get every link from the 'root' URL, then retrieve the script for each
    episode's page using 8 threads, and write it to a text file."""
    args = parse_args()
    every_link = return_links_to_all_episodes()
    link_queue = Queue()
    for link_tag in every_link:
        link_queue.put(link_tag)

    for _ in range(8):
        thread = threading.Thread(target=write_script_to_file(link_queue, args))
        thread.daemon = True
        thread.start()

    link_queue.join()


if __name__ == '__main__':
    main()
