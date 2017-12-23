"""Create a dictionary containing all the vocabulary in every episode of The Simpsons"""

import os
from glob import glob

__author__ = 'Tim Woods'
__copyright__ = 'Copyright (c) 2017 Tim Woods'
__license__ = 'MIT'

TRANSCRIPT_DIR = '/transcripts'

def get_all_script_files():
    """Get all text files from the transcript directory"""
    os.chdir(TRANSCRIPT_DIR)
    return glob('.txt')


def read_file_to_dict(filename):
    """Read each word in a script and add it to the overall dictionary."""
    print(filename)


def main():
    """Do the damn thing"""
    get_all_script_files()


if __name__ == '__main__':
    main()
