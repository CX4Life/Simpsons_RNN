"""Create a dictionary containing all the vocabulary in every episode of The Simpsons"""

import os
import string
import json
from collections import Counter
from glob import glob

__author__ = 'Tim Woods'
__copyright__ = 'Copyright (c) 2017 Tim Woods'
__license__ = 'MIT'

TRANSCRIPT_DIR = 'transcripts/'
DICT_SIZE = 10000

overall_dictionary = {}

def get_all_script_files():
    """Get all text files from the transcript directory"""
    pwd = os.getcwd()
    os.chdir(TRANSCRIPT_DIR)
    ret = glob('*.txt')
    os.chdir(pwd)
    return ret


def extract_words_from_text(text):
    """Clean the words of any leading or trailing punctuation, and return as a list."""
    every_word = text.split(' ')
    return [y for y in[x.strip(string.punctuation).lower().rstrip() for x in every_word] if len(y)]


def update_dict_from_list(list_of_words):
    """For every cleaned word in the list, update the dictionary's count of that word."""
    for word in list_of_words:
        if word in overall_dictionary:
            overall_dictionary[word] += 1
        else:
            overall_dictionary[word] = 1


def read_file_to_dict(filename):
    """Read each word in a script and add it to the overall dictionary."""
    with open(TRANSCRIPT_DIR + filename, 'rb') as file:
        for bytes in file:
            readable_text = bytes.decode('ascii')
            stripped = extract_words_from_text(readable_text)
            update_dict_from_list(stripped)

def main():
    """Read every script, add each word to a dictionary, then write the 10000 most common
    words to a JSON file.
    """
    every_file = get_all_script_files()
    for filename in every_file:
        read_file_to_dict(filename)

    most_common = dict(Counter(overall_dictionary).most_common(DICT_SIZE))
    with open('most_common.json', 'w') as common:
        json.dump(most_common, common)


if __name__ == '__main__':
    main()
