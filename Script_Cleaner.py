"""Takes a dictionary of the 10000 most common words, and replaces any words outside of said
dictionary with an unknown symbol. This will simplify using word2vec to create word embeddings.
"""

import json
import os
import operator
from glob import glob
from Vocabulary_Finder import extract_words_from_text

__author__ = 'Tim Woods'
__copyright__ = 'Copyright (c) 2017 Tim Woods'
__license__ = 'MIT'

TRANSCRIPT_DIR = 'transcripts/'
CLEANED_DIR = 'cleaned/'
JSON_FILENAME = 'most_common.json'
UNKNOWN_CHAR = '<x>'


def load_dictionary():
    """Load the most common words from a JSON file"""
    dict = {}
    with open(JSON_FILENAME, 'r') as json_file:
        for line in json_file:
            dict = json.loads(line)
    return dict


def replace_text(list_of_words, dict):
    """Removes words not in most common from script and replaces with
    UNKNOWN_CHAR.
    """
    return [x if (x in dict) else UNKNOWN_CHAR for x in list_of_words]


def write_to_file_using_ints(list_of_text, lookup_dict, filename):
    """Write the cleaned text to file."""
    list_of_ints = [lookup_dict[x] if x in lookup_dict else -1 for x in list_of_text]
    with open(CLEANED_DIR + filename, 'w') as output:
        json.dump(list_of_ints, output)


def clean_script(filename, dict):
    """Opens, decodes and splits script in same manner as Vocabulary_Finder.py,
    and passes list of words to replace_text.
    """
    with open(TRANSCRIPT_DIR + filename, 'rb') as script_file:
        for bytes in script_file:
            decoded = bytes.decode('ascii')
            list_of_words = extract_words_from_text(decoded)
            cleaned_words = replace_text(list_of_words, dict)
            return cleaned_words


def get_all_scripts():
    pwd = os.getcwd()
    os.chdir(TRANSCRIPT_DIR)
    every_script = glob('*.txt')
    os.chdir(pwd)
    return every_script


def make_key_val_by_frequency(dictionary):
    """Make a dictionary that links each word to its frequency rank."""
    word_to_val = {}
    val_to_word = {}
    sorted_x = sorted(dictionary.items(), key=operator.itemgetter(1), reverse=True)
    for rank, tuple in enumerate(sorted_x):
        word_to_val[tuple[0]] = rank
        val_to_word[rank] = tuple[0]
    return word_to_val, val_to_word


def write_val_to_word_to_file(val_map):
    """Write the lookup dictionary to file."""
    with open('lookup_dict.json', 'w') as out_file:
        json.dump(val_map, out_file)


def main():
    """Load the dictionary of most common words, then convert each script to a
    list of integer values based on the word frequency within in the common
    word dictionary, and write those to a file. Additionally, write a lookup
    dictionary to a JSON file for converting the integer values back to words."""

    most_common_words = load_dictionary()
    word_to_val, val_to_word = make_key_val_by_frequency(most_common_words)
    just_keys = set(most_common_words.keys())
    scripts = get_all_scripts()
    for script in scripts:
        text_without_uncommon = clean_script(script, just_keys)
        if text_without_uncommon:
            write_to_file_using_ints(text_without_uncommon, word_to_val, script)

    write_val_to_word_to_file(val_to_word)


if __name__ == '__main__':
    main()