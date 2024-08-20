from argparse import ArgumentParser
import codecs
import csv
import os
import re

from nltk import wordpunct_tokenize


def is_russian(text: str) -> bool:
    re_for_russian = re.compile(r'^[АБВГДЕЁЖЗИЙКЛМНОПРСТУФХЦЧШЩЪЫЬЭЮЯ]+$')
    words = wordpunct_tokenize(text)
    if len(words) == 0:
        return False
    russian_words = list(filter(lambda it: re_for_russian.match(it.upper()) is not None, words))
    if len(russian_words) == 0:
        return False
    return (len(russian_words) / float(len(words))) >= 0.5


def main():
    parser = ArgumentParser()
    parser.add_argument('-i', '--input', dest='input_name', type=str, required=True,
                        help='The input CSV file name.')
    parser.add_argument('-o', '--output', dest='output_name', type=str, required=True,
                        help='The output CSV file name (after data filtering).')
    args = parser.parse_args()

    input_fname = os.path.normpath(args.input_name)
    if not os.path.isfile(input_fname):
        raise IOError(f'The file "{input_fname}" does not exist!')

    output_fname = os.path.normpath(args.output_name)
    if not os.path.isfile(output_fname):
        basedir = os.path.dirname(output_fname)
        if len(basedir) > 0:
            if not os.path.isdir(basedir):
                raise IOError(f'The directory "{basedir}" does not exist!')

    with codecs.open(input_fname, mode='r', encoding='utf-8', errors='ignore') as fp:
        data_reader = csv.reader(fp, delimiter=',', quotechar='"')
        rows = list(filter(lambda it: len(it) > 1, data_reader))
    print(f'{len(rows)} are loaded from the file "{input_fname}".')
    if len(rows) < 2:
        raise IOError(f'The file "{input_fname}" is empty!')
    rows_without_header = rows[1:]
    filtered_rows_without_header = list(filter(lambda it: is_russian(it[0]) and is_russian(it[1]), rows_without_header))
    if len(filtered_rows_without_header) == 0:
        raise RuntimeError(f'The file "{input_fname}" does not contain any Russian input text!')

    print(f'{len(filtered_rows_without_header) + 1} will be saved into the file "{output_fname}".')
    with codecs.open(output_fname, mode='w', encoding='utf-8', errors='ignore') as fp:
        data_writer = csv.writer(fp, delimiter=',', quotechar='"')
        data_writer.writerow(rows[0])
        for it in filtered_rows_without_header:
            data_writer.writerow(it)


if __name__ == '__main__':
    main()
