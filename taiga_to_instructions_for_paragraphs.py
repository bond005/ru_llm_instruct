from argparse import ArgumentParser
import codecs
import csv
import random
import os
import warnings
from typing import List

from nltk import wordpunct_tokenize


def find_all_textfiles(basedir: str) -> List[str]:
    all_items = list(filter(
        lambda it3: os.path.isdir(it3),
        map(
            lambda it2: os.path.join(basedir, it2),
            filter(
                lambda it1: it1 not in {'.', '..'},
                os.listdir(basedir)
            )
        )
    ))
    if len(all_items) == 0:
        return []
    textfiles = []
    for it in all_items:
        textfiles += list(filter(lambda x: x.endswith('.txt'), os.listdir(it)))
        textfiles += find_all_textfiles(it)
    return textfiles


def load_text_file(fname: str) -> List[str]:
    texts = []
    new_text = []
    with codecs.open(fname, mode='r', encoding='utf-8') as fp:
        curline = fp.readline()
        while len(curline) > 0:
            prepline = curline.strip()
            if len(prepline) > 0:
                words_in_line = wordpunct_tokenize(prepline)
                if len(words_in_line) < 5:
                    if len(new_text) > 0:
                        texts.append('\n'.join(new_text))
                    del new_text
                    new_text = []
                else:
                    new_text.append(' '.join(prepline.split()))
            curline = fp.readline()
    if len(new_text) > 0:
        texts.append('\n'.join(new_text))
    return texts


def main():
    parser = ArgumentParser()
    parser.add_argument('-s', '--src', dest='source_corpus_name', type=str, required=True,
                        help='The path to the input Taiga corpus.')
    parser.add_argument('-t', '--train', dest='train_fname', type=str, required=True,
                        help='The output CSV file name with instructions for training.')
    parser.add_argument('-e', '--eval', dest='eval_fname', type=str, required=True,
                        help='The output CSV file name with instructions for evaluation.')
    args = parser.parse_args()

    corpus_dir = os.path.normpath(args.source_corpus_name)
    if not os.path.isdir(corpus_dir):
        raise IOError(f'The directory "{corpus_dir}" does not exist!')

    output_csv_fname_for_training = os.path.normpath(args.train_fname)
    if not os.path.isfile(output_csv_fname_for_training):
        basedir = os.path.dirname(output_csv_fname_for_training)
        if not os.path.isdir(basedir):
            raise ValueError(f'The directory "{basedir}" does not exist!')

    output_csv_fname_for_validation = os.path.normpath(args.eval_fname)
    if not os.path.isfile(output_csv_fname_for_validation):
        basedir = os.path.dirname(output_csv_fname_for_validation)
        if not os.path.isdir(basedir):
            raise ValueError(f'The directory "{basedir}" does not exist!')

    all_textfiles = find_all_textfiles(corpus_dir)
    if len(all_textfiles) == 0:
        raise ValueError(f'The corpus "{corpus_dir}" does not contain any text file!')

    if len(all_textfiles) < 10:
        err_msg = (f'The corpus "{corpus_dir}" contains a few number of text files! '
                   f'Expected 10 or greater, got {len(all_textfiles)}.')
        raise ValueError(err_msg)

    random.shuffle(all_textfiles)
    train_fp = None
    validation_fp = None
    try:
        train_fp = codecs.open(output_csv_fname_for_training, mode='w', encoding='utf-8')
        validation_fp = codecs.open(output_csv_fname_for_validation, mode='w', encoding='utf-8')
        trainset_writer = csv.writer(train_fp, delimiter=',', quotechar='"')
        valset_writer = csv.writer(validation_fp, delimiter=',', quotechar='"')
        trainset_writer.writerow(['input', 'target'])
        valset_writer.writerow(['input', 'target'])

    finally:
        if train_fp is not None:
            train_fp.close()
        if validation_fp is not None:
            validation_fp.close()
    pass


if __name__ == '__main__':
    main()
