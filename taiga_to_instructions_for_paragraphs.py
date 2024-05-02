from argparse import ArgumentParser
import codecs
import csv
import random
import os
from typing import List

from tqdm import tqdm

from segmentation.segmentation import load_samples_from_taiga
from instructions.instructions import KNOWN_TASKS


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
        textfiles += list(map(
            lambda y: os.path.join(it, y),
            filter(
                lambda x: x.endswith('.txt'),
                os.listdir(it)
            )
        ))
        textfiles += find_all_textfiles(it)
    return textfiles


def main():
    parser = ArgumentParser()
    parser.add_argument('-s', '--src', dest='source_corpus_name', type=str, required=True,
                        help='The path to the input Taiga corpus.')
    parser.add_argument('-t', '--train', dest='train_fname', type=str, required=True,
                        help='The output CSV file name with instructions for training.')
    parser.add_argument('-e', '--eval', dest='eval_fname', type=str, required=True,
                        help='The output CSV file name with instructions for evaluation.')
    args = parser.parse_args()

    segmentation_instruction = ''
    for val in KNOWN_TASKS:
        if val[1] == 'segmentation':
            segmentation_instruction = val[0]
            break
    if len(segmentation_instruction) == 0:
        raise RuntimeError(f'The segmentation instruction is unknown! {KNOWN_TASKS}')

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
    if len(all_textfiles) > 2000:
        all_textfiles = all_textfiles[:2000]
    print(f'There are {len(all_textfiles)} text files.')
    train_fp = None
    validation_fp = None
    try:
        train_fp = codecs.open(output_csv_fname_for_training, mode='w', encoding='utf-8')
        validation_fp = codecs.open(output_csv_fname_for_validation, mode='w', encoding='utf-8')
        trainset_writer = csv.writer(train_fp, delimiter=',', quotechar='"')
        valset_writer = csv.writer(validation_fp, delimiter=',', quotechar='"')
        trainset_writer.writerow(['input', 'target'])
        valset_writer.writerow(['input', 'target'])
        for cur_textfile in tqdm(all_textfiles):
            new_samples = list(filter(
                lambda cur: len(cur[0]) < 20_000,
                load_samples_from_taiga(cur_textfile)
            ))
            if len(new_samples) > 0:
                if random.random() > 0.9:
                    for cur_input, cur_target in new_samples:
                        valset_writer.writerow([f'<LM>{segmentation_instruction} ' + cur_input, cur_target + '</s>'])
                else:
                    for cur_input, cur_target in new_samples:
                        trainset_writer.writerow([f'<LM>{segmentation_instruction} ' + cur_input, cur_target + '</s>'])
    finally:
        if train_fp is not None:
            train_fp.close()
        if validation_fp is not None:
            validation_fp.close()


if __name__ == '__main__':
    main()
