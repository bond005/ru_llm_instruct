import codecs
import json
from argparse import ArgumentParser
import logging
import os
import random
import sys

import numpy as np

from training.training import load_trainset, add_few_shot_tasks
from training.training import training_logger


fredt5_trainset_logger = logging.getLogger(__name__)


def main():
    random.seed(42)
    np.random.seed(42)

    parser = ArgumentParser()
    parser.add_argument('-i', '--input', dest='input_name', type=str, required=True,
                        help='The path to the input dataset with "train_data.csv" and "test_data.csv".')
    parser.add_argument('-o', '--output', dest='output_name', type=str, required=True,
                        help='The output name of JSON file with structured dataset.')
    parser.add_argument('--no_lm_tag', dest='no_lm_tag', action='store_true', required=False,
                        help='The <LM> tag is not used.')
    parser.add_argument('--few_shot', dest='few_shot', action='store_true', required=False,
                        help='The few-shot inference is added.')
    args = parser.parse_args()

    structured_dataset_fname = os.path.normpath(args.output_name)
    if (len(structured_dataset_fname) == 0) or (structured_dataset_fname == '.'):
        err_msg = f'The file name "{structured_dataset_fname}" is incorrect!'
        fredt5_trainset_logger.error(err_msg)
        raise ValueError(err_msg)
    parent_dir_name = os.path.dirname(structured_dataset_fname)
    if not os.path.isdir(parent_dir_name):
        err_msg = f'The directory {parent_dir_name} does not exist!'
        fredt5_trainset_logger.error(err_msg)
        raise ValueError(err_msg)

    dataset_path = os.path.normpath(args.input_name)
    if not os.path.isdir(dataset_path):
        err_msg = f'The directory {dataset_path} does not exist!'
        fredt5_trainset_logger.error(err_msg)
        raise ValueError(err_msg)
    traindata_name = os.path.join(dataset_path, 'train_data.csv')
    testdata_name = os.path.join(dataset_path, 'test_data.csv')
    if not os.path.isfile(traindata_name):
        err_msg = f'The file {traindata_name} does not exist!'
        fredt5_trainset_logger.error(err_msg)
        raise ValueError(err_msg)
    if not os.path.isfile(testdata_name):
        err_msg = f'The file {testdata_name} does not exist!'
        fredt5_trainset_logger.error(err_msg)
        raise ValueError(err_msg)

    united_text_corpus = []
    max_text_len = 0

    n_training_samples = 0
    try:
        data_for_training = load_trainset(traindata_name, lm_tag=not args.no_lm_tag)
    except Exception as err:
        fredt5_trainset_logger.error(str(err))
        raise
    tasks_for_training = sorted(list(data_for_training.keys()))
    fredt5_trainset_logger.info(f'There are {len(tasks_for_training)} tasks for training.')
    for cur_task in tasks_for_training:
        fredt5_trainset_logger.info(f'There are {len(data_for_training[cur_task])} training samples for task {cur_task}.')
        n_training_samples += len(data_for_training[cur_task])
        for text_pair in data_for_training[cur_task]:
            if cur_task != 'asr_correction':
                if args.no_lm_tag:
                    united_text_corpus.append(' '.join(text_pair[0].strip().split()))
                else:
                    united_text_corpus.append(' '.join(text_pair[0][4:].strip().split()))
                if len(united_text_corpus[-1]) > max_text_len:
                    max_text_len = len(united_text_corpus[-1])
            united_text_corpus.append(' '.join(text_pair[1][:-4].strip().split()))
            if len(united_text_corpus[-1]) > max_text_len:
                max_text_len = len(united_text_corpus[-1])
    fredt5_trainset_logger.info(f'The total number of training samples is {n_training_samples}.')

    try:
        data_for_validation = load_trainset(testdata_name, lm_tag=not args.no_lm_tag)
    except Exception as err:
        fredt5_trainset_logger.error(str(err))
        raise
    tasks_for_validation = sorted(list(data_for_validation.keys()))
    fredt5_trainset_logger.info(f'There are {len(tasks_for_validation)} tasks for validation.')
    if set(tasks_for_training) != set(tasks_for_validation):
        err_msg = (f'The training tasks do not correspond to the validation tasks! '
                   f'{tasks_for_training} != {tasks_for_validation}')
        fredt5_trainset_logger.error(err_msg)
        raise ValueError(err_msg)
    for cur_task in tasks_for_validation:
        fredt5_trainset_logger.info(f'There are {len(data_for_validation[cur_task])} validation samples for task {cur_task}.')
        for text_pair in data_for_validation[cur_task]:
            if cur_task != 'asr_correction':
                if args.no_lm_tag:
                    united_text_corpus.append(' '.join(text_pair[0].strip().split()))
                else:
                    united_text_corpus.append(' '.join(text_pair[0][4:].strip().split()))
                if len(united_text_corpus[-1]) > max_text_len:
                    max_text_len = len(united_text_corpus[-1])
            united_text_corpus.append(' '.join(text_pair[1][:-4].strip().split()))
            if len(united_text_corpus[-1]) > max_text_len:
                max_text_len = len(united_text_corpus[-1])

    united_text_corpus = sorted(list(set(united_text_corpus)))
    info_msg = f'There are {len(united_text_corpus)} unique texts. The maximal text length is {max_text_len}.'
    fredt5_trainset_logger.info(info_msg)
    fredt5_trainset_logger.info(f'The maximal characters in the text is {max_text_len}.')

    if args.few_shot:
        try:
            data_for_training = add_few_shot_tasks(data_for_training)
        except Exception as err:
            fredt5_trainset_logger.error(str(err))
            raise
        n_training_samples = 0
        for cur_task in tasks_for_training:
            n_training_samples += len(data_for_training[cur_task])
        info_msg = (f'The few-shot examples are added. The total number of training samples '
                    f'after the training set extension is {n_training_samples}.')
        fredt5_trainset_logger.info(info_msg)

    with codecs.open(structured_dataset_fname, mode='w', encoding='utf-8', errors='ignore') as fp:
        json.dump(
            fp=fp,
            obj={
                'train': data_for_training,
                'validation': data_for_validation,
                'text_corpus': united_text_corpus
            },
            ensure_ascii=False, indent=4
        )


if __name__ == '__main__':
    fredt5_trainset_logger.setLevel(logging.INFO)
    training_logger.setLevel(logging.INFO)
    fmt_str = '%(filename)s[LINE:%(lineno)d]# %(levelname)-8s ' \
              '[%(asctime)s]  %(message)s'
    formatter = logging.Formatter(fmt_str)
    stdout_handler = logging.StreamHandler(sys.stdout)
    stdout_handler.setFormatter(formatter)
    fredt5_trainset_logger.addHandler(stdout_handler)
    training_logger.addHandler(stdout_handler)
    file_handler = logging.FileHandler('fredt5_instruct_trainset.log')
    file_handler.setFormatter(formatter)
    fredt5_trainset_logger.addHandler(file_handler)
    training_logger.addHandler(file_handler)
    main()
