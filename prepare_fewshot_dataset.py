import codecs
import csv
from argparse import ArgumentParser
import logging
import os
import random
import sys

import numpy as np

from training.training import load_trainset, add_few_shot_tasks
from training.training import training_logger


fredt5_trainset_logger = logging.getLogger(__name__)
MIN_SAMPLES_PER_TASK: int = 100


def main():
    random.seed(42)
    np.random.seed(42)

    parser = ArgumentParser()
    parser.add_argument('-i', '--input', dest='input_name', type=str, required=True,
                        help='The path to the input dataset with "train_data.csv" and "test_data.csv".')
    parser.add_argument('-o', '--output', dest='output_name', type=str, required=True,
                        help='The path to the output dataset with "train_data.csv" and "test_data.csv".')
    parser.add_argument('--no_lm_tag', dest='no_lm_tag', action='store_true', required=False,
                        help='The <LM> tag is not used.')
    args = parser.parse_args()

    input_dataset_path = os.path.normpath(args.input_name)
    if not os.path.isdir(input_dataset_path):
        err_msg = f'The directory {input_dataset_path} does not exist!'
        fredt5_trainset_logger.error(err_msg)
        raise ValueError(err_msg)
    traindata_name = os.path.join(input_dataset_path, 'train_data.csv')
    testdata_name = os.path.join(input_dataset_path, 'test_data.csv')
    if not os.path.isfile(traindata_name):
        err_msg = f'The file {traindata_name} does not exist!'
        fredt5_trainset_logger.error(err_msg)
        raise ValueError(err_msg)
    if not os.path.isfile(testdata_name):
        err_msg = f'The file {testdata_name} does not exist!'
        fredt5_trainset_logger.error(err_msg)
        raise ValueError(err_msg)

    output_dataset_path = os.path.normpath(args.output_name)
    if not os.path.isdir(output_dataset_path):
        base_dir = os.path.dirname(output_dataset_path)
        if len(base_dir) > 0:
            if not os.path.isdir(base_dir):
                err_msg = f'The directory {base_dir} does not exist!'
                fredt5_trainset_logger.error(err_msg)
                raise ValueError(err_msg)
        os.mkdir(output_dataset_path)

    n_training_samples = 0
    try:
        data_for_training = load_trainset(traindata_name, lm_tag=not args.no_lm_tag)
    except Exception as err:
        fredt5_trainset_logger.error(str(err))
        raise
    tasks_for_training = sorted(list(data_for_training.keys()))
    data_for_fewshot_samples = dict()
    data_for_training_questions = dict()
    data_for_validation_questions = dict()
    fredt5_trainset_logger.info(f'There are {len(tasks_for_training)} tasks for training.')
    for cur_task in tasks_for_training:
        n_samples_per_task = len(data_for_training[cur_task])
        info_msg = f'There are {n_samples_per_task} training samples for task {cur_task}.'
        fredt5_trainset_logger.info(info_msg)
        if n_samples_per_task < MIN_SAMPLES_PER_TASK:
            err_msg = f'Number of samples for the task {cur_task} is too small! ' \
                      f'Expected {MIN_SAMPLES_PER_TASK} or greater, got {n_samples_per_task}.'
            fredt5_trainset_logger.error(err_msg)
            raise ValueError(err_msg)
        n_training_samples += len(data_for_training[cur_task])
        random.shuffle(data_for_training[cur_task])
        n_fewshot_samples = round(0.6 * n_samples_per_task)
        n_validation_questions = min(round(0.1 * n_samples_per_task), max(5, round(300 / len(tasks_for_training))))
        n_training_questions = n_samples_per_task - n_fewshot_samples - n_validation_questions
        data_for_fewshot_samples[cur_task] = data_for_training[cur_task][0:n_fewshot_samples]
        data_for_training_questions[cur_task] = data_for_training[cur_task][n_fewshot_samples:(n_fewshot_samples + n_training_questions)]
        data_for_validation_questions[cur_task] = data_for_training[cur_task][(n_fewshot_samples + n_training_questions):]
    fredt5_trainset_logger.info(f'The total number of training samples is {n_training_samples}.')
    del data_for_training

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

    prepared_data_for_training = add_few_shot_tasks(
        few_shot=data_for_fewshot_samples,
        questions=data_for_training_questions
    )
    info_msg = f'The training data for the few-shot dataset are prepared. ' \
               f'There are {len(prepared_data_for_training)} training samples.'
    fredt5_trainset_logger.info(info_msg)
    prepared_data_for_validation = add_few_shot_tasks(
        few_shot=data_for_fewshot_samples,
        questions=data_for_validation_questions
    )
    info_msg = f'The validation data for the few-shot dataset are prepared. ' \
               f'There are {len(prepared_data_for_validation)} validation samples.'
    fredt5_trainset_logger.info(info_msg)
    prepared_data_for_testing = add_few_shot_tasks(
        few_shot=data_for_fewshot_samples,
        questions=data_for_validation
    )
    info_msg = f'The test data for the few-shot dataset are prepared. ' \
               f'There are {len(prepared_data_for_testing)} test samples.'
    fredt5_trainset_logger.info(info_msg)

    with codecs.open(os.path.join(output_dataset_path, 'train_data.csv'), mode='w', encoding='utf-8') as fp:
        data_writer = csv.writer(fp, delimiter=',', quotechar='"')
        data_writer.writerow(['input', 'target'])
        for input_text, target_text in prepared_data_for_training:
            data_writer.writerow([input_text, target_text])

    with codecs.open(os.path.join(output_dataset_path, 'validation_data.csv'), mode='w', encoding='utf-8') as fp:
        data_writer = csv.writer(fp, delimiter=',', quotechar='"')
        data_writer.writerow(['input', 'target'])
        for input_text, target_text in prepared_data_for_validation:
            data_writer.writerow([input_text, target_text])

    with codecs.open(os.path.join(output_dataset_path, 'test_data.csv'), mode='w', encoding='utf-8') as fp:
        data_writer = csv.writer(fp, delimiter=',', quotechar='"')
        data_writer.writerow(['input', 'target'])
        for input_text, target_text in prepared_data_for_testing:
            data_writer.writerow([input_text, target_text])


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
    file_handler = logging.FileHandler('fredt5_fewshot_trainset.log')
    file_handler.setFormatter(formatter)
    fredt5_trainset_logger.addHandler(file_handler)
    training_logger.addHandler(file_handler)
    main()
