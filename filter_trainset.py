from argparse import ArgumentParser
import codecs
import json
import os
import random

import numpy as np
from transformers import GPT2Tokenizer


def main():
    random.seed(42)
    np.random.seed(42)

    n_processes = max(os.cpu_count(), 1)
    print(f'Number of parallel processes is {n_processes}.')

    parser = ArgumentParser()
    parser.add_argument('-m', '--model', dest='model_name', type=str, required=True,
                        help='The FRED-T5 model name.')
    parser.add_argument('-i', '--input', dest='input_name', type=str, required=True,
                        help='The input name of JSON file with source structured dataset.')
    parser.add_argument('-o', '--output', dest='output_name', type=str, required=True,
                        help='The output name of JSON file with structured dataset after filtering.')
    parser.add_argument('--train_maxlen', dest='train_maxlen', type=int, required=True,
                        help='The maximal number of tokens per input or target for training.')
    parser.add_argument('--test_maxlen', dest='test_maxlen', type=int, required=True,
                        help='The maximal number of tokens per input or target for testing.')
    args = parser.parse_args()

    model_path = os.path.normpath(args.model_name)
    if not os.path.isdir(model_path):
        err_msg = f'The directory {model_path} does not exist!'
        raise IOError(err_msg)

    input_dataset_fname = os.path.normpath(args.input_name)
    if not os.path.isfile(input_dataset_fname):
        err_msg = f'The file {input_dataset_fname} does not exist!'
        raise IOError(err_msg)

    output_dataset_fname = os.path.normpath(args.output_name)
    if not os.path.isfile(output_dataset_fname):
        base_dir = os.path.dirname(output_dataset_fname)
        if not os.path.isdir(base_dir):
            err_msg = f'The directory {base_dir} does not exist!'
            raise IOError(err_msg)

    if args.train_maxlen is None:
        maximal_number_of_tokens_for_training = None
    else:
        maximal_number_of_tokens_for_training = args.train_maxlen
        if maximal_number_of_tokens_for_training < 10:
            err_msg = (f'The maximal number of tokens per input or target is too small! Expected 10 or greater, '
                       f'got {maximal_number_of_tokens_for_training}.')
            raise ValueError(err_msg)

    if args.test_maxlen is None:
        maximal_number_of_tokens_for_testing = None
    else:
        maximal_number_of_tokens_for_testing = args.test_maxlen
        if maximal_number_of_tokens_for_testing < 10:
            err_msg = (f'The maximal number of tokens per input or target is too small! Expected 10 or greater, '
                       f'got {maximal_number_of_tokens_for_testing}.')
            raise ValueError(err_msg)

    tokenizer = GPT2Tokenizer.from_pretrained(model_path)
    print(f'The pre-trained tokenizer "{os.path.basename(model_path)}" is loaded.')

    with codecs.open(input_dataset_fname, mode='r', encoding='utf-8', errors='ignore') as fp:
        data = json.load(fp)
    if not isinstance(data, dict):
        err_msg = f'The file "{input_dataset_fname}" contains a wrong data!'
        raise IOError(err_msg)
    if set(data.keys()) != {'text_corpus', 'validation', 'train'}:
        err_msg = f'The file "{input_dataset_fname}" contains a wrong data!'
        raise IOError(err_msg)
    united_text_corpus = data['text_corpus']
    data_for_validation = data['validation']
    data_for_training = data['train']

    n_training_samples = 0
    tasks_for_training = sorted(list(data_for_training.keys()))
    print(f'There are {len(tasks_for_training)} tasks for training.')
    for cur_task in tasks_for_training:
        if maximal_number_of_tokens_for_training is not None:
            data_for_training[cur_task] = list(filter(
                lambda sample: (len(tokenizer.tokenize(sample[0])) <= maximal_number_of_tokens_for_training) and
                               (len(tokenizer.tokenize(sample[1])) <= maximal_number_of_tokens_for_training),
                data_for_training[cur_task]
            ))
            if len(data_for_training[cur_task]) == 0:
                err_msg = (f'The maximal number of tokens per input or target = {maximal_number_of_tokens_for_training}'
                           f' is too strict! There are no samples of task {cur_task} in the training data '
                           f'after filtering.')
                raise ValueError(err_msg)
        print(f'There are {len(data_for_training[cur_task])} training samples for task {cur_task}.')
        n_training_samples += len(data_for_training[cur_task])
    print(f'The total number of training samples is {n_training_samples}.')

    tasks_for_validation = sorted(list(data_for_validation.keys()))
    print(f'There are {len(tasks_for_validation)} tasks for validation.')
    if set(tasks_for_training) != set(tasks_for_validation):
        err_msg = (f'The training tasks do not correspond to the validation tasks! '
                   f'{tasks_for_training} != {tasks_for_validation}')
        raise ValueError(err_msg)
    for cur_task in tasks_for_validation:
        if maximal_number_of_tokens_for_testing is not None:
            data_for_validation[cur_task] = list(filter(
                lambda sample: (len(tokenizer.tokenize(sample[0])) <= maximal_number_of_tokens_for_testing) and
                               (len(tokenizer.tokenize(sample[1])) <= maximal_number_of_tokens_for_testing),
                data_for_validation[cur_task]
            ))
            if len(data_for_validation[cur_task]) == 0:
                err_msg = (f'The maximal number of tokens per input or target = {maximal_number_of_tokens_for_testing} '
                           f'is too strict! There are no samples of task {cur_task} in the validation data '
                           f'after filtering.')
                raise ValueError(err_msg)
        print(f'There are {len(data_for_validation[cur_task])} validation samples for task {cur_task}.')

    united_text_corpus = list(filter(
        lambda sample: len(tokenizer.tokenize(sample)) <= maximal_number_of_tokens_for_testing,
        united_text_corpus
    ))

    with codecs.open(output_dataset_fname, mode='w', encoding='utf-8', errors='ignore') as fp:
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
    main()
