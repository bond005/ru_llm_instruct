from argparse import ArgumentParser
import codecs
import csv
import json
import os
import random
import warnings

import numpy as np
from sklearn.model_selection import train_test_split


RANDOM_SEED: int = 42


def main():
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    parser = ArgumentParser()
    parser.add_argument('-i', '--input', dest='input_name', type=str, required=True,
                        help='The input JSONL file with RuTurboAlpaca data.')
    parser.add_argument('-o', '--output', dest='output_name', type=str, required=True,
                        help='The output HF-formatted dataset path.')
    args = parser.parse_args()

    input_fname = os.path.normpath(args.input_name)
    if not os.path.isfile(input_fname):
        raise IOError(f'The file {input_fname} does not exist!')
    output_dataset_path = os.path.normpath(args.output_name)
    if not os.path.isdir(output_dataset_path):
        basedir = os.path.dirname(output_dataset_path)
        if not os.path.isdir(basedir):
            raise IOError(f'The directory {basedir} does not exist!')
        os.mkdir(output_dataset_path)
    trainset_fname = os.path.join(output_dataset_path, 'train_data.csv')
    valset_fname = os.path.join(output_dataset_path, 'validation_data.csv')
    testset_fname = os.path.join(output_dataset_path, 'test_data.csv')

    source_data = []
    meta_data = []
    line_idx = 1
    with codecs.open(input_fname, mode='r', encoding='utf-8') as fp:
        curline = fp.readline()
        while len(curline) > 0:
            prepline = curline.strip()
            if len(prepline) > 0:
                err_msg = f'"{input_fname}": the line {line_idx} is wrong!'
                try:
                    new_sample = json.loads(prepline)
                except Exception as err:
                    raise IOError(err_msg + ' ' + str(err))
                if not isinstance(new_sample, dict):
                    err_msg_ = err_msg + f' Expected {type({"a": 1, "b": 2})}, got {type(new_sample)}.'
                    raise IOError(err_msg_)
                if 'source' not in new_sample:
                    err_msg_ = err_msg + ' The "source" key is not found!'
                    raise IOError(err_msg_)
                if 'model_name' not in new_sample:
                    err_msg_ = err_msg + ' The "model_name" key is not found!'
                    raise IOError(err_msg_)
                if 'messages' not in new_sample:
                    err_msg_ = err_msg + ' The "messages" key is not found!'
                    raise IOError(err_msg_)
                messages = new_sample['messages']
                if not isinstance(messages, list):
                    err_msg_ = err_msg + f' The messages are wrong! Expected {type([1, 2])}, got {type(messages)}.'
                    raise IOError(err_msg_)
                if len(messages) == 0:
                    warn_msg = err_msg + ' The messages are empty!'
                    warnings.warn(warn_msg)
                else:
                    if len(messages) < 2:
                        warn_msg = err_msg + f' The messages list is too short! ' \
                                             f'Expected 2 or greater, got {len(messages)}.'
                        warnings.warn(warn_msg)
                    else:
                        new_dialogue = []
                        expected_role = 'user'
                        for cur_msg in messages:
                            if not isinstance(cur_msg, dict):
                                err_msg_ = err_msg + ' The messages content is wrong!'
                                raise IOError(err_msg_)
                            if ('role' not in cur_msg) or ('content' not in cur_msg):
                                err_msg_ = err_msg + ' The messages content is wrong!'
                                raise IOError(err_msg_)
                            if cur_msg['role'] not in {'user', 'bot'}:
                                err_msg_ = err_msg + ' The messages content is wrong!'
                                raise IOError(err_msg_)
                            if cur_msg['role'] != expected_role:
                                err_msg_ = err_msg + ' The messages content is wrong!'
                                raise IOError(err_msg_)
                            if len(cur_msg['content'].strip()) == 0:
                                err_msg_ = err_msg + ' The messages content is wrong!'
                                raise IOError(err_msg_)
                            new_dialogue.append(' '.join(cur_msg['content'].strip().split()))
                            if expected_role == 'user':
                                expected_role = 'bot'
                            else:
                                expected_role = 'user'
                        if (len(new_dialogue) < 2) or ((len(new_dialogue) % 2) != 0):
                            warn_msg = err_msg + ' The messages content is wrong!'
                            warnings.warn(warn_msg)
                        else:
                            source_data.append(new_dialogue)
                            sample_info = f'{new_sample["source"].strip().lower()}'.strip()
                            sample_info += f' {new_sample["model_name"].strip().lower()}'.strip()
                            meta_data.append(sample_info.strip())
                        del new_dialogue
            curline = fp.readline()
            line_idx += 1
    if len(source_data) < 5_000:
        raise IOError(f'There are too few samples in the "{input_fname}".')

    labels_for_stratification = []
    unique_labels = sorted(list(set(meta_data)))
    for val in meta_data:
        labels_for_stratification.append(unique_labels.index(val))

    X_train, X_val, y_train, y_val = train_test_split(
        source_data, labels_for_stratification, test_size=100, random_state=RANDOM_SEED,
        stratify=labels_for_stratification,
        shuffle=True
    )

    test_size = max(500, round(0.05 * len(X_train)))
    X_train, X_test, y_train, y_test = train_test_split(
        X_train, y_train, test_size=test_size, random_state=RANDOM_SEED,
        stratify=y_train,
        shuffle=True
    )

    with codecs.open(trainset_fname, mode='w', encoding='utf-8', errors='ignore') as fp:
        data_writer = csv.writer(fp, delimiter=',', quotechar='"')
        data_writer.writerow(['input', 'target'])
        for sample in X_train:
            n = len(sample) // 2
            history = ''
            for idx in range(n):
                user = sample[idx * 2]
                bot = sample[idx * 2 + 1]
                history += ' ' + user
                data_writer.writerow([history.strip(), bot])
                history += ' ' + bot
            del history

    with codecs.open(valset_fname, mode='w', encoding='utf-8', errors='ignore') as fp:
        data_writer = csv.writer(fp, delimiter=',', quotechar='"')
        data_writer.writerow(['input', 'target'])
        for sample in X_val:
            n = len(sample) // 2
            history = ''
            for idx in range(n):
                user = sample[idx * 2]
                bot = sample[idx * 2 + 1]
                history += ' ' + user
                data_writer.writerow([history.strip(), bot])
                history += ' ' + bot
            del history

    with codecs.open(testset_fname, mode='w', encoding='utf-8', errors='ignore') as fp:
        data_writer = csv.writer(fp, delimiter=',', quotechar='"')
        data_writer.writerow(['input', 'target'])
        for sample in X_test:
            n = len(sample) // 2
            history = ''
            for idx in range(n):
                user = sample[idx * 2]
                bot = sample[idx * 2 + 1]
                history += ' ' + user
                data_writer.writerow([history.strip(), bot])
                history += ' ' + bot
            del history


if __name__ == '__main__':
    main()
