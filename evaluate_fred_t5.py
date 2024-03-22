import codecs
import json
from argparse import ArgumentParser
import logging
import math
import os
import random
import sys

import numpy as np
from transformers import T5ForConditionalGeneration, GPT2Tokenizer, GenerationConfig
import torch

from instructions.instructions import evaluate, load_evaluator
from training.training import load_trainset
from utils.utils import process_multiline


fredt5_logger = logging.getLogger(__name__)


def main():
    random.seed(42)
    torch.manual_seed(42)
    np.random.seed(42)

    if not torch.cuda.is_available():
        err_msg = 'CUDA is not available!'
        fredt5_logger.error(err_msg)
        raise ValueError(err_msg)
    device = torch.device('cuda')

    n_processes = max(os.cpu_count(), 1)
    fredt5_logger.info(f'Number of parallel processes is {n_processes}.')

    parser = ArgumentParser()
    parser.add_argument('-m', '--model', dest='model_name', type=str, required=True,
                        help='The input name of FRED-T5.')
    parser.add_argument('-o', '--output', dest='output_report_name', type=str, required=True,
                        help='The output name of the JSON report.')
    parser.add_argument('-i', '--input', dest='data_name', type=str, required=True,
                        help='The path to the dataset with "train_data.csv" and "test_data.csv".')
    parser.add_argument('--batch', dest='batch_size', type=int, required=True,
                        help='The mini-batch size for FRED-T5.')
    parser.add_argument('--eval_model', dest='eval_model', type=str, required=True,
                        help='The path to the pre-trained FRED-T5 for using as a BERT score calculator.')
    parser.add_argument('--eval_batch', dest='eval_batch_size', type=int, required=True,
                        help='The mini-batch size for the BERT score calculator.')
    parser.add_argument('--testsize', dest='testsize', type=int, required=False, default=None,
                        help='The maximal number of validation samples per validated task.')
    args = parser.parse_args()

    report_name = os.path.normpath(args.output_report_name)
    if not os.path.isfile(report_name):
        dirname = os.path.dirname(report_name)
        if len(dirname) > 0:
            if not os.path.isdir(dirname):
                err_msg = f'The directory {dirname} does not exist!'
                fredt5_logger.error(err_msg)
                raise ValueError(err_msg)

    minibatch_size = args.batch_size
    if minibatch_size <= 0:
        err_msg = f'The mini-batch size {args.batch_size} is wrong!'
        fredt5_logger.error(err_msg)
        raise ValueError(err_msg)
    fredt5_logger.info(f'Mini-batch size is {minibatch_size}.')

    dataset_path = os.path.normpath(args.data_name)
    if not os.path.isdir(dataset_path):
        err_msg = f'The directory {dataset_path} does not exist!'
        fredt5_logger.error(err_msg)
        raise ValueError(err_msg)
    traindata_name = os.path.join(dataset_path, 'train_data.csv')
    testdata_name = os.path.join(dataset_path, 'test_data.csv')
    if not os.path.isfile(traindata_name):
        err_msg = f'The file {traindata_name} does not exist!'
        fredt5_logger.error(err_msg)
        raise ValueError(err_msg)
    if not os.path.isfile(testdata_name):
        err_msg = f'The file {testdata_name} does not exist!'
        fredt5_logger.error(err_msg)
        raise ValueError(err_msg)

    fredt5_name = os.path.normpath(args.model_name)
    if not os.path.isdir(fredt5_name):
        err_msg = f'The directory {fredt5_name} does not exist!'
        fredt5_logger.error(err_msg)
        raise ValueError(err_msg)

    scorer_path = os.path.normpath(args.eval_model)
    if not os.path.isdir(scorer_path):
        err_msg = f'The directory {scorer_path} does not exist!'
        fredt5_logger.error(err_msg)
        raise ValueError(err_msg)

    united_text_corpus = []
    max_text_len = 0

    n_training_samples = 0
    try:
        data_for_training = load_trainset(traindata_name)
    except Exception as err:
        fredt5_logger.error(str(err))
        raise
    tasks_for_training = sorted(list(data_for_training.keys()))
    fredt5_logger.info(f'There are {len(tasks_for_training)} tasks for training.')
    for cur_task in tasks_for_training:
        fredt5_logger.info(f'There are {len(data_for_training[cur_task])} training samples for task {cur_task}.')
        n_training_samples += len(data_for_training[cur_task])
        for text_pair in data_for_training[cur_task]:
            if cur_task != 'asr_correction':
                united_text_corpus.append(' '.join(text_pair[0][4:].strip().split()))
                if len(united_text_corpus[-1]) > max_text_len:
                    max_text_len = len(united_text_corpus[-1])
            united_text_corpus.append(' '.join(text_pair[1][:-4].strip().split()))
            if len(united_text_corpus[-1]) > max_text_len:
                max_text_len = len(united_text_corpus[-1])
    fredt5_logger.info(f'The total number of training samples is {n_training_samples}.')

    try:
        data_for_validation = load_trainset(testdata_name)
    except Exception as err:
        fredt5_logger.error(str(err))
        raise
    tasks_for_validation = sorted(list(data_for_validation.keys()))
    fredt5_logger.info(f'There are {len(tasks_for_validation)} tasks for validation.')
    for cur_task in tasks_for_validation:
        fredt5_logger.info(f'There are {len(data_for_validation[cur_task])} validation samples for task {cur_task}.')
        for text_pair in data_for_validation[cur_task]:
            if cur_task != 'asr_correction':
                united_text_corpus.append(' '.join(text_pair[0][4:].strip().split()))
                if len(united_text_corpus[-1]) > max_text_len:
                    max_text_len = len(united_text_corpus[-1])
            united_text_corpus.append(' '.join(text_pair[1][:-4].strip().split()))
            if len(united_text_corpus[-1]) > max_text_len:
                max_text_len = len(united_text_corpus[-1])

    united_text_corpus = sorted(list(set(united_text_corpus)))
    fredt5_logger.info(f'There are {len(united_text_corpus)} unique texts. The maximal text length is {max_text_len}.')
    fredt5_logger.info(f'The maximal characters in the text is {max_text_len}.')

    fredt5_logger.info('There are 5 randomly samples texts:')
    for it in random.sample(united_text_corpus, k=5):
        fredt5_logger.info(it)

    try:
        scorer = load_evaluator(scorer_path, args.eval_batch_size, united_text_corpus)
    except Exception as err:
        fredt5_logger.error(str(err))
        raise
    fredt5_logger.info(f'The BERT scorer based on FRED-T5 "{os.path.basename(scorer_path)}" is loaded.')

    model = T5ForConditionalGeneration.from_pretrained(fredt5_name).to(device)
    model.eval()
    tokenizer = GPT2Tokenizer.from_pretrained(fredt5_name)
    fredt5_logger.info(f'The pre-trained model "{os.path.basename(fredt5_name)}" is loaded.')

    generation_config = GenerationConfig.from_pretrained(fredt5_name)
    fredt5_logger.info(f'{generation_config}')

    fredt5_logger.info(f'The known tasks are {tasks_for_validation}')
    if args.testsize is not None:
        if args.testsize < 6:
            err_msg = (f'The maximal number of validation samples per validated task is too small! '
                       f'Expected 6 or greater, got {args.testsize}.')
            fredt5_logger.error(err_msg)
            raise ValueError(err_msg)
        for task in tasks_for_validation:
            if task.startswith('ner_'):
                samples_with_ne = list(filter(
                    lambda it: it[1].find('нет именованных сущностей такого типа') < 0,
                    data_for_validation[task]
                ))
                samples_without_ne = list(filter(
                    lambda it: it[1].find('нет именованных сущностей такого типа') >= 0,
                    data_for_validation[task]
                ))
                if (len(samples_with_ne) + len(samples_without_ne)) != len(data_for_validation[task]):
                    err_msg = f'The data for task {task} is incorrect!'
                    fredt5_logger.error(err_msg)
                    raise ValueError(err_msg)
                if len(samples_with_ne) == 0:
                    err_msg = f'The data for task {task} is incorrect!'
                    fredt5_logger.error(err_msg)
                    raise ValueError(err_msg)
                if len(samples_without_ne) == 0:
                    err_msg = f'The data for task {task} is incorrect!'
                    fredt5_logger.error(err_msg)
                    raise ValueError(err_msg)
                if len(data_for_validation[task]) > args.testsize:
                    if len(samples_with_ne) > math.ceil(args.testsize / 2):
                        samples_for_ner = random.sample(samples_with_ne, k=math.ceil(args.testsize / 2))
                    else:
                        samples_for_ner = samples_with_ne
                    if len(samples_without_ne) > (args.testsize - len(samples_for_ner)):
                        samples_for_ner += random.sample(samples_without_ne, k=args.testsize - len(samples_for_ner))
                    else:
                        samples_for_ner += samples_without_ne
                    del data_for_validation[task]
                    data_for_validation[task] = samples_for_ner
                    del samples_for_ner
                del samples_with_ne, samples_without_ne
            else:
                if len(data_for_validation[task]) > args.testsize:
                    data_for_validation[task] = random.sample(data_for_validation[task], k=args.testsize)

        for task in tasks_for_validation:
            info_msg = f'There are {len(data_for_validation[task])} test samples for task {task} after reducing.'
            fredt5_logger.info(info_msg)

    try:
        best_score, results_by_tasks = evaluate(data_for_validation,
                                                tokenizer, generation_config, model, minibatch_size,
                                                scorer)
    except Exception as err:
        fredt5_logger.error(str(err))
        raise
    fredt5_logger.info(f'United recognition score is {best_score}.')
    for cur_task in tasks_for_validation:
        fredt5_logger.info(f'Recognition results for the task {cur_task}:')
        cur_score, printed_results = results_by_tasks[cur_task]
        printed_results_for_json = []
        for old_item in printed_results:
            new_item = {
                'INPUT': process_multiline(old_item['INPUT']),
                'PREDICTED': process_multiline(old_item['PREDICTED']),
                'TRUE': process_multiline(old_item['TRUE'])
            }
            printed_results_for_json.append(new_item)
        del printed_results
        if cur_task == 'asr_correction':
            fredt5_logger.info('Word accuracy is {0:.5%}.'.format(cur_score))
        elif cur_task == 'segmentation':
            fredt5_logger.info('Paragraph accuracy is {0:.5%}.'.format(cur_score))
        if cur_task.startswith('ner_'):
            fredt5_logger.info('F1 by entities is {0:.6f}.'.format(cur_score))
        else:
            fredt5_logger.info('BERT-score F1 is {0:.6f}.'.format(cur_score))
        results_by_tasks[cur_task] = (cur_score, printed_results_for_json)
        del printed_results_for_json

    with codecs.open(report_name, mode='w', encoding='utf-8', errors='ignore') as fp:
        json.dump(obj=results_by_tasks, fp=fp, ensure_ascii=False, indent=4)


if __name__ == '__main__':
    fredt5_logger.setLevel(logging.INFO)
    fmt_str = '%(filename)s[LINE:%(lineno)d]# %(levelname)-8s ' \
              '[%(asctime)s]  %(message)s'
    formatter = logging.Formatter(fmt_str)
    stdout_handler = logging.StreamHandler(sys.stdout)
    stdout_handler.setFormatter(formatter)
    fredt5_logger.addHandler(stdout_handler)
    file_handler = logging.FileHandler('fredt5_instruct_evaluation.log')
    file_handler.setFormatter(formatter)
    fredt5_logger.addHandler(file_handler)
    main()
