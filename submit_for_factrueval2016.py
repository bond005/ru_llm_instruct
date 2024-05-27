from argparse import ArgumentParser
import codecs
import logging
import os
import random
import sys
from typing import List, Tuple

import numpy as np
import spacy
from transformers import T5ForConditionalGeneration, GPT2Tokenizer, GenerationConfig
import torch
from tqdm import tqdm

from inference.inference import generate_answer
from instructions.instructions import KNOWN_TASKS
from ner.ner import calculate_entity_bounds
from utils.utils import split_long_text


fredt5_logger = logging.getLogger(__name__)


def check_entity_bounds(entity_bounds: List[Tuple[int, int]], text: str):
    for entity_start, entity_end in entity_bounds:
        err_msg = f'The entity ({entity_start}, {entity_end}) is wrong for text {text}'
        if text[entity_start].isspace():
            fredt5_logger.error(err_msg)
            raise ValueError(err_msg)
        if text[entity_end - 1].isspace():
            fredt5_logger.error(err_msg)
            raise ValueError(err_msg)
        if entity_start > 0:
            if text[entity_start - 1].isalnum():
                fredt5_logger.error(err_msg)
                raise ValueError(err_msg)
        if entity_end < len(text):
            if text[entity_end].isalnum():
                fredt5_logger.error(err_msg)
                raise ValueError(err_msg)


def main():
    task_types = [it[1] for it in KNOWN_TASKS]
    ne_org_ID = task_types.index('ner_organization')
    ne_loc_ID = task_types.index('ner_location')
    ne_per_ID = task_types.index('ner_person')
    del task_types

    random.seed(42)
    torch.manual_seed(42)
    np.random.seed(42)

    if not torch.cuda.is_available():
        err_msg = 'CUDA is not available!'
        fredt5_logger.error(err_msg)
        raise ValueError(err_msg)
    device = torch.device('cuda')

    nlp = spacy.load('ru_core_news_sm')

    n_processes = max(os.cpu_count(), 1)
    fredt5_logger.info(f'Number of parallel processes is {n_processes}.')

    parser = ArgumentParser()
    parser.add_argument('-m', '--model', dest='model_name', type=str, required=True,
                        help='The input name of FRED-T5.')
    parser.add_argument('-o', '--output', dest='output_submission_dir', type=str, required=True,
                        help='The output directory for the submission.')
    parser.add_argument('-i', '--input', dest='input_data_name', type=str, required=True,
                        help='The FactRuEval-2016 directory with test data.')
    parser.add_argument('--no_lm_tag', dest='no_lm_tag', action='store_true', required=False,
                        help='The <LM> tag is not used.')
    parser.add_argument('--dtype', dest='dtype', required=False, default='float32', type=str,
                        choices=['float32', 'float16', 'bfloat16', 'bf16', 'fp16', 'fp32'],
                        help='The PyTorch tensor type for inference.')
    args = parser.parse_args()

    submission_dirname: str = os.path.normpath(args.output_submission_dir)
    if not os.path.isdir(submission_dirname):
        basedir = os.path.dirname(submission_dirname)
        if len(basedir) > 0:
            if not os.path.isdir(basedir):
                err_msg = f'The directory {basedir} does not exist!'
                fredt5_logger.error(err_msg)
                raise IOError(err_msg)
        os.mkdir(submission_dirname)

    dataset_dirname = os.path.normpath(args.input_data_name)
    if not os.path.isdir(dataset_dirname):
        err_msg = f'The directory {dataset_dirname} does not exist!'
        fredt5_logger.error(err_msg)
        raise IOError(err_msg)

    fredt5_name = os.path.normpath(args.model_name)
    if not os.path.isdir(fredt5_name):
        err_msg = f'The directory {fredt5_name} does not exist!'
        fredt5_logger.error(err_msg)
        raise IOError(err_msg)

    all_text_files = list(filter(
        lambda it: it.lower().endswith('.txt') and it.lower().startswith('book_'),
        os.listdir(dataset_dirname)
    ))
    if len(all_text_files) == 0:
        err_msg = f'The directory "{dataset_dirname}" does not contain any text file!'
        fredt5_logger.error(err_msg)
        raise IOError(err_msg)
    base_names: List[str] = []
    source_texts: List[str] = []
    for cur_fname in tqdm(all_text_files):
        full_fname = os.path.join(dataset_dirname, cur_fname)
        point_pos = cur_fname.rfind('.')
        if point_pos >= 0:
            new_base_name = cur_fname[:point_pos].strip()
        else:
            new_base_name = cur_fname
        if len(new_base_name) == 0:
            err_msg = f'The name "{new_base_name}" is wrong!'
            fredt5_logger.error(err_msg)
            raise RuntimeError(err_msg)
        base_names.append(new_base_name)
        with open(full_fname, 'rb') as fp:
            full_text = fp.read().decode('utf-8')
        if len(full_text.strip()) == 0:
            err_msg = f'The file "{full_fname}" is empty!'
            fredt5_logger.error(err_msg)
            raise IOError(err_msg)
        source_texts.append(full_text)
    info_msg = f'There are {len(source_texts)} texts in the "{dataset_dirname}".'
    fredt5_logger.info(info_msg)
    text_lengths = sorted([len(it) for it in source_texts])
    info_msg = (f'The minimal text length is {text_lengths[0]}, '
                f'the mean text length is {round(sum(text_lengths) / len(source_texts))}, '
                f'the maximal text length is {text_lengths[-1]}.')
    fredt5_logger.info(info_msg)

    if args.dtype in {'float16', 'fp16'}:
        model = T5ForConditionalGeneration.from_pretrained(fredt5_name, torch_dtype=torch.float16).to(device)
    elif args.dtype in {'bfloat16', 'bf16'}:
        model = T5ForConditionalGeneration.from_pretrained(fredt5_name, torch_dtype=torch.bfloat16).to(device)
    else:
        model = T5ForConditionalGeneration.from_pretrained(fredt5_name, torch_dtype=torch.float32).to(device)
    model.eval()
    tokenizer = GPT2Tokenizer.from_pretrained(fredt5_name)
    generation_config = GenerationConfig.from_pretrained(fredt5_name)
    generation_config.no_repeat_ngram_size = 0
    fredt5_logger.info(f'The instruct model "{os.path.basename(fredt5_name)}" is loaded.')
    fredt5_logger.info(f'{generation_config}'.replace('\n', ' ').replace('\r', ''))

    if args.no_lm_tag:
        question_prefix = ''
    else:
        question_prefix = '<LM>'
    max_text_len = generation_config.max_length * 2
    fredt5_logger.info(f'Maximal text length is {max_text_len}.')
    no_entity_text_len = len(tokenizer.tokenize('В этом тексте нет именованных сущностей такого типа.</s>')) + 1
    for base_fname, source_text in tqdm(zip(base_names, source_texts), total=len(source_texts)):
        bounds_of_paragraphs = split_long_text(source_text, max_text_len, nlp)
        recognition_results = []
        for start_, end_ in bounds_of_paragraphs:
            questions = [
                question_prefix + KNOWN_TASKS[ne_org_ID][0] + ' ' + source_text[start_:end_],
                question_prefix + KNOWN_TASKS[ne_loc_ID][0] + ' ' + source_text[start_:end_],
                question_prefix + KNOWN_TASKS[ne_per_ID][0] + ' ' + source_text[start_:end_]
            ]
            n_tokens_in_text = len(tokenizer.tokenize(source_text[start_:end_]))
            if n_tokens_in_text < 3:
                err_msg = f'The text is too short! {source_text[start_:end_]}'
                fredt5_logger.error(err_msg)
                raise ValueError(err_msg)
            generation_config.max_length = max(round(0.33 * n_tokens_in_text) + 1, no_entity_text_len)
            answers = generate_answer(
                questions=questions,
                tokenizer=tokenizer,
                model=model,
                config=generation_config
            )
            organizations = list(filter(lambda it2: len(it2) > 0, map(lambda it1: it1.strip(), answers[0].split('\n'))))
            locations = list(filter(lambda it2: len(it2) > 0, map(lambda it1: it1.strip(), answers[1].split('\n'))))
            persons = list(filter(lambda it2: len(it2) > 0, map(lambda it1: it1.strip(), answers[2].split('\n'))))
            bounds_of_organizations = calculate_entity_bounds(source_text[start_:end_], organizations)
            check_entity_bounds(bounds_of_organizations, source_text[start_:end_])
            bounds_of_locations = calculate_entity_bounds(source_text[start_:end_], locations)
            check_entity_bounds(bounds_of_locations, source_text[start_:end_])
            bounds_of_persons = calculate_entity_bounds(source_text[start_:end_], persons)
            check_entity_bounds(bounds_of_persons, source_text[start_:end_])
            recognition_results += [('org', it[0] + start_, it[1] - it[0]) for it in bounds_of_organizations]
            recognition_results += [('loc', it[0] + start_, it[1] - it[0]) for it in bounds_of_locations]
            recognition_results += [('per', it[0] + start_, it[1] - it[0]) for it in bounds_of_persons]
            del bounds_of_persons, bounds_of_locations, bounds_of_organizations
            del organizations, persons, locations
            del answers, questions
        recognition_results.sort(key=lambda it: (it[1], it[2], it[0]))
        del bounds_of_paragraphs
        full_fname = os.path.join(submission_dirname, base_fname + '.task1')
        with codecs.open(filename=full_fname, mode='w', encoding='utf-8') as fp:
            for res in recognition_results:
                fp.write(f'{res[0]} {res[1]} {res[2]}\n')


if __name__ == '__main__':
    fredt5_logger.setLevel(logging.INFO)
    fmt_str = '%(filename)s[LINE:%(lineno)d]# %(levelname)-8s ' \
              '[%(asctime)s]  %(message)s'
    formatter = logging.Formatter(fmt_str)
    stdout_handler = logging.StreamHandler(sys.stdout)
    stdout_handler.setFormatter(formatter)
    fredt5_logger.addHandler(stdout_handler)
    file_handler = logging.FileHandler('fredt5_instruct_evaluation_on_factrueval2016.log')
    file_handler.setFormatter(formatter)
    fredt5_logger.addHandler(file_handler)
    main()
