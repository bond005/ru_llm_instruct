from argparse import ArgumentParser
import codecs
import gc
import json
import logging
import os
import random
import sys
from typing import Dict, List, Optional, Set, Tuple, Union
import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)

from augmentex.char import CharAug
import bitsandbytes as bnb
from datasets import load_dataset
from nltk import wordpunct_tokenize
import numpy as np
from tqdm import trange, tqdm
from transformers import GPT2Tokenizer, GenerationConfig
from transformers import LongformerTokenizerFast, LongformerModel
from turbot5 import T5ForConditionalGeneration
import torch

from instructions.instructions import evaluate_any_task
from utils.utils import generate_arithmetic_sample, generate_sample_with_choice, generate_sample_with_comparison


fredt5_training_logger = logging.getLogger(__name__)
NEGATIVE_ANSWER_1 = 'к сожалению я не могу ответить на ваш вопрос'
NEGATIVE_ANSWER_2 = 'в этом тексте нет именованных сущностей такого типа'
MAX_GENERATION_TIME: float = 2.0
RANDOM_SEED: int = 42


def is_positive_answer(answer: str) -> bool:
    words_of_answer = set(filter(
        lambda it2: it2.isalnum(),
        map(lambda it1: it1.strip(), wordpunct_tokenize(answer.lower()))
    ))
    if words_of_answer == set(NEGATIVE_ANSWER_1.split()):
        return False
    return words_of_answer != set(NEGATIVE_ANSWER_2.split())


def augment_text(input_text: str, augmenters: List[CharAug], use_lm_tag: bool,
                 existed_texts: Union[Set[str], None]) -> Union[str, None]:
    selected_augmenter = random.choice(augmenters)
    if use_lm_tag:
        new_input_text = selected_augmenter.augment(input_text[4:])
    else:
        new_input_text = selected_augmenter.augment(input_text)
    if existed_texts is not None:
        if new_input_text in existed_texts:
            new_input_text = None
    if new_input_text is None:
        return None
    if use_lm_tag:
        new_input_text = '<LM>' + new_input_text
    return new_input_text


def calculate_sample_weights(data_for_training: List[Tuple[str, str, bool]]) -> List[float]:
    sample_weights = []
    for cur_sample in data_for_training:
        if cur_sample[2]:
            sample_weights.append(1.00)
        else:
            sample_weights.append(0.3)
    return sample_weights


def generate_answer_choice_on_stackoverflow(sample: Dict[str, List[Dict[str, Union[str, int]]]]) -> Tuple[str, str]:
    variants_of_prompt = [
        'Прочитай, пожалуйста, вопрос по программированию и выбери наиболее правильный вариант ответа.',
        'Прочитай вопрос по программной инженерии и выбери самый близкий к правде вариант ответа.',
        'Из всех вариантов ответа на вопрос выбери тот, который ближе всего к истине.',
        'Тебе задают вопрос по теме, связанной с программированием и компьютерами. Вместе с вопросом приведены '
        'варианты ответов. Какой из этих вариантов корректнее всего отвечает на вопрос?',
    ]
    letters = ['A', 'B', 'C', 'D', 'E']
    variants_of_answers = sample['answers']
    if len(variants_of_answers) > len(letters):
        variants_of_answers = variants_of_answers[:len(letters)]
    random.shuffle(variants_of_answers)
    if random.random() > 0.2:
        if random.random() > 0.5:
            if random.random() > 0.5:
                one_letter_msg = ' Запиши только букву верного варианта.'
            else:
                one_letter_msg = (' Запиши только букву верного варианта: ' +
                                  ', '.join([letters[idx] for idx in range(len(variants_of_answers) - 1)]) +
                                  ' или ' + letters[len(variants_of_answers) - 1])
        else:
            one_letter_msg = ''
        input_text = random.choice(variants_of_prompt)
        one_letter_before_question = (random.random() > 0.5)
        if len(one_letter_msg) > 0:
            if one_letter_before_question:
                random_value = random.random()
                if random_value > 0.7:
                    input_text += ' '
                elif random_value > 0.3:
                    input_text += '\n'
                else:
                    input_text += '\n\n'
                input_text += one_letter_msg
        random_value = random.random()
        if random_value > 0.7:
            input_text += ' '
        elif random_value > 0.3:
            input_text += '\n'
        else:
            input_text += '\n\n'
        input_text += 'Вопрос:'
        random_value = random.random()
        if random_value > 0.7:
            input_text += ' '
        elif random_value > 0.3:
            input_text += '\n'
        else:
            input_text += '\n\n'
        input_text += sample['question']
        random_value = random.random()
        if random_value > 0.7:
            input_text += ' Варианты ответов: ' + ' '.join(
                [(letters[idx] + '. ' + val['answer']) for idx, val in enumerate(variants_of_answers)]
            )
        elif random_value > 0.3:
            input_text += '\nВарианты ответов:\n' + '\n'.join(
                [(letters[idx] + '. ' + val['answer']) for idx, val in enumerate(variants_of_answers)]
            )
        else:
            input_text += '\n\nВарианты ответов:\n\n' + '\n\n'.join(
                [(letters[idx] + '. ' + val['answer']) for idx, val in enumerate(variants_of_answers)]
            )
        if len(one_letter_msg) > 0:
            if not one_letter_before_question:
                random_value = random.random()
                if random_value > 0.7:
                    input_text += ' '
                elif random_value > 0.3:
                    input_text += '\n'
                else:
                    input_text += '\n\n'
                input_text += one_letter_msg
    else:
        input_text = 'Вопрос:'
        random_value = random.random()
        if random_value > 0.7:
            input_text += ' '
        elif random_value > 0.3:
            input_text += '\n'
        else:
            input_text += '\n\n'
        input_text += sample['question']
        random_value = random.random()
        if random_value > 0.7:
            input_text += ' Варианты ответов: ' + ' '.join(
                [(letters[idx] + '. ' + val['answer']) for idx, val in enumerate(variants_of_answers)]
            )
        elif random_value > 0.3:
            input_text += '\nВарианты ответов:\n' + '\n'.join(
                [(letters[idx] + '. ' + val['answer']) for idx, val in enumerate(variants_of_answers)]
            )
        else:
            input_text += '\n\nВарианты ответов:\n\n' + '\n\n'.join(
                [(letters[idx] + '. ' + val['answer']) for idx, val in enumerate(variants_of_answers)]
            )
        random_value = random.random()
        if random_value > 0.7:
            input_text += ' '
        elif random_value > 0.3:
            input_text += '\n'
        else:
            input_text += '\n\n'
        input_text += random.choice([
            'Какой из ответов самый правильный?',
            'Какой из ответов ближе всего к истине?',
            'Какой ответ корректнее?',
            'Какой ответ вернее?',
            'Какой из ответов корректнее?'
        ])
        if random.random() > 0.5:
            if random.random() > 0.5:
                input_text += ' Запиши только букву верного варианта.'
            else:
                input_text += (' Запиши только букву верного варианта: ' +
                               ', '.join([letters[idx] for idx in range(len(variants_of_answers) - 1)]) +
                               ' или ' + letters[len(variants_of_answers) - 1])
    true_idx = 0
    for idx in range(1, len(variants_of_answers)):
        if variants_of_answers[idx]['score'] > variants_of_answers[true_idx]['score']:
            true_idx = idx
    target_text = letters[true_idx] + '</s>'
    return input_text, target_text


def generate_samples_for_minibatch(data_for_training: List[Tuple[str, str, bool]], sample_weights: List[float],
                                   tokenizer: GPT2Tokenizer, arithmetics: bool, use_lm_tag: bool, minibatch_size: int,
                                   stackoverflow_data: Optional[Dict[str, List[Dict[str, \
                                           Union[str, List[Dict[str, Union[int, str]]]]]]]] = None,
                                   existed_texts: Optional[Set[str]] = None,
                                   augmenters: Optional[List[CharAug]] = None) -> List[Tuple[List[int], List[int]]]:
    selected_samples = random.choices(
        population=data_for_training,
        weights=sample_weights,
        k=minibatch_size
    )
    samples_for_batch = []
    for selected_sample in selected_samples:
        random_value = random.random()
        if arithmetics and (random_value < 0.15):
            if random_value < 0.05:
                input_text, target_text = generate_arithmetic_sample()
            elif random_value < 0.10:
                input_text, target_text = generate_sample_with_comparison()
            else:
                input_text, target_text = generate_sample_with_choice()
        elif (stackoverflow_data is not None) and (random_value < 0.25):
            if random.random() > 0.5:
                stackoverflow_sample = random.choice(stackoverflow_data['question_and_answer'])
                input_text = stackoverflow_sample['question']
                target_text = stackoverflow_sample['answer']
                if not target_text.endswith('</s>'):
                    target_text += '</s>'
            else:
                stackoverflow_sample = random.choice(stackoverflow_data['question_and_many_answers'])
                input_text, target_text = generate_answer_choice_on_stackoverflow(stackoverflow_sample)
            del stackoverflow_sample
        else:
            source_input = selected_sample[0]
            if augmenters is not None:
                augmented_input = augment_text(source_input, augmenters, use_lm_tag, existed_texts)
            else:
                augmented_input = None
            if augmented_input is None:
                input_text = source_input
            else:
                input_text = augmented_input
            target_text = selected_sample[1]
        if not target_text.endswith('</s>'):
            err_msg = f'The target text of the sample is wrong! ' \
                      f'Input: {" ".join(input_text.split())}, Target: {" ".join(target_text.split())}'
            fredt5_training_logger.error(err_msg)
            raise RuntimeError(err_msg)
        samples_for_batch.append((
            tokenizer.encode(input_text),
            tokenizer.encode(target_text)
        ))
    del selected_samples
    return samples_for_batch


def main():
    random.seed(RANDOM_SEED)
    torch.manual_seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    if not torch.cuda.is_available():
        err_msg = 'CUDA is not available!'
        fredt5_training_logger.error(err_msg)
        raise ValueError(err_msg)
    device = torch.device('cuda')
    torch.cuda.manual_seed(RANDOM_SEED)

    n_processes = max(os.cpu_count(), 1)
    fredt5_training_logger.info(f'Number of parallel processes is {n_processes}.')

    parser = ArgumentParser()
    parser.add_argument('-i', '--input', dest='input_name', type=str, required=True,
                        help='The input name of pre-trained FRED-T5.')
    parser.add_argument('-o', '--output', dest='output_name', type=str, required=True,
                        help='The output name of FRED-T5 after fine-tuning.')
    parser.add_argument('-d', '--data', dest='data_name', type=str, required=True,
                        help='The HF-formatted dataset name.')
    parser.add_argument('-a', '--additional', dest='additional_data_name', type=str, required=False,
                        default=None, action='append',
                        help='The additional HF-formatted dataset name (it will be used only for training).')
    parser.add_argument('--batch', dest='batch_size', type=int, required=True,
                        help='The mini-batch size for FRED-T5 training.')
    parser.add_argument('--alg', dest='algorithm', type=str, required=False, choices=['rmsprop', 'adamw', 'sgd', 'adam8bit'],
                        default='adamw', help='The training algorithm (RMSprop, AdamW, SGD, or Adam8Bit).')
    parser.add_argument('--eval_batch', dest='eval_batch_size', type=int, required=False, default=None,
                        help='The mini-batch size for FRED-T5 evaluation.')
    parser.add_argument('--eval_model', dest='eval_model', type=str, required=False,
                        default='kazzand/ru-longformer-tiny-16384',
                        help='The Longformer model for BERT score.')
    parser.add_argument('--eval_task', dest='eval_task', type=str, required=False,
                        default=None, action='append',
                        help='The evaluation task for monitoring.')
    parser.add_argument('--lr', dest='learning_rate', type=float, required=False, default=3e-4,
                        help='The learning rate.')
    parser.add_argument('--no_lm_tag', dest='no_lm_tag', action='store_true', required=False,
                        help='The <LM> tag is not used.')
    parser.add_argument('--augment', dest='augment', action='store_true', required=False,
                        help='The augmentation is not used.')
    parser.add_argument('--maxtokens', dest='maxtokens', type=int, required=False, default=None,
                        help='The maximal number of tokens for the training inputs.')
    parser.add_argument('--iters', dest='iters_per_epoch', type=int, required=False, default=None,
                        help='The iterations per epoch.')
    parser.add_argument('--no_pre_eval', dest='no_pre_eval', action='store_true', required=False,
                        help='The preliminary evaluation (before training) is not realized.')
    parser.add_argument('--arithmetics', dest='arithmetics', action='store_true', required=False,
                        help='Do you want to add arithmetic possibility into the training set?')
    parser.add_argument('--stackoverflow', dest='stackoverflow_data', type=str, required=False,
                        default=None, help='The name of name of JSON file with structured RuStackoverflow for '
                                           'training and validation.')
    parser.add_argument('--random', dest='random_seed', type=int, required=False, default=RANDOM_SEED,
                        help='The random seed.')
    args = parser.parse_args()

    random.seed(args.random_seed)
    torch.manual_seed(args.random_seed)
    np.random.seed(args.random_seed)
    torch.cuda.manual_seed(args.random_seed)

    finetuned_dir_name = os.path.normpath(args.output_name)
    if (len(finetuned_dir_name) == 0) or (finetuned_dir_name == '.'):
        err_msg = f'The directory "{finetuned_dir_name}" is incorrect!'
        fredt5_training_logger.error(err_msg)
        raise IOError(err_msg)
    parent_dir_name = os.path.dirname(finetuned_dir_name)
    if not os.path.isdir(parent_dir_name):
        err_msg = f'The directory {parent_dir_name} does not exist!'
        fredt5_training_logger.error(err_msg)
        raise IOError(err_msg)
    if not os.path.isdir(finetuned_dir_name):
        os.mkdir(finetuned_dir_name)

    if args.iters_per_epoch is not None:
        if args.iters_per_epoch < 2:
            err_msg = f'The iterations per epoch is too small. Expected 2 or greater, got {args.iters_per_epoch}.'
            fredt5_training_logger.error(err_msg)
            raise ValueError(err_msg)

    if args.stackoverflow_data is None:
        stackoverflow_data_for_training = None
        stackoverflow_data_for_validation = None
    else:
        fname = os.path.normpath(args.stackoverflow_data)
        if not os.path.isfile(fname):
            err_msg = f'The file "{fname}" does not exist!'
            fredt5_training_logger.error(err_msg)
            raise ValueError(err_msg)
        with codecs.open(fname, mode='r', encoding='utf-8', errors='ignore') as fp:
            stackoverflow_data = json.load(fp)
        stackoverflow_data_for_training = stackoverflow_data['train']
        stackoverflow_data_for_validation = stackoverflow_data['validation']
        del stackoverflow_data
        info_msg = f'The number of stackoverflow samples for training is ' \
                   f'{len(stackoverflow_data_for_training["question_and_answer"])}.'
        fredt5_training_logger.info(info_msg)
        info_msg = f'The number of stackoverflow samples for training with multiply choice is ' \
                   f'{len(stackoverflow_data_for_training["question_and_many_answers"])}.'
        fredt5_training_logger.info(info_msg)

    minibatch_size = args.batch_size
    if minibatch_size <= 0:
        err_msg = f'The mini-batch size {args.batch_size} is wrong!'
        fredt5_training_logger.error(err_msg)
        raise ValueError(err_msg)
    if minibatch_size < 3:
        err_msg = f'The mini-batch size {args.batch_size} is too small! Expected 3 or greater, got {minibatch_size}.'
        fredt5_training_logger.error(err_msg)
        raise ValueError(err_msg)
    fredt5_training_logger.info(f'Mini-batch size is {minibatch_size}.')

    if args.eval_batch_size is None:
        eval_minibatch_size = minibatch_size
    else:
        eval_minibatch_size = args.eval_batch_size
        if eval_minibatch_size <= 0:
            err_msg = f'The mini-batch size {args.eval_batch_size} is wrong!'
            fredt5_training_logger.error(err_msg)
            raise ValueError(err_msg)
        fredt5_training_logger.info(f'Mini-batch size for evaluation is {eval_minibatch_size}.')

    eval_tasks = None
    if args.eval_task is not None:
        eval_tasks = set()
        for it in args.eval_task:
            eval_tasks.add(it.lower().strip())
        fredt5_training_logger.info(f'The evaluation tasks for monitoring are: {eval_tasks}.')

    scorer = (
        LongformerTokenizerFast.from_pretrained(args.eval_model),
        LongformerModel.from_pretrained(args.eval_model)
    )

    dataset_path = os.path.normpath(args.data_name)
    if not os.path.isdir(dataset_path):
        err_msg = f'The directory {dataset_path} does not exist!'
        fredt5_training_logger.error(err_msg)
        raise IOError(err_msg)

    additional_datasets = []
    if args.additional_data_name is not None:
        for it in args.additional_data_name:
            additional_dataset_path = os.path.normpath(it)
            if not os.path.isdir(additional_dataset_path):
                err_msg = f'The directory {additional_dataset_path} does not exist!'
                fredt5_training_logger.error(err_msg)
                raise IOError(err_msg)
            additional_datasets.append(additional_dataset_path)

    pretrained_dir_name = os.path.normpath(args.input_name)
    if not os.path.isdir(pretrained_dir_name):
        err_msg = f'The directory {pretrained_dir_name} does not exist!'
        fredt5_training_logger.error(err_msg)
        raise IOError(err_msg)

    try:
        tokenizer = GPT2Tokenizer.from_pretrained(pretrained_dir_name)
    except Exception as err:
        fredt5_training_logger.error((str(err)))
        raise
    fredt5_training_logger.info(f'The pre-trained tokenizer "{os.path.basename(pretrained_dir_name)}" is loaded.')

    if args.maxtokens is not None:
        stackoverflow_data_for_training = {
            'question_and_answer': list(filter(
                lambda x: len(tokenizer.tokenize(x['question'])) + len(tokenizer.tokenize(x['answer'])) <=
                          args.maxtokens,
                tqdm(stackoverflow_data_for_training['question_and_answer'])
            )),
            'question_and_many_answers': list(filter(
                lambda y: len(tokenizer.tokenize(y['question'])) <= args.maxtokens // 3,
                tqdm(stackoverflow_data_for_training['question_and_many_answers'])
            ))
        }
        info_msg = f'The number of filtered stackoverflow samples for training is ' \
                   f'{len(stackoverflow_data_for_training["question_and_answer"])}.'
        fredt5_training_logger.info(info_msg)
        info_msg = f'The number of filtered stackoverflow samples for training with multiply choice is ' \
                   f'{len(stackoverflow_data_for_training["question_and_many_answers"])}.'
        fredt5_training_logger.info(info_msg)
        stackoverflow_data_for_validation = {
            'question_and_answer': list(filter(
                lambda x: len(tokenizer.tokenize(x['question'])) + len(tokenizer.tokenize(x['answer'])) <=
                          args.maxtokens,
                stackoverflow_data_for_validation['question_and_answer']
            )),
            'question_and_many_answers': list(filter(
                lambda y: len(tokenizer.tokenize(y['question'])) <= args.maxtokens // 3,
                stackoverflow_data_for_validation['question_and_many_answers']
            ))
        }
        info_msg = f'The number of filtered stackoverflow samples for validation is ' \
                   f'{len(stackoverflow_data_for_validation["question_and_answer"])}.'
        fredt5_training_logger.info(info_msg)
        info_msg = f'The number of filtered stackoverflow samples for validation with multiply choice is ' \
                   f'{len(stackoverflow_data_for_validation["question_and_many_answers"])}.'
        fredt5_training_logger.info(info_msg)

    try:
        trainset = load_dataset(dataset_path, split='train')
    except Exception as err:
        fredt5_training_logger.error((str(err)))
        raise
    fredt5_training_logger.info(f'There are {len(trainset)} samples in the training set.')
    if args.maxtokens is not None:
        trainset = trainset.filter(
            lambda it: (len(tokenizer.tokenize(it['input'])) + len(tokenizer.tokenize(it['target']))) <= args.maxtokens
        )
        fredt5_training_logger.info(f'There are {len(trainset)} samples in the training set after filtering.')

    try:
        valset = load_dataset(dataset_path, split='validation')
    except Exception as err:
        fredt5_training_logger.error((str(err)))
        raise
    fredt5_training_logger.info(f'There are {len(valset)} samples in the validation set.')

    input_texts = [str(it) for it in trainset['input']]
    target_texts = [str(it) for it in trainset['target']]
    task_types = [str(it) for it in trainset['task_type']]
    if len(additional_datasets) > 0:
        for it in additional_datasets:
            try:
                additional_trainset = load_dataset(it, split='train')
            except Exception as err:
                fredt5_training_logger.error((str(err)))
                raise
            info_msg = f'There are {len(additional_trainset)} samples in the additional training set ' \
                       f'{os.path.basename(it)}.'
            fredt5_training_logger.info(info_msg)
            if args.maxtokens is not None:
                additional_trainset = additional_trainset.filter(
                    lambda it: (len(tokenizer.tokenize(it['input'])) + len(
                        tokenizer.tokenize(it['target']))) <= args.maxtokens
                )
                info_msg = f'There are {len(additional_trainset)} samples in the additional training set ' \
                           f'{os.path.basename(it)} after filtering.'
                fredt5_training_logger.info(info_msg)
            input_texts += [str(it) for it in additional_trainset['input']]
            target_texts += [str(it) for it in additional_trainset['target']]
            task_types += [os.path.basename(it) for _ in range(len(additional_trainset))]
            del additional_trainset
    data_for_training = []
    all_existed_texts = set()
    for inp, tgt, task in zip(input_texts, target_texts, task_types):
        input_text = inp
        target_text = tgt
        is_positive = is_positive_answer(target_text)
        if args.no_lm_tag:
            if input_text.startswith('<LM>'):
                input_text = input_text[4:]
        else:
            if not input_text.startswith('<LM>'):
                input_text = '<LM>' + input_text
        if not target_text.endswith('</s>'):
            target_text += '</s>'
        data_for_training.append(
            (
                input_text,
                target_text,
                is_positive
            )
        )
        all_existed_texts.add(input_text)
    del input_texts, target_texts, trainset
    random.shuffle(data_for_training)
    sample_weights = calculate_sample_weights(data_for_training)
    gc.collect()

    input_texts = [str(it) for it in valset['input']]
    target_texts = [str(it) for it in valset['target']]
    task_types = [str(it) for it in valset['task_type']]
    if len(additional_datasets) > 0:
        for it in additional_datasets:
            try:
                additional_valset = load_dataset(it, split='validation')
            except Exception as err:
                fredt5_training_logger.error((str(err)))
                raise
            info_msg = f'There are {len(additional_valset)} samples in the additional validation set ' \
                       f'{os.path.basename(it)}.'
            fredt5_training_logger.info(info_msg)
            input_texts += [str(it) for it in additional_valset['input']]
            target_texts += [str(it) for it in additional_valset['target']]
            task_types += [os.path.basename(it) for _ in range(len(additional_valset))]
            del additional_valset
    data_for_validation = dict()
    for val in zip(input_texts, target_texts, task_types):
        input_text, target_text, task_type = val
        if args.no_lm_tag:
            if input_text.startswith('<LM>'):
                input_text = input_text[4:]
        else:
            if not input_text.startswith('<LM>'):
                input_text = '<LM>' + input_text
        if not target_text.endswith('</s>'):
            target_text += '</s>'
        if eval_tasks is None:
            can_add = True
        else:
            can_add = (task_type in eval_tasks)
        if can_add:
            if task_type in data_for_validation:
                data_for_validation[task_type].append((input_text, target_text))
            else:
                data_for_validation[task_type] = [(input_text, target_text)]
        all_existed_texts.add(input_text)
    del input_texts, target_texts, task_types, valset
    if stackoverflow_data_for_validation is not None:
        if eval_tasks is None:
            can_add = True
        else:
            can_add = ('stackoverflow' in eval_tasks)
        if can_add:
            task_type = 'stackoverflow_answer'
            for it in stackoverflow_data_for_validation['question_and_answer']:
                input_text = it['question']
                target_text = it['answer']
                if args.no_lm_tag:
                    if input_text.startswith('<LM>'):
                        input_text = input_text[4:]
                else:
                    if not input_text.startswith('<LM>'):
                        input_text = '<LM>' + input_text
                if not target_text.endswith('</s>'):
                    target_text += '</s>'
                if task_type in data_for_validation:
                    data_for_validation[task_type].append((input_text, target_text))
                else:
                    data_for_validation[task_type] = [(input_text, target_text)]
                all_existed_texts.add(input_text)
            task_type = 'stackoverflow_choice'
            for it in stackoverflow_data_for_validation['question_and_many_answers']:
                input_text, target_text = generate_answer_choice_on_stackoverflow(it)
                if args.no_lm_tag:
                    if input_text.startswith('<LM>'):
                        input_text = input_text[4:]
                else:
                    if not input_text.startswith('<LM>'):
                        input_text = '<LM>' + input_text
                if not target_text.endswith('</s>'):
                    target_text += '</s>'
                if task_type in data_for_validation:
                    data_for_validation[task_type].append((input_text, target_text))
                else:
                    data_for_validation[task_type] = [(input_text, target_text)]
                all_existed_texts.add(input_text)
    tasks = sorted(list(data_for_validation.keys()))
    fredt5_training_logger.info(f'There are {len(tasks)} validation task types.')

    if args.augment:
        augmenters_list = [
            CharAug(
                unit_prob=0.1,
                min_aug=1,
                max_aug=5,
                mult_num=3,
                lang='rus',
                platform='pc',
                random_seed=args.random_seed
            ),
            CharAug(
                unit_prob=0.1,
                min_aug=1,
                max_aug=5,
                mult_num=3,
                lang='rus',
                platform='mobile',
                random_seed=args.random_seed
            )
        ]
    else:
        augmenters_list = None
        fredt5_training_logger.info('Augmentations are not used.')

    try:
        model = T5ForConditionalGeneration.from_pretrained(
            pretrained_dir_name,
            attention_type='flash',
            use_triton=True
        ).to(device)
    except Exception as err:
        fredt5_training_logger.error((str(err)))
        raise
    model.eval()
    fredt5_training_logger.info(f'The pre-trained model "{os.path.basename(pretrained_dir_name)}" is loaded.')

    tokenizer.save_pretrained(finetuned_dir_name)
    max_text_len = max([len(it[1]) for it in data_for_training])
    n_training_samples = len(data_for_training)
    for task in tasks:
        max_text_len_ = max([len(it[1]) for it in data_for_validation[task]])
        info_msg = f'Task {task} contains {len(data_for_validation[task])} validation samples.'
        fredt5_training_logger.info(info_msg)
        if max_text_len_ > max_text_len:
            max_text_len = max_text_len_
    fredt5_training_logger.info(f'The maximal subwords in the generated text is {max_text_len}.')

    generation_max_length = 3 + round(1.1 * max_text_len)
    generation_config = GenerationConfig(
        top_k=10,
        penalty_alpha=0.6,
        max_length=generation_max_length,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
        decoder_start_token_id=tokenizer.pad_token_id
    )
    generation_config.save_pretrained(finetuned_dir_name)
    fredt5_training_logger.info(f'{generation_config}')

    if args.algorithm == 'adamw':
        try:
            optimizer = torch.optim.AdamW(
                params=[p for p in model.parameters() if p.requires_grad],
                weight_decay=1e-1,
                lr=args.learning_rate,
                fused=False
            )
        except:
            optimizer = torch.optim.AdamW(
                params=[p for p in model.parameters() if p.requires_grad],
                weight_decay=1e-1,
                lr=args.learning_rate
            )
    elif args.algorithm == 'rmsprop':
        optimizer = torch.optim.RMSprop(
            params=[p for p in model.parameters() if p.requires_grad],
            lr=args.learning_rate
        )
    elif args.algorithm == 'sgd':
        try:
            optimizer = torch.optim.SGD(
                params=[p for p in model.parameters() if p.requires_grad],
                lr=args.learning_rate,
                fused=False
            )
        except:
            optimizer = torch.optim.SGD(
                params=[p for p in model.parameters() if p.requires_grad],
                lr=args.learning_rate
            )
    else:
        optimizer = bnb.optim.Adam8bit(
            params=[p for p in model.parameters() if p.requires_grad],
            lr=args.learning_rate
        )
    max_epochs = 200

    n_training_batches = int(np.ceil(n_training_samples / minibatch_size))
    max_iters = max_epochs * n_training_batches
    if args.iters_per_epoch is not None:
        n_training_batches = args.iters_per_epoch
        max_epochs = max(max_iters // n_training_batches, 3)
    fredt5_training_logger.info(f'Number of epochs is {max_epochs}. Iterations per epoch is {n_training_batches}.')

    scheduler = torch.optim.lr_scheduler.CyclicLR(
        optimizer,
        base_lr=args.learning_rate,
        max_lr=10.0 * args.learning_rate,
        step_size_up=min(3 * n_training_batches, 3000),
        cycle_momentum=False,
        mode='triangular2'
    )

    if args.no_pre_eval:
        best_score = None
    else:
        scores = []
        for task in tasks:
            try:
                eval_score, results_by_tasks = evaluate_any_task(data_for_validation[task],
                                                                 tokenizer, generation_config, model,
                                                                 eval_minibatch_size, scorer,
                                                                 max_time=MAX_GENERATION_TIME)
            except Exception as err:
                fredt5_training_logger.error(str(err))
                raise
            del results_by_tasks
            scores.append(eval_score)
            fredt5_training_logger.info(f'Before training: BERT score for task {task} is {round(eval_score, 6)}.')
            torch.cuda.empty_cache()
            gc.collect()
        best_score = sum(scores) / len(scores)
        del scores
        fredt5_training_logger.info(f'Before training: mean BERT score is {round(best_score, 6)}.')

    for epoch in range(1, max_epochs + 1):
        fredt5_training_logger.info(f'Epoch {epoch} is started.')
        model.train()
        total_training_loss_val = 0.0
        for _ in trange(n_training_batches):
            samples_in_batch = generate_samples_for_minibatch(
                data_for_training, sample_weights, tokenizer,
                arithmetics=args.arithmetics,
                augmenters=augmenters_list, existed_texts=all_existed_texts,
                use_lm_tag=not args.no_lm_tag, minibatch_size=minibatch_size,
                stackoverflow_data=stackoverflow_data_for_training
            )
            for input_sequence, output_sequence in samples_in_batch:
                x_input_ids_ = [torch.tensor(input_sequence, dtype=torch.long)]
                x_attention_mask_ = [torch.tensor([1 for _ in range(len(input_sequence))], dtype=torch.long)]
                y_input_ids_ = [torch.tensor(output_sequence, dtype=torch.long)]
                y_attention_mask_ = [torch.tensor([1 for _ in range(len(output_sequence))], dtype=torch.long)]
                x_input_ids = torch.nn.utils.rnn.pad_sequence(
                    x_input_ids_,
                    batch_first=True,
                    padding_value=tokenizer.pad_token_id
                ).to(device)
                x_attention_mask = torch.nn.utils.rnn.pad_sequence(
                    x_attention_mask_,
                    batch_first=True,
                    padding_value=0
                ).to(device)
                y_input_ids = torch.nn.utils.rnn.pad_sequence(
                    y_input_ids_,
                    batch_first=True,
                    padding_value=-100
                ).to(device)
                y_attention_mask = torch.nn.utils.rnn.pad_sequence(
                    y_attention_mask_,
                    batch_first=True,
                    padding_value=0
                ).to(device)
                del x_input_ids_, y_input_ids_
                del x_attention_mask_, y_attention_mask_
                loss = model(
                    input_ids=x_input_ids,
                    attention_mask=x_attention_mask,
                    labels=y_input_ids,
                    decoder_attention_mask=y_attention_mask,
                    return_dict=True
                ).loss / len(samples_in_batch)
                total_training_loss_val += float(loss.detach().cpu())
                loss.backward()
                del x_input_ids, y_input_ids
                del x_attention_mask, y_attention_mask
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            torch.cuda.empty_cache()
        total_training_loss_val /= float(n_training_batches)
        info_msg = f'Epoch {epoch}: total training loss is {total_training_loss_val}.'
        fredt5_training_logger.info(info_msg)
        model.eval()
        gc.collect()
        torch.cuda.empty_cache()
        scores = []
        for task in tasks:
            try:
                eval_score, results_by_tasks = evaluate_any_task(data_for_validation[task],
                                                                 tokenizer, generation_config, model,
                                                                 eval_minibatch_size, scorer,
                                                                 max_time=MAX_GENERATION_TIME)
            except Exception as err:
                fredt5_training_logger.error(str(err))
                raise
            del results_by_tasks
            scores.append(eval_score)
            fredt5_training_logger.info(f'Epoch {epoch}: BERT score for task {task} is {round(eval_score, 6)}.')
            torch.cuda.empty_cache()
            gc.collect()
        new_score = sum(scores) / len(scores)
        del scores
        fredt5_training_logger.info(f'Epoch {epoch}: mean BERT score is {round(new_score, 6)}.')
        if best_score is None:
            updated = True
        else:
            updated = (new_score > best_score)
        if updated:
            best_score = new_score
            model.save_pretrained(save_directory=finetuned_dir_name, safe_serialization=False)
            model.save_pretrained(save_directory=finetuned_dir_name, safe_serialization=True)
            generation_config.save_pretrained(finetuned_dir_name)
            fredt5_training_logger.info(f'Epoch {epoch}: the model is updated with score = {best_score}.')


if __name__ == '__main__':
    fredt5_training_logger.setLevel(logging.INFO)
    fmt_str = '%(filename)s[LINE:%(lineno)d]# %(levelname)-8s ' \
              '[%(asctime)s]  %(message)s'
    formatter = logging.Formatter(fmt_str)
    stdout_handler = logging.StreamHandler(sys.stdout)
    stdout_handler.setFormatter(formatter)
    fredt5_training_logger.addHandler(stdout_handler)
    file_handler = logging.FileHandler('fredt5_instruct_training.log')
    file_handler.setFormatter(formatter)
    fredt5_training_logger.addHandler(file_handler)
    main()
