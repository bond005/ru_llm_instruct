from argparse import ArgumentParser
import codecs
import json
import logging
import math
import os
import random
import sys
from typing import List, Tuple

import numpy as np
from tqdm import tqdm, trange
from transformers import T5ForConditionalGeneration, GPT2Tokenizer, Adafactor, GenerationConfig
import torch

from instructions.instructions import evaluate
from training.training import sample_batch
from training.training import training_logger


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
    parser.add_argument('-i', '--input', dest='input_name', type=str, required=True,
                        help='The input name of pre-trained FRED-T5.')
    parser.add_argument('-o', '--output', dest='output_name', type=str, required=True,
                        help='The output name of FRED-T5 after fine-tuning.')
    parser.add_argument('-d', '--data', dest='data_name', type=str, required=True,
                        help='The JSON structured dataset name.')
    parser.add_argument('--batch', dest='batch_size', type=int, required=True,
                        help='The mini-batch size for FRED-T5 training.')
    parser.add_argument('--eval_batch', dest='eval_batch_size', type=int, required=False, default=None,
                        help='The mini-batch size for FRED-T5 evaluation.')
    parser.add_argument('--accum', dest='gradient_accumulation', type=int, required=False, default=None,
                        help='The gradient accumulation for FRED-T5 training.')
    parser.add_argument('--lr', dest='learning_rate', type=float, required=False, default=3e-4,
                        help='The learning rate.')
    parser.add_argument('--trainsize', dest='trainsize', type=int, required=False, default=None,
                        help='The samples per training epoch.')
    parser.add_argument('--testsize', dest='testsize', type=int, required=False, default=None,
                        help='The maximal number of validation samples per validated task.')
    parser.add_argument('--train_maxlen', dest='train_maxlen', type=int, required=False, default=None,
                        help='The maximal number of tokens per input or target for training.')
    parser.add_argument('--test_maxlen', dest='test_maxlen', type=int, required=False, default=None,
                        help='The maximal number of tokens per input or target for testing.')
    parser.add_argument('--bf16', dest='bf16', action='store_true', help='Is bfloat16 used?')
    args = parser.parse_args()

    finetuned_dir_name = os.path.normpath(args.output_name)
    if (len(finetuned_dir_name) == 0) or (finetuned_dir_name == '.'):
        err_msg = f'The directory "{finetuned_dir_name}" is incorrect!'
        fredt5_logger.error(err_msg)
        raise IOError(err_msg)
    parent_dir_name = os.path.dirname(finetuned_dir_name)
    if not os.path.isdir(parent_dir_name):
        err_msg = f'The directory {parent_dir_name} does not exist!'
        fredt5_logger.error(err_msg)
        raise IOError(err_msg)
    if not os.path.isdir(finetuned_dir_name):
        os.mkdir(finetuned_dir_name)

    minibatch_size = args.batch_size
    if minibatch_size <= 0:
        err_msg = f'The mini-batch size {args.batch_size} is wrong!'
        fredt5_logger.error(err_msg)
        raise ValueError(err_msg)
    fredt5_logger.info(f'Mini-batch size is {minibatch_size}.')

    if args.eval_batch_size is None:
        eval_minibatch_size = minibatch_size
    else:
        eval_minibatch_size = args.eval_batch_size
        if eval_minibatch_size <= 0:
            err_msg = f'The mini-batch size {args.eval_batch_size} is wrong!'
            fredt5_logger.error(err_msg)
            raise ValueError(err_msg)
        fredt5_logger.info(f'Mini-batch size for evaluation is {eval_minibatch_size}.')

    if args.gradient_accumulation is None:
        gradient_accumulation = 1
    else:
        gradient_accumulation = args.gradient_accumulation
        if gradient_accumulation <= 0:
            err_msg = f'The gradient accumulation {args.gradient_accumulation} is wrong!'
            fredt5_logger.error(err_msg)
            raise ValueError(err_msg)
        if gradient_accumulation > 1:
            fredt5_logger.info(f'Gradient accumulation step size is {gradient_accumulation}.')

    if args.train_maxlen is None:
        maximal_number_of_tokens_for_training = None
    else:
        maximal_number_of_tokens_for_training = args.train_maxlen
        if maximal_number_of_tokens_for_training < 10:
            err_msg = (f'The maximal number of tokens per input or target is too small! Expected 10 or greater, '
                       f'got {maximal_number_of_tokens_for_training}.')
            fredt5_logger.error(err_msg)
            raise ValueError(err_msg)

    if args.test_maxlen is None:
        maximal_number_of_tokens_for_testing = None
    else:
        maximal_number_of_tokens_for_testing = args.test_maxlen
        if maximal_number_of_tokens_for_testing < 10:
            err_msg = (f'The maximal number of tokens per input or target is too small! Expected 10 or greater, '
                       f'got {maximal_number_of_tokens_for_testing}.')
            fredt5_logger.error(err_msg)
            raise ValueError(err_msg)

    dataset_fname = os.path.normpath(args.data_name)
    if not os.path.isfile(dataset_fname):
        err_msg = f'The file {dataset_fname} does not exist!'
        fredt5_logger.error(err_msg)
        raise IOError(err_msg)

    pretrained_dir_name = os.path.normpath(args.input_name)
    if not os.path.isdir(pretrained_dir_name):
        err_msg = f'The directory {pretrained_dir_name} does not exist!'
        fredt5_logger.error(err_msg)
        raise IOError(err_msg)

    tokenizer = GPT2Tokenizer.from_pretrained(pretrained_dir_name)
    fredt5_logger.info(f'The pre-trained tokenizer "{os.path.basename(pretrained_dir_name)}" is loaded.')

    with codecs.open(dataset_fname, mode='r', encoding='utf-8', errors='ignore') as fp:
        data = json.load(fp)
    if not isinstance(data, dict):
        err_msg = f'The file "{dataset_fname}" contains a wrong data!'
        fredt5_logger.error(err_msg)
        raise IOError(err_msg)
    if set(data.keys()) != {'text_corpus', 'validation', 'train'}:
        err_msg = f'The file "{dataset_fname}" contains a wrong data!'
        fredt5_logger.error(err_msg)
        raise IOError(err_msg)
    united_text_corpus = data['text_corpus']
    data_for_validation = data['validation']
    data_for_training = data['train']

    n_training_samples = 0
    tasks_for_training = sorted(list(data_for_training.keys()))
    fredt5_logger.info(f'There are {len(tasks_for_training)} tasks for training.')
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
                fredt5_logger.error(err_msg)
                raise ValueError(err_msg)
        fredt5_logger.info(f'There are {len(data_for_training[cur_task])} training samples for task {cur_task}.')
        n_training_samples += len(data_for_training[cur_task])
    fredt5_logger.info(f'The total number of training samples is {n_training_samples}.')

    tasks_for_validation = sorted(list(data_for_validation.keys()))
    fredt5_logger.info(f'There are {len(tasks_for_validation)} tasks for validation.')
    if set(tasks_for_training) != set(tasks_for_validation):
        err_msg = (f'The training tasks do not correspond to the validation tasks! '
                   f'{tasks_for_training} != {tasks_for_validation}')
        fredt5_logger.error(err_msg)
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
                fredt5_logger.error(err_msg)
                raise ValueError(err_msg)
        fredt5_logger.info(f'There are {len(data_for_validation[cur_task])} validation samples for task {cur_task}.')

    if args.bf16:
        model = T5ForConditionalGeneration.from_pretrained(pretrained_dir_name, torch_dtype=torch.bfloat16).to(device)
    else:
        model = T5ForConditionalGeneration.from_pretrained(pretrained_dir_name, torch_dtype=torch.float32).to(device)
    model.eval()
    fredt5_logger.info(f'The pre-trained model "{os.path.basename(pretrained_dir_name)}" is loaded.')

    tokenizer.save_pretrained(finetuned_dir_name)
    max_text_len = max([len(tokenizer.tokenize(it)) for it in tqdm(united_text_corpus)])
    fredt5_logger.info(f'The maximal subwords in the text is {max_text_len}.')

    if maximal_number_of_tokens_for_testing is None:
        generation_max_length = 3 + round(1.1 * max_text_len)
    else:
        generation_max_length = 3 + round(1.1 * maximal_number_of_tokens_for_testing)
    generation_config = GenerationConfig(
        top_k=10,
        penalty_alpha=0.6,
        max_length=generation_max_length,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
        decoder_start_token_id=tokenizer.pad_token_id
    )
    generation_config.save_pretrained(finetuned_dir_name)
    fredt5_logger.info(f'{generation_config}')

    optimizer = Adafactor(
        params=[p for p in model.parameters() if p.requires_grad],
        scale_parameter=False,
        relative_step=False,
        warmup_init=False,
        lr=args.learning_rate,
        clip_threshold=1.0
    )
    max_epochs = 200

    fredt5_logger.info(f'Tokenization of training texts is started.')
    data_for_training_ = dict()
    for cur_task in tqdm(tasks_for_validation):
        tokenized: List[Tuple[List[int], List[int]]] = []
        for text_pair in data_for_training[cur_task]:
            tokenized_input = tokenizer.encode(text=text_pair[0], add_special_tokens=False)
            tokenized_target = tokenizer.encode(text=text_pair[1], add_special_tokens=False)
            tokenized.append((tokenized_input, tokenized_target))
        del data_for_training[cur_task]
        data_for_training_[cur_task] = tokenized
        del tokenized
    del data_for_training
    fredt5_logger.info(f'All training texts are tokenized.')

    n_training_batches = int(np.ceil(n_training_samples / (minibatch_size * gradient_accumulation)))
    if args.trainsize is not None:
        if args.trainsize < 100:
            err_msg = f'The samples per training epoch is too small! Expected 100 or greater, got {args.trainsize}.'
            fredt5_logger.error(err_msg)
            raise ValueError(err_msg)
        if args.trainsize < n_training_samples:
            n_training_batches = max(10, int(np.ceil(args.trainsize / (minibatch_size * gradient_accumulation))))
            max_epochs = round(max_epochs * (n_training_samples / args.trainsize))
    fredt5_logger.info(f'Number of epochs is {max_epochs}. Iterations per epoch is {n_training_batches}.')

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
                                                tokenizer, generation_config, model, eval_minibatch_size)
    except Exception as err:
        fredt5_logger.error(str(err))
        raise
    fredt5_logger.info(f'United recognition score before training is {best_score}.')
    for cur_task in tasks_for_validation:
        info_msg = f'Before training: the task {cur_task} '
        instant_score, _ = results_by_tasks[cur_task]
        if cur_task == 'asr_correction':
            info_msg += 'word accuracy is {0:.5%}.'.format(instant_score)
        elif cur_task == 'segmentation':
            info_msg += 'paragraph accuracy is {0:.5%}.'.format(instant_score)
        elif cur_task.startswith('ner_'):
            info_msg += 'F1 by entities is {0:.6f}.'.format(instant_score)
        elif cur_task.endswith('_detection'):
            info_msg += 'Yes/No F1 is {0:.6f}.'.format(instant_score)
        else:
            info_msg += 'ChrF score F1 is {0:.6f}.'.format(instant_score)
        fredt5_logger.info(info_msg)

    for epoch in range(1, max_epochs + 1):
        fredt5_logger.info(f'Epoch {epoch} is started.')
        model.train()
        train_loss_val = 0.0
        for _ in trange(n_training_batches):
            try:
                x_input_ids, x_attention_mask, y_input_ids, y_attention_mask = sample_batch(
                    data_for_training_,
                    tokenizer.pad_token_id,
                    minibatch_size * gradient_accumulation
                )
            except Exception as err:
                fredt5_logger.error(str(err))
                raise
            for batch_idx in range(gradient_accumulation):
                batch_start = batch_idx * minibatch_size
                batch_end = batch_start + minibatch_size
                loss = model(
                    input_ids=x_input_ids[batch_start:batch_end].to(device),
                    attention_mask=x_attention_mask[batch_start:batch_end].to(device),
                    labels=y_input_ids[batch_start:batch_end].to(device),
                    decoder_attention_mask=y_attention_mask[batch_start:batch_end].to(device),
                    return_dict=True
                ).loss / gradient_accumulation
                train_loss_val += float(loss.detach().cpu())
                loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        train_loss_val /= n_training_batches
        fredt5_logger.info(f'Epoch {epoch}: training loss = {train_loss_val}.')
        model.eval()
        torch.cuda.empty_cache()
        try:
            cur_score, results_by_tasks = evaluate(data_for_validation,
                                                   tokenizer, generation_config, model, eval_minibatch_size)
        except Exception as err:
            fredt5_logger.error(str(err))
            raise
        fredt5_logger.info(f'Epoch {epoch}: united recognition score is {cur_score}.')
        for cur_task in tasks_for_validation:
            info_msg = f'Epoch {epoch}: the task {cur_task} '
            instant_score, _ = results_by_tasks[cur_task]
            if cur_task == 'asr_correction':
                info_msg += 'word accuracy is {1:.5%}.'.format(epoch, instant_score)
            elif cur_task == 'segmentation':
                info_msg += 'paragraph accuracy is {1:.5%}.'.format(epoch, instant_score)
            elif cur_task.startswith('ner_'):
                info_msg += 'F1 by entities is {1:.6f}.'.format(epoch, instant_score)
            elif cur_task.endswith('_detection'):
                info_msg += 'Yes/No F1 is {1:.6f}.'.format(epoch, instant_score)
            else:
                info_msg += 'ChrF score F1 is {1:.6f}.'.format(epoch, instant_score)
            fredt5_logger.info(info_msg)
        if cur_score > best_score:
            best_score = cur_score
            model.save_pretrained(save_directory=finetuned_dir_name, safe_serialization=False)
            model.save_pretrained(save_directory=finetuned_dir_name, safe_serialization=True)
            generation_config.save_pretrained(finetuned_dir_name)
            fredt5_logger.info(f'The model is updated with score = {best_score}.')


if __name__ == '__main__':
    fredt5_logger.setLevel(logging.INFO)
    training_logger.setLevel(logging.INFO)
    fmt_str = '%(filename)s[LINE:%(lineno)d]# %(levelname)-8s ' \
              '[%(asctime)s]  %(message)s'
    formatter = logging.Formatter(fmt_str)
    stdout_handler = logging.StreamHandler(sys.stdout)
    stdout_handler.setFormatter(formatter)
    fredt5_logger.addHandler(stdout_handler)
    training_logger.addHandler(stdout_handler)
    file_handler = logging.FileHandler('fredt5_instruct_training.log')
    file_handler.setFormatter(formatter)
    fredt5_logger.addHandler(file_handler)
    training_logger.addHandler(file_handler)
    main()
