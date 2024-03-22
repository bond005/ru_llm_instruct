import copy
import math
from multiprocessing import Pool
import os
import random
from typing import Dict, List, Tuple

from bert_score import bert_cos_score_idf, get_idf_dict
from nltk import wordpunct_tokenize
import numpy as np
from seqeval.metrics import f1_score
from scipy.stats import hmean
import torch
from tqdm import trange
from transformers import GPT2Tokenizer, GenerationConfig, T5ForConditionalGeneration, T5EncoderModel

from inference.inference import generate_answer, fix_recognition_error
from ner.ner import find_entities_in_text
from utils.utils import calculate_word_error_rate, process_target


KNOWN_TASKS = [
    (
        'Исправь, пожалуйста, ошибки распознавания речи в следующем тексте.',
        'asr_correction'
    ),
    (
        'Выполни саммаризацию и выдели, пожалуйста, основную мысль следующего текста.',
        'summarization'
    ),
    (
        'Разбей, пожалуйста, следующий текст на абзацы.',
        'segmentation'
    ),
    (
        'Упрости, пожалуйста, следующий текст.',
        'simplification'
    ),
    (
        'Найди, пожалуйста, все именованные сущности типа "Организация" в следующем тексте '
        'и выпиши список таких сущностей.',
        'ner_organization'
    ),
    (
        'Найди, пожалуйста, все именованные сущности типа "Человек" в следующем тексте '
        'и выпиши список таких сущностей.',
        'ner_person'
    ),
    (
        'Найди, пожалуйста, все именованные сущности типа "Местоположение" в следующем тексте '
        'и выпиши список таких сущностей.',
        'ner_location'
    )
]


def evaluate_asr_correction(data_for_validation: List[Tuple[str, str]], tokenizer: GPT2Tokenizer,
                            config: GenerationConfig, model: T5ForConditionalGeneration,
                            minibatch: int) -> Tuple[float, List[Dict[str, str]]]:
    printed_results = []
    arguments = []
    n_batches = math.ceil(len(data_for_validation) / minibatch)
    for batch_idx in trange(n_batches):
        batch_start = batch_idx * minibatch
        batch_end = min(len(data_for_validation), batch_start + minibatch)
        input_texts = [it[0] for it in data_for_validation[batch_start:batch_end]]
        target_texts = [process_target(it[1]) for it in data_for_validation[batch_start:batch_end]]
        predicted_texts = fix_recognition_error(input_texts, tokenizer, config, model)
        if len(predicted_texts) != len(input_texts):
            err_msg = f'The predicted texts do not correspond to the input texts! {predicted_texts} != {input_texts}'
            raise ValueError(err_msg)
        for input_, target_, predicted_ in zip(input_texts, target_texts, predicted_texts):
            printed_results.append({'INPUT': input_, 'PREDICTED': predicted_, 'TRUE': target_})
            arguments.append((wordpunct_tokenize(predicted_), wordpunct_tokenize(target_)))
    with Pool(processes=max(1, os.cpu_count())) as pool:
        res = pool.starmap(calculate_word_error_rate, arguments)
    del arguments
    n_total_word_dist = 0
    n_total_words = 0
    for cur_dist, cur_word_number in res:
        n_total_word_dist += cur_dist
        n_total_words += cur_word_number
    if n_total_words > 0:
        wer = n_total_word_dist / float(n_total_words)
    else:
        wer = 0.0
    del res
    return 1.0 - wer, printed_results


def evaluate_segmentation(data_for_validation: List[Tuple[str, str]], tokenizer: GPT2Tokenizer,
                          config: GenerationConfig, model: T5ForConditionalGeneration, minibatch: int) -> (
        Tuple)[float, List[Dict[str, str]]]:
    printed_results = []
    arguments = []
    n_batches = math.ceil(len(data_for_validation) / minibatch)
    for batch_idx in trange(n_batches):
        batch_start = batch_idx * minibatch
        batch_end = min(len(data_for_validation), batch_start + minibatch)
        input_texts = [it[0] for it in data_for_validation[batch_start:batch_end]]
        target_texts = [process_target(it[1]) for it in data_for_validation[batch_start:batch_end]]
        predicted_texts = generate_answer(input_texts, tokenizer, config, model)
        for input_, target_, predicted_ in zip(input_texts, target_texts, predicted_texts):
            printed_results.append({'INPUT': input_, 'PREDICTED': predicted_, 'TRUE': target_})
            predicted_paragraphs = list(map(
                lambda it3: ' '.join(list(filter(lambda x: x.isalnum(), wordpunct_tokenize(it3)))).strip(),
                filter(
                    lambda it2: len(it2) > 0,
                    map(lambda it1: it1.strip(), predicted_.split('\n'))
                )
            ))
            target_paragraphs = list(map(
                lambda it3: ' '.join(list(filter(lambda x: x.isalnum(), wordpunct_tokenize(it3)))).strip(),
                filter(
                    lambda it2: len(it2) > 0,
                    map(lambda it1: it1.strip(), target_.split('\n'))
                )
            ))
            arguments.append((predicted_paragraphs, target_paragraphs))
    with Pool(processes=max(1, os.cpu_count())) as pool:
        res = pool.starmap(calculate_word_error_rate, arguments)
    del arguments
    n_total_paragraph_dist = 0
    n_total_paragraphs = 0
    for cur_dist, cur_paragraph_number in res:
        n_total_paragraph_dist += cur_dist
        n_total_paragraphs += cur_paragraph_number
    if n_total_paragraphs > 0:
        per = n_total_paragraph_dist / float(n_total_paragraphs)
    else:
        per = 0.0
    del res
    return 1.0 - per, printed_results


def evaluate_ner(data_for_validation: List[Tuple[str, str]], entity_class: str, tokenizer: GPT2Tokenizer,
                 config: GenerationConfig, model: T5ForConditionalGeneration,
                 minibatch: int) -> Tuple[float, List[Dict[str, str]]]:
    printed_results = []
    prompt_tail = ' и выпиши список таких сущностей.'
    n_batches = math.ceil(len(data_for_validation) / minibatch)
    for batch_idx in trange(n_batches):
        batch_start = batch_idx * minibatch
        batch_end = min(len(data_for_validation), batch_start + minibatch)
        input_texts = [it[0] for it in data_for_validation[batch_start:batch_end]]
        target_texts = [process_target(it[1]) for it in data_for_validation[batch_start:batch_end]]
        predicted_texts = generate_answer(input_texts, tokenizer, config, model)
        for input_, target_, predicted_ in zip(input_texts, target_texts, predicted_texts):
            found_idx = input_.find(prompt_tail)
            if found_idx < 0:
                raise ValueError(f'The text "{input_}" has not a correct prompt!')
            input_text_without_prompt = input_[(found_idx + len(prompt_tail)):].strip()
            predicted_text_ = ' '.join(list(filter(lambda x: x.isalnum(), wordpunct_tokenize(predicted_.lower()))))
            if predicted_text_.find('в этом тексте нет именованных сущностей такого типа') >= 0:
                predicted_named_entities = []
            else:
                predicted_named_entities = list(map(
                    lambda it3: it3.strip(),
                    filter(
                        lambda it2: len(it2) > 0,
                        map(lambda it1: it1.strip(), predicted_.split('\n'))
                    )
                ))
            target_text_ = ' '.join(list(filter(lambda x: x.isalnum(), wordpunct_tokenize(target_.lower()))))
            if target_text_.find('в этом тексте нет именованных сущностей такого типа') >= 0:
                target_named_entities = []
            else:
                target_named_entities = list(map(
                    lambda it3: it3.strip(),
                    filter(
                        lambda it2: len(it2) > 0,
                        map(lambda it1: it1.strip(), target_.split('\n'))
                    )
                ))
            predicted_named_entities = find_entities_in_text(
                input_text_without_prompt, predicted_named_entities, entity_class
            )
            target_named_entities = find_entities_in_text(
                input_text_without_prompt, target_named_entities, entity_class, raise_exception=True
            )
            if len(target_named_entities) != len(predicted_named_entities):
                err_msg = (
                    f'The target named entities do not correspond to the text! Text: "{input_text_without_prompt}". '
                    f'Entities: {target_named_entities}')
                raise ValueError(err_msg)
            printed_results.append({
                'INPUT': input_,
                'PREDICTED': predicted_named_entities,
                'TRUE': target_named_entities
            })
    y_true = [[x[1] for x in cur['TRUE']] for cur in printed_results]
    y_pred = [[x[1] for x in cur['PREDICTED']] for cur in printed_results]
    f1 = f1_score(y_true, y_pred)
    printed_results = [{'INPUT': it['INPUT'], 'PREDICTED': f'{it["PREDICTED"]}', 'TRUE': f'{it["TRUE"]}'}
                       for it in printed_results]
    return f1, printed_results


def evaluate_any_task(data_for_validation: List[Tuple[str, str]], tokenizer: GPT2Tokenizer,
                      config: GenerationConfig, model: T5ForConditionalGeneration, minibatch: int,
                      scorer: Tuple[GPT2Tokenizer, T5EncoderModel, int, List[str]]) -> (
        Tuple)[float, List[Dict[str, str]]]:
    printed_results = []
    candidates = []
    references = []
    texts_for_idf = copy.copy(scorer[3])
    n_batches = math.ceil(len(data_for_validation) / minibatch)
    for batch_idx in trange(n_batches):
        batch_start = batch_idx * minibatch
        batch_end = min(len(data_for_validation), batch_start + minibatch)
        input_texts = [it[0] for it in data_for_validation[batch_start:batch_end]]
        target_texts = [process_target(it[1]) for it in data_for_validation[batch_start:batch_end]]
        predicted_texts = generate_answer(input_texts, tokenizer, config, model)
        for input_, target_, predicted_ in zip(input_texts, target_texts, predicted_texts):
            texts_for_idf.append(predicted_)
            candidates.append(predicted_)
            if len(target_) == 0:
                err_msg = f'The evaluation pair ({input_}, {target_}) is wrong, because the target is empty!'
                raise ValueError(err_msg)
            references.append(target_)
            printed_results.append({'INPUT': input_, 'PREDICTED': predicted_, 'TRUE': target_})
    if len(printed_results) > 5:
        printed_results = random.sample(printed_results, k=5)
    idf_dict = get_idf_dict(texts_for_idf, scorer[0], nthreads=max(1, os.cpu_count()))
    del texts_for_idf
    try:
        all_preds = bert_cos_score_idf(
            model=scorer[1],
            refs=references,
            hyps=candidates,
            tokenizer=scorer[0],
            idf_dict=idf_dict,
            device=scorer[1].device,
            batch_size=scorer[2],
            all_layers=False
        ).cpu()
    except:
        indices = random.sample(population=list(range(len(references))), k=5)
        for idx in indices:
            print('')
            print('TRUE:')
            print(references[idx])
            print('PREDICTED:')
            print(candidates[idx])
        raise
    f1_list = all_preds[..., 2].numpy().tolist()
    if len(references) != len(f1_list):
        err_msg = f'The true answers do not correspond to the BERT scores! {len(references)} != {len(f1_list)}.'
        raise ValueError(err_msg)
    f1_mean = float(np.mean(f1_list))
    del idf_dict, f1_list, references, candidates
    return f1_mean, printed_results


def evaluate(data_for_validation: Dict[str, List[Tuple[str, str]]],
             tokenizer: GPT2Tokenizer, config: GenerationConfig, model: T5ForConditionalGeneration, minibatch: int,
             intelligent_scorer: Tuple[GPT2Tokenizer, T5EncoderModel, int, List[str]]) -> (
        Tuple)[float, Dict[str, Tuple[float, List[Dict[str, str]]]]]:
    res = dict()
    scores = []
    for task in data_for_validation:
        if task == 'asr_correction':
            res[task] = evaluate_asr_correction(data_for_validation[task], tokenizer, config, model, minibatch)
        elif task == 'segmenation':
            res[task] = evaluate_segmentation(data_for_validation[task], tokenizer, config, model, minibatch)
        elif task.startswith('ner_'):
            entity_class = task[4:].lower()
            res[task] = evaluate_ner(data_for_validation[task], entity_class, tokenizer, config, model, minibatch)
        else:
            res[task] = evaluate_any_task(data_for_validation[task], tokenizer, config, model, minibatch,
                                          intelligent_scorer)
        scores.append(max(res[task][0], 1e-9))
    mean_score = float(hmean(scores))
    del scores
    return mean_score, res


def load_evaluator(model_path: str, evaluation_batch_size: int,
                   corpus: List[str]) -> Tuple[GPT2Tokenizer, T5EncoderModel, int, List[str]]:
    tokenizer = GPT2Tokenizer.from_pretrained(model_path)
    model = T5EncoderModel.from_pretrained(model_path).cuda()
    model.eval()
    num_layers = len(model.encoder.block) - 1
    model.encoder.block = torch.nn.ModuleList(
        [layer for layer in model.encoder.block[:num_layers]]
    )
    return tokenizer, model, evaluation_batch_size, corpus
