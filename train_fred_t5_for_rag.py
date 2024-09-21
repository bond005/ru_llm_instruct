from argparse import ArgumentParser
import gc
import logging
import os
import random
import sys
from typing import List, Optional, Tuple
import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)

import bitsandbytes as bnb
from datasets import load_dataset
import numpy as np
from tqdm import trange
from transformers import GPT2Tokenizer, GenerationConfig
from transformers import LongformerTokenizerFast, LongformerForMaskedLM
from transformers import Adafactor
from turbot5 import T5ForConditionalGeneration
import torch

from instructions.instructions import evaluate_any_task


fredt5_training_logger = logging.getLogger(__name__)
MAX_GENERATION_TIME: float = 5.0
RANDOM_SEED: int = 42


def prepare_sample_for_rag(question: str, context: str, true_answer: str) -> Tuple[str, str]:
    random_val = random.random()
    if random_val > 0.5:
        if random.random() > 0.5:
            input_template = ('Сгенерируй ответ на вопрос по тексту.'
                              '\n\nТекст:\n\n{context}\n\nВопрос:\n\n{question}Ответ:\n\n')
        else:
            input_template = ('Сгенерируй ответ на вопрос по тексту.'
                              '\n\nВопрос:\n\n{question}Текст:\n\n{context}\n\nОтвет:\n\n')
    elif random_val > 0.333:
        if random.random() > 0.5:
            input_template = '{context}\n\n{question}\n\n'
        else:
            input_template = '{question}\n\n{context}\n\n'
    elif random_val > 0.167:
        if random.random() > 0.5:
            input_template = '{context}\n{question}\n'
        else:
            input_template = '{question}\n{context}\n'
    else:
        if random.random() > 0.5:
            input_template = '{context} {question} '
        else:
            input_template = '{question} {context} '
    new_instruction = input_template.format(question=question, context=context)
    return new_instruction, true_answer


def generate_samples_for_minibatch(rag_data_training: List[Tuple[str, str, str]],
                                   additional_data_for_training: Optional[List[Tuple[str, str]]],
                                   tokenizer: GPT2Tokenizer, minibatch_size: int) -> List[Tuple[List[int], List[int]]]:
    samples_for_batch = []
    if additional_data_for_training is not None:
        selected_samples = random.sample(
            population=additional_data_for_training,
            k=minibatch_size // 2
        )
        for current_sample in selected_samples:
            samples_for_batch.append((
                tokenizer.encode(current_sample[0]),
                tokenizer.encode(current_sample[1])
            ))
        del selected_samples
    selected_samples = random.sample(
        population=rag_data_training,
        k=minibatch_size - len(samples_for_batch)
    )
    for question, context, target in selected_samples:
        sample = prepare_sample_for_rag(question, context, target)
        samples_for_batch.append((
            tokenizer.encode(sample[0]),
            tokenizer.encode(sample[1])
        ))
        del sample
    del selected_samples
    random.shuffle(samples_for_batch)
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
    parser.add_argument('--alg', dest='algorithm', type=str, required=False,
                        choices=['rmsprop', 'adamw', 'sgd', 'adam8bit', 'adafactor'],
                        default='adafactor', help='The training algorithm (RMSprop, AdamW, SGD, Adafactor, '
                                                  'or Adam8Bit).')
    parser.add_argument('--eval_batch', dest='eval_batch_size', type=int, required=False, default=None,
                        help='The mini-batch size for FRED-T5 evaluation.')
    parser.add_argument('--eval_model', dest='eval_model', type=str, required=False,
                        default='kazzand/ru-longformer-tiny-16384',
                        help='The Longformer model for BERT score.')
    parser.add_argument('--lr', dest='learning_rate', type=float, required=False, default=3e-4,
                        help='The learning rate.')
    parser.add_argument('--maxtokens', dest='maxtokens', type=int, required=False, default=None,
                        help='The maximal number of tokens for the training inputs.')
    parser.add_argument('--eval_maxtokens', dest='eval_maxtokens', type=int, required=False, default=None,
                        help='The maximal number of tokens for the evaluation inputs.')
    parser.add_argument('--iters', dest='iters_per_epoch', type=int, required=False, default=None,
                        help='The iterations per epoch.')
    parser.add_argument('--no_pre_eval', dest='no_pre_eval', action='store_true', required=False,
                        help='The preliminary evaluation (before training) is not realized.')
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

    scorer = (
        LongformerTokenizerFast.from_pretrained(args.eval_model),
        LongformerForMaskedLM.from_pretrained(args.eval_model)
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

    try:
        trainset = load_dataset(dataset_path, split='train')
    except Exception as err:
        fredt5_training_logger.error((str(err)))
        raise
    fredt5_training_logger.info(f'There are {len(trainset)} samples in the training set.')
    if args.maxtokens is not None:
        trainset = trainset.filter(
            lambda it: ((len(tokenizer.tokenize(it['question'])) + len(tokenizer.tokenize(it['context']))) <= args.maxtokens) and
                       (len(tokenizer.tokenize(it['response'])) < max(3, args.maxtokens // 2))
        )
        fredt5_training_logger.info(f'There are {len(trainset)} samples in the training set after filtering.')

    try:
        valset = load_dataset(dataset_path, split='validation')
    except Exception as err:
        fredt5_training_logger.error((str(err)))
        raise
    fredt5_training_logger.info(f'There are {len(valset)} samples in the validation set.')
    if args.eval_maxtokens is not None:
        valset = valset.filter(
            lambda it: ((len(tokenizer.tokenize(it['question'])) + len(tokenizer.tokenize(it['context']))) <= args.eval_maxtokens) and
                       (len(tokenizer.tokenize(it['response'])) < max(3, args.eval_maxtokens // 2))
        )
        fredt5_training_logger.info(f'There are {len(valset)} samples in the validation set after filtering.')

    questions = [str(it) for it in trainset['question']]
    documents = [str(it) for it in trainset['context']]
    answers = [str(it) for it in trainset['response']]
    del trainset
    rag_trainset = list(zip(questions, documents, answers))
    del questions, documents, answers

    questions = [str(it) for it in valset['question']]
    documents = [str(it) for it in valset['context']]
    answers = [str(it) for it in valset['response']]
    del valset
    rag_valset = list(zip(questions, documents, answers))
    del questions, documents, answers

    n_training_samples = len(rag_trainset)
    if len(additional_datasets) > 0:
        additional_inputs = []
        additional_targets = []
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
                    lambda it: (len(tokenizer.tokenize(it['input'])) <= args.maxtokens) and
                               (len(tokenizer.tokenize(it['target'])) <= max(3, args.maxtokens // 2))
                )
                info_msg = f'There are {len(additional_trainset)} samples in the additional training set ' \
                           f'{os.path.basename(it)} after filtering.'
                fredt5_training_logger.info(info_msg)
            additional_inputs += [str(it) for it in additional_trainset['input']]
            additional_targets += [str(it) for it in additional_trainset['target']]
            del additional_trainset
        additional_trainset = list(zip(additional_inputs, additional_targets))
        n_training_samples += len(additional_trainset)
    else:
        additional_trainset = None

    gc.collect()

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

    max_text_len = max([len(it[-1]) for it in rag_trainset] + [len(it[-1]) for it in rag_valset])
    if additional_trainset is not None:
        max_text_len = max([len(it[-1]) for it in additional_trainset] + [max_text_len])
    fredt5_training_logger.info(f'The maximal subwords in the generated text is {max_text_len}.')

    generation_max_length = 3 + round(1.1 * max_text_len)
    generation_config = GenerationConfig(
        top_k=20,
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
    elif args.algorithm == 'adafactor':
        optimizer = Adafactor(
            params=[p for p in model.parameters() if p.requires_grad],
            lr=args.learning_rate,
            scale_parameter=False,
            relative_step=False,
            warmup_init=False,
            clip_threshold=1.0
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
        try:
            best_score, results_by_tasks = evaluate_any_task([prepare_sample_for_rag(*it) for it in rag_valset],
                                                             tokenizer, generation_config, model,
                                                             eval_minibatch_size, scorer,
                                                             max_time=MAX_GENERATION_TIME)
        except Exception as err:
            fredt5_training_logger.error(str(err))
            raise
        del results_by_tasks
        fredt5_training_logger.info(f'Before training: the BERT score is {round(best_score, 6)}.')
        torch.cuda.empty_cache()
        gc.collect()

    for epoch in range(1, max_epochs + 1):
        fredt5_training_logger.info(f'Epoch {epoch} is started.')
        model.train()
        total_training_loss_val = 0.0
        for _ in trange(n_training_batches):
            samples_in_batch = generate_samples_for_minibatch(
                rag_trainset, additional_trainset, tokenizer,
                minibatch_size=minibatch_size
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
        total_training_loss_val /= float(n_training_batches)
        info_msg = f'Epoch {epoch}: total training loss is {total_training_loss_val}.'
        fredt5_training_logger.info(info_msg)
        model.eval()
        gc.collect()
        torch.cuda.empty_cache()
        try:
            new_score, results_by_tasks = evaluate_any_task([prepare_sample_for_rag(*it) for it in rag_valset],
                                                             tokenizer, generation_config, model,
                                                             eval_minibatch_size, scorer,
                                                             max_time=MAX_GENERATION_TIME)
        except Exception as err:
            fredt5_training_logger.error(str(err))
            raise
        del results_by_tasks
        fredt5_training_logger.info(f'Epoch {epoch}: the BERT score is {round(new_score, 6)}.')
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
