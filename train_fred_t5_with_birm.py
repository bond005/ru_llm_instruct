from argparse import ArgumentParser
import codecs
import json
import logging
import os
import random
import sys
from typing import List, Tuple

import numpy as np
from tqdm import tqdm, trange
from transformers import T5ForConditionalGeneration, GPT2Tokenizer, Adafactor, GenerationConfig
from transformers import LongformerTokenizerFast, LongformerModel
import torch

from instructions.instructions import evaluate
from training.training import sample_batch
from training.training import training_logger
from birm.birm import EBD, calculate_environments, split_by_environments


fredt5_logger = logging.getLogger(__name__)
L2_REGULARIZER_WEIGHT: float = 1.0


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
    parser.add_argument('--eval_model', dest='eval_model', type=str, required=False,
                        default='kazzand/ru-longformer-tiny-16384',
                        help='The Longformer model for BERT score.')
    parser.add_argument('--accum', dest='gradient_accumulation', type=int, required=False, default=None,
                        help='The gradient accumulation for FRED-T5 training.')
    parser.add_argument('--lr', dest='learning_rate', type=float, required=False, default=3e-4,
                        help='The learning rate.')
    parser.add_argument('--trainsize', dest='trainsize', type=int, required=False, default=None,
                        help='The samples per training epoch.')
    parser.add_argument('--penalty', dest='birm_penalty', type=float, required=False, default=10000.0,
                        help='The penalty weight for BIRM.')
    parser.add_argument('--samples', dest='birm_samples', type=int, required=False, default=5,
                        help='The samples number for BIRM.')
    parser.add_argument('--bf16', dest='bf16', action='store_true', help='Is bfloat16 used?')
    args = parser.parse_args()

    if args.birm_samples < 2:
        err_msg = f'The samples number for BIRM is wrong! Expected an integer greater than 1, got {args.birm_samples}.'
        fredt5_logger.error(err_msg)
        raise ValueError(err_msg)

    if args.birm_penalty <= 0.0:
        err_msg = (f'The penalty weight for BIRM is wrong! Expected a non-negative floating-point, '
                   f'got {args.birm_penalty}.')
        fredt5_logger.error(err_msg)
        raise ValueError(err_msg)

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
        fredt5_logger.info(f'There are {len(data_for_validation[cur_task])} validation samples for task {cur_task}.')

    if args.bf16:
        model = T5ForConditionalGeneration.from_pretrained(pretrained_dir_name, torch_dtype=torch.bfloat16).to(device)
        ebd = EBD(envs_num=4, num_classes=model.config.vocab_size, device=device, dtype=torch.bfloat16)
    else:
        model = T5ForConditionalGeneration.from_pretrained(pretrained_dir_name, torch_dtype=torch.float32).to(device)
        ebd = EBD(envs_num=4, num_classes=model.config.vocab_size, device=device, dtype=torch.float32)
    model.eval()
    loss_fct = torch.nn.CrossEntropyLoss().to(device)
    fredt5_logger.info(f'The pre-trained model "{os.path.basename(pretrained_dir_name)}" is loaded.')

    llm_total_params = sum(p.numel() for p in model.parameters())
    with torch.no_grad():
        weight_norm = torch.tensor(0.).to(device)
        for w in model.parameters():
            weight_norm += w.norm().pow(2)
        weight_norm_val = float(weight_norm.detach().cpu())
    n = 1.0
    weight_norm_val_ = weight_norm_val
    while weight_norm_val_ > 1.0:
        n *= 10.0
        weight_norm_val_ /= 10.0
    l2_regularizer_weight = L2_REGULARIZER_WEIGHT / n
    info_msg = (f'Total number of the FRED-T5\'s weights is {llm_total_params}, '
                f'the weight norm is {weight_norm_val}, and L2 regularizer weight is {l2_regularizer_weight}.')
    fredt5_logger.info(info_msg)

    tokenizer.save_pretrained(finetuned_dir_name)
    all_text_lengths = sorted([len(tokenizer.tokenize(it)) for it in tqdm(united_text_corpus)])
    max_text_len = all_text_lengths[-1]
    median_text_len = all_text_lengths[(len(all_text_lengths) - 1) // 2]
    mean_text_len = round(sum(all_text_lengths) / len(all_text_lengths))
    fredt5_logger.info(f'The maximal subwords in the text is {max_text_len}.')
    fredt5_logger.info(f'The minimal subwords in the text is {all_text_lengths[0]}.')
    fredt5_logger.info(f'The median subwords in the text is {median_text_len}.')
    fredt5_logger.info(f'The mean subwords in the text is {mean_text_len}.')

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

    data_for_training = split_by_environments(data_for_training_, mean_text_len)
    del data_for_training_
    info_msg = ''
    for env in sorted(list(data_for_training.keys())):
        info_msg += f' environment {env}: {len(data_for_training[env])} samples,'
    info_msg = info_msg.strip()
    if info_msg.endswith(','):
        info_msg = info_msg[:-1] + '.'
    info_msg = info_msg[0].upper() + info_msg[1:]
    fredt5_logger.info(info_msg)

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

    scorer = (
        LongformerTokenizerFast.from_pretrained(args.eval_model),
        LongformerModel.from_pretrained(args.eval_model)
    )

    try:
        best_score, results_by_tasks = evaluate(data_for_validation,
                                                tokenizer, generation_config, model, eval_minibatch_size, scorer)
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
            info_msg += 'BERT score F1 is {0:.6f}.'.format(instant_score)
        fredt5_logger.info(info_msg)

    for epoch in range(1, max_epochs + 1):
        fredt5_logger.info(f'Epoch {epoch} is started.')
        env_freq = dict()
        model.train()
        total_training_loss_val = 0.0
        training_nll_val = 0.0
        weight_norm_val = 0.0
        training_penalty_val = 0.0
        for _ in trange(n_training_batches):
            try:
                x_input_ids, x_attention_mask, y_input_ids, y_attention_mask = sample_batch(
                    data_for_training,
                    tokenizer.pad_token_id,
                    minibatch_size * gradient_accumulation,
                    warn=False
                )
            except Exception as err:
                fredt5_logger.error(str(err))
                raise
            try:
                envs = calculate_environments(x_attention_mask, y_attention_mask, mean_text_len)
            except Exception as err:
                fredt5_logger.error(str(err))
                raise
            for val in envs.numpy().tolist():
                if val in env_freq:
                    env_freq[val] += 1
                else:
                    env_freq[val] = 1
            for batch_idx in range(gradient_accumulation):
                batch_start = batch_idx * minibatch_size
                batch_end = batch_start + minibatch_size
                train_labels = y_input_ids[batch_start:batch_end].to(device)
                res = model(
                    input_ids=x_input_ids[batch_start:batch_end].to(device),
                    attention_mask=x_attention_mask[batch_start:batch_end].to(device),
                    labels=y_input_ids[batch_start:batch_end].to(device),
                    decoder_attention_mask=y_attention_mask[batch_start:batch_end].to(device),
                    return_dict=True
                )
                train_nll = res.loss
                training_nll_val += float(train_nll.detach().cpu())
                if args.bf16:
                    train_penalty = torch.tensor(0.).to(device).bfloat16()
                else:
                    train_penalty = torch.tensor(0.).to(device)
                for _ in range(args.birm_samples):
                    ebd.re_init_with_noise(0.1)
                    env_embeddings = ebd(envs[batch_start:batch_end].to(device))
                    train_logits_w = env_embeddings * res.logits
                    train_nll_ = loss_fct(
                        train_logits_w.view(-1, model.config.vocab_size),
                        train_labels.reshape(-1)
                    )
                    grad = torch.autograd.grad(
                        train_nll_ * ebd.envs_num,
                        ebd.parameters(),
                        create_graph=True
                    )[0]
                    train_penalty += (1.0 / args.birm_samples) * torch.mean(grad ** 2)
                    del train_logits_w, env_embeddings
                train_penalty *= args.birm_penalty
                training_penalty_val += float(train_penalty.detach().cpu())
                if args.bf16:
                    weight_norm = torch.tensor(0.).to(device).bfloat16()
                else:
                    weight_norm = torch.tensor(0.).to(device)
                for w in model.parameters():
                    weight_norm += w.norm().pow(2)
                weight_norm *= l2_regularizer_weight
                weight_norm_val += float(weight_norm.detach().cpu())
                loss = train_nll.clone()
                loss += weight_norm
                loss += train_penalty
                loss /= gradient_accumulation
                total_training_loss_val += float(loss.detach().cpu())
                loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            del envs
        total_training_loss_val /= float(n_training_batches)
        training_nll_val /= float(n_training_batches * gradient_accumulation)
        weight_norm_val /= float(n_training_batches * gradient_accumulation)
        training_penalty_val /= float(n_training_batches * gradient_accumulation)
        info_msg = (f'Epoch {epoch}: total training loss is {total_training_loss_val}, '
                    f'training cross-entropy is {training_nll_val}, invariance penalty is {training_penalty_val}, '
                    f'weight norm penalty is {weight_norm_val}.')
        fredt5_logger.info(info_msg)
        if len(env_freq) > 1:
            info_msg = f'Epoch {epoch}: {len(env_freq)} environments are used. They are: '
            env_keys = sorted(
                list(env_freq.keys()),
                key=lambda it: -env_freq[it]
            )
            total_sum = env_freq[env_keys[0]]
            for k in env_keys[1:]:
                total_sum += env_freq[k]
            info_msg += f'environment {env_keys[0]} = {round(100.0 * env_freq[env_keys[0]] / total_sum)}%'
            for k in env_keys[1:]:
                info_msg += f', environment {k} = {round(100.0 * env_freq[k] / total_sum)}%'
            info_msg += '.'
        else:
            info_msg = f'Epoch {epoch}: only environment {list(env_freq.keys())[0]} is used.'
        fredt5_logger.info(info_msg)
        model.eval()
        torch.cuda.empty_cache()
        try:
            cur_score, results_by_tasks = evaluate(data_for_validation,
                                                   tokenizer, generation_config, model, eval_minibatch_size, scorer)
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
                info_msg += 'BERT score F1 is {1:.6f}.'.format(epoch, instant_score)
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
    file_handler = logging.FileHandler('birm_fredt5_instruct_training.log')
    file_handler.setFormatter(formatter)
    fredt5_logger.addHandler(file_handler)
    training_logger.addHandler(file_handler)
    main()
