from argparse import ArgumentParser
import codecs
import csv
import gc
import logging
import os
import random
import sys

from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from tqdm import tqdm


RANDOM_SEED: int = 42
ru_stackoverflow_logger = logging.getLogger(__name__)


def main():
    random.seed(RANDOM_SEED)

    parser = ArgumentParser()
    parser.add_argument('-i', '--input', dest='input_name', type=str, required=True,
                        help='The input dataset name.')
    parser.add_argument('-o', '--output', dest='output_name', type=str, required=True,
                        help='The output name of CSV file with structured and verified RuStackoverflow.')
    parser.add_argument('-m', '--model', dest='model_name', type=str, required=True,
                        help='The DeepSeekCoder model name.')
    parser.add_argument('--maxlen', dest='maxlen', type=int, required=False, default=None,
                        help='The maximal length of the prompt.')
    args = parser.parse_args()

    input_dataset_name = os.path.normpath(args.input_name)
    if not os.path.isdir(input_dataset_name):
        raise IOError(f'The directory {input_dataset_name} does not exist!')

    model_name = os.path.normpath(args.model_name)
    if not os.path.isdir(model_name):
        raise IOError(f'The directory {model_name} does not exist!')

    output_fname = os.path.normpath(args.output_name)
    if not os.path.isfile(output_fname):
        basedir = os.path.dirname(output_fname)
        if len(basedir) > 0:
            if not os.path.isdir(basedir):
                raise IOError(f'The directory "{basedir}" does not exist!')

    dataset = load_dataset(args.input_name, split='train')
    filtered_dataset = dataset.filter(lambda it: it['answer_count'] > 0)

    print(f'There are {len(filtered_dataset)} samples in the source RuStackoverflow dataset.')

    questions_with_answers = []
    questions = [str(it).strip() for it in filtered_dataset['text_markdown']]
    answers = [([str(x1).strip() for x1 in it['text_markdown']], [int(x2) for x2 in it['score']])
               for it in filtered_dataset['answers']]
    del dataset, filtered_dataset

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    prompt = ('You are a programming specialist who also knows Russian. '
              'Please read the question and the answer to this question. '
              'Then think carefully and tell me if this answer is correct, '
              'complete enough and as clear as possible an answer to the question asked? '
              'Just write "yes" or "no", that\'s all.\n\n\n'
              'The question:\n\n\n{question}\n\n\nThe answer to this question:\n\n\n{answer}\n\n\n'
              'Your verdict on the correctness, completeness and clarity of the answer to the question '
              '(only "yes" or "no"): ')
    max_seq_len = 0
    for question_text, answers_on_question in tqdm(zip(questions, answers), total=len(questions)):
        if len(question_text) > 0:
            variants_of_answer = []
            for answer_text, answer_score in zip(answers_on_question[0], answers_on_question[1]):
                if (answer_score > 0) and (len(answer_text) > 0):
                    variants_of_answer.append(
                        {
                            'answer': answer_text,
                            'score': answer_score
                        }
                    )
            if len(variants_of_answer) > 0:
                variants_of_answer.sort(key=lambda it: (-it['score'], -len(it['answer']), it['answer']))
                messages = [
                    {
                        'role': 'user',
                        'content': prompt.format(question=question_text, answer=variants_of_answer[0]['answer'])
                    }
                ]
                tokens = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=True)
                if not isinstance(tokens, list):
                    err_msg = f'The message cannot be tokenized! {messages}'
                    ru_stackoverflow_logger.error(err_msg)
                    raise RuntimeError(err_msg)
                for cur_token in tokens:
                    if not isinstance(cur_token, int):
                        err_msg = f'The message cannot be tokenized! {messages}'
                        ru_stackoverflow_logger.error(err_msg)
                        raise RuntimeError(err_msg)
                if args.maxlen is None:
                    can_add = True
                else:
                    can_add = (len(tokens) <= args.maxlen)
                if can_add:
                    if len(tokens) > max_seq_len:
                        max_seq_len = len(tokens)
                    questions_with_answers.append(
                        {
                            'question': question_text,
                            'answer': variants_of_answer[0]['answer'],
                            'prompt_for_verification': tokenizer.decode(tokens, skip_special_tokens=True)
                        }
                    )
                del messages, tokens
            del variants_of_answer
    print(f'There are {len(questions_with_answers)} samples in the prepared RuStackoverflow dataset.')
    print(f'The maximal prompt length is {max_seq_len}.')

    model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True, torch_dtype=torch.bfloat16).cuda()


    with codecs.open(output_fname, mode='w', encoding='utf-8', errors='ignore', buffering=0) as fp:
        data_writer = csv.writer(fp, delimiter=',', quotechar='"')
        data_writer.writerow(['QUESTION', 'ANSWER', 'VERIFICATION'])
        for sample in tqdm(questions_with_answers):
            inputs = tokenizer.apply_chat_template(sample['prompt_for_verification'],
                                                   return_tensors='pt').to(model.device)
            seqlen = len(inputs[0])
            outputs = model.generate(inputs, max_new_tokens=5, do_sample=False, top_k=50, top_p=0.95,
                                     num_return_sequences=1, eos_token_id=tokenizer.eos_token_id,
                                     pad_token_id=tokenizer.eos_token_id)
            result = ' '.join(tokenizer.decode(outputs[0][seqlen:], skip_special_tokens=True).strip().split())
            data_writer.writerow([sample['question'], sample['answer'], result])
            del outputs, result, inputs
            gc.collect()
            torch.cuda.empty_cache()


if __name__ == '__main__':
    ru_stackoverflow_logger.setLevel(logging.INFO)
    fmt_str = '%(filename)s[LINE:%(lineno)d]# %(levelname)-8s ' \
              '[%(asctime)s]  %(message)s'
    formatter = logging.Formatter(fmt_str)
    stdout_handler = logging.StreamHandler(sys.stdout)
    stdout_handler.setFormatter(formatter)
    ru_stackoverflow_logger.addHandler(stdout_handler)
    file_handler = logging.FileHandler('ru_stackoverflow_filter.log')
    file_handler.setFormatter(formatter)
    ru_stackoverflow_logger.addHandler(file_handler)
    main()
