from argparse import ArgumentParser
import codecs
import json
import os
import random

from datasets import load_dataset
from nltk import wordpunct_tokenize
from tqdm import tqdm


def contains_russian_letters(s: str) -> bool:
    russian_letters = set('абвгдеёжзийклмнопрстуфхцчшщъыьэюя')
    words = list(filter(lambda it: it.isalpha(), wordpunct_tokenize(s)))
    if len(words) == 0:
        return True
    return len(set(''.join(words)) & russian_letters) > 0


def main():
    parser = ArgumentParser()
    parser.add_argument('-o', '--output', dest='output_name', type=str, required=True,
                        help='The output name of the Saiga Preferences dataset.')
    args = parser.parse_args()

    output_fname = os.path.normpath(args.output_name)
    if not os.path.isfile(output_fname):
        basedir = os.path.dirname(output_fname)
        if len(basedir) > 0:
            if not os.path.isdir(basedir):
                raise IOError(f'The directory "{basedir}" does not exist!')

    system_prompts = ['Вы - Менон, разработанный Иваном Бондаренко. Вы - полезный помощник',
                      'Твоё имя - Менон. Ты создан Иваном Бондаренко и являешься полезным ассистентом.',
                      'Ты - Менон. Ты создан Иваном Бондаренко для того, чтобы помогать людям.',
                      'You are Meno, created by Ivan Bondarenko. You are a helpful assistant.']
    ds = load_dataset('IlyaGusev/saiga_preferences', split='train')
    samples = []
    for idx, (prompt, chosen, rejected) in enumerate(tqdm(zip(ds['prompt'], ds['chosen'], ds['rejected']),
                                                          total=len(ds))):
        err_msg = f'The sample {idx} is wrong!'
        if (len(chosen) != 1) or (len(rejected) != 1):
            raise IOError(err_msg)
        if (not isinstance(chosen[0], dict)) or (not isinstance(rejected[0], dict)):
            raise IOError(err_msg)
        if (chosen[0]['role'] != 'assistant') or (rejected[0]['role'] != 'assistant'):
            raise IOError(err_msg)
        response = chosen[0]['content'].replace('Сайга', 'Менон')
        rejected_response = rejected[0]['content']
        if contains_russian_letters(response):
            if len(prompt) < 1:
                raise IOError(err_msg)
            for it in prompt:
                if not isinstance(it, dict):
                    raise IOError(err_msg)
            if prompt[0]['role'] == 'system':
                system_prompt = prompt[0]['content']
                if prompt[-1]['role'] != 'user':
                    raise IOError(err_msg)
                query = prompt[-1]['content'].replace('Сайга', 'Менон')
                source_history = prompt[1:-1]
            else:
                system_prompt = ''
                if prompt[-1]['role'] != 'user':
                    raise IOError(err_msg)
                query = prompt[-1]['content'].replace('Сайга', 'Менон')
                source_history = prompt[0:-1]
            if (len(source_history) % 2) == 0:
                prepared_history = []
                ok = True
                for idx in range(len(source_history) // 2):
                    if source_history[idx * 2]['role'] != 'user':
                        ok = False
                        break
                    if source_history[idx * 2 + 1]['role'] != 'assistant':
                        ok = False
                        break
                    if not contains_russian_letters(source_history[idx * 2 + 1]['content']):
                        ok = False
                        break
                    prepared_history.append([
                        source_history[idx * 2]['content'].replace('Сайга', 'Менон'),
                        source_history[idx * 2 + 1]['content'].replace('Сайга', 'Менон')
                    ])
                if ok:
                    samples.append({
                        'system': '' if len(system_prompt) == 0 else random.choice(system_prompts),
                        'query': query,
                        'response': response,
                        'rejected_response': rejected_response,
                        'history': prepared_history
                    })

    with codecs.open(output_fname, mode='w', encoding='utf-8', errors='ignore') as fp:
        for it in samples:
            fp.write(json.dumps(it, ensure_ascii=False) + '\n')


if __name__ == '__main__':
    main()
