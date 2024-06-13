from argparse import ArgumentParser
import codecs
import json
import os
import random
from typing import Dict, List, Union

from tqdm import tqdm
from transformers import GPT2Tokenizer

from wiki_parsing.wiki_parsing import load_wiki_file


def select_text_files(dirname: str) -> List[str]:
    all_items = list(map(
        lambda it2: os.path.join(dirname, it2),
        filter(
            lambda it1: it1 not in {'.', '..'},
            os.listdir(dirname)
        )
    ))
    if len(all_items) == 0:
        return []
    all_files = list(filter(lambda it: os.path.isfile(it) and os.path.basename(it).startswith('wiki_'), all_items))
    all_subdirs = list(filter(lambda it: os.path.isdir(it), all_items))
    for cur_subdir in all_subdirs:
        all_files += select_text_files(cur_subdir)
    return all_files


def article_to_tokens(article: List[Dict[str, Union[str, List[str]]]], tokenizer: GPT2Tokenizer) -> List[str]:
    united_text = ''
    for section in article:
        united_text += f' {section["section_title"]}\n'
        for paragraph in section['section_body']:
            united_text += f' {paragraph}\n'
    new_united_text = united_text.replace(' \n', '\n').replace('\n ', '\n')
    while new_united_text != united_text:
        united_text = new_united_text
        new_united_text = united_text.replace(' \n', '\n').replace('\n ', '\n')
    united_text = new_united_text.strip()
    return tokenizer.tokenize(united_text)


def main():
    random.seed(42)

    parser = ArgumentParser()
    parser.add_argument('-i', '--input', dest='input_name', type=str, required=True,
                        help='The input path to the parsed Wikipedia.')
    parser.add_argument('-o', '--output', dest='output_name', type=str, required=True,
                        help='The output name of JSON file with structured Wikipedia articles.')
    parser.add_argument('-m', '--model', dest='large_language_model', type=str, required=True,
                        help='The name of large language model (some variant of FRED-T5).')
    parser.add_argument('-n', '--number', dest='articles_number', type=int, required=True,
                        help='The number of selected articles.')
    parser.add_argument('--maxtokens', dest='maxtokens', type=int, required=False, default=10_000,
                        help='The maximal number of tokens per article.')
    parser.add_argument('--mintokens', dest='mintokens', type=int, required=False, default=1_000,
                        help='The minimal number of tokens per article.')
    args = parser.parse_args()

    wiki_dir = os.path.normpath(args.input_name)
    if not os.path.isdir(wiki_dir):
        raise IOError(f'The directory "{wiki_dir}" does not exist!')

    output_fname = os.path.normpath(args.output_name)
    if not os.path.isfile(output_fname):
        basedir = os.path.dirname(output_fname)
        if not os.path.isdir(basedir):
            raise IOError(f'The directory "{basedir}" does not exist!')

    if args.mintokens < 0:
        raise ValueError(f'The minimal number of tokens per article must be a non-negative integer!')
    if args.maxtokens <= 0:
        raise ValueError(f'The maximal number of tokens per article must be a positive integer!')
    if args.maxtokens <= args.mintokens:
        raise ValueError(f'The maximal number of tokens per article must be greater than the minimal one!')

    tokenizer = GPT2Tokenizer.from_pretrained(args.large_language_model)

    all_wiki_files = sorted(select_text_files(wiki_dir))
    print(f'There are {len(all_wiki_files)} Wiki files are found!')

    articles = []
    article_names = set()
    for cur_wiki_fname in tqdm(all_wiki_files):
        loaded = load_wiki_file(cur_wiki_fname)
        for val in loaded:
            if val['title'] in article_names:
                raise RuntimeError(f'The article {val["title"]} is duplicated!')
            article_names.add(val['title'])
            articles.append(val)
    del article_names
    print(f'There are {len(articles)} articles are loaded.')

    filtered_articles = []
    for val in tqdm(articles):
        article_tokens = article_to_tokens(val['document'], tokenizer)
        if (len(article_tokens) >= args.mintokens) and (len(article_tokens) <= args.maxtokens):
            filtered_articles.append(val)
    print(f'There are {len(filtered_articles)} articles are filtered by length.')

    if len(filtered_articles) > args.articles_number:
        filtered_articles = random.sample(population=filtered_articles, k=args.articles_number)

    with codecs.open(output_fname, mode='w', encoding='utf-8', errors='ignore') as fp:
        json.dump(obj=filtered_articles, fp=fp, ensure_ascii=False, indent=4)


if __name__ == '__main__':
    main()
