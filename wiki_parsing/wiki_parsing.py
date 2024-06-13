import codecs
import re
from typing import Dict, List, Union

from nltk import wordpunct_tokenize


MAX_WORDS_IN_TITLE: int = 7


def remove_wiki_links(source_text: str) -> str:
    prepared_text = ' '.join(source_text.replace(' ', ' ').strip().split())
    re_for_link = re.compile(r'\[\[.*?\]\]')
    search_res = re_for_link.search(prepared_text)
    if search_res is None:
        return prepared_text
    if search_res.start() < 0:
        return prepared_text
    if search_res.end() <= search_res.start():
        return prepared_text
    text_of_link = prepared_text[(search_res.start() + 2):(search_res.end() - 2)].strip()
    if len(text_of_link) == 0:
        prepared_text = prepared_text[:search_res.start()] + prepared_text[search_res.end():]
    else:
        found_idx = text_of_link.find('|')
        if found_idx >= 0:
            text_of_link = text_of_link[(found_idx + 1):].strip()
        prepared_text = prepared_text[:search_res.start()] + text_of_link + prepared_text[search_res.end():]
    prepared_text = ' '.join(prepared_text.strip().split())
    search_res = re_for_link.search(prepared_text)
    while True:
        if search_res is None:
            break
        if search_res.start() < 0:
            break
        if search_res.end() <= search_res.start():
            break
        text_of_link = prepared_text[(search_res.start() + 2):(search_res.end() - 2)].strip()
        if len(text_of_link) == 0:
            prepared_text = prepared_text[:search_res.start()] + prepared_text[search_res.end():]
        else:
            found_idx = text_of_link.find('|')
            if found_idx >= 0:
                text_of_link = text_of_link[(found_idx + 1):].strip()
            prepared_text = prepared_text[:search_res.start()] + text_of_link + prepared_text[search_res.end():]
        prepared_text = ' '.join(prepared_text.strip().split())
        search_res = re_for_link.search(prepared_text)
    return prepared_text


def load_wiki_file(fname: str) -> List[Dict[str, Union[str, List[Dict[str, Union[str, List[str]]]]]]]:
    doc_title: str = ''
    section_title: str = ''
    section_paragraphs: List[str] = []
    doc_structure: List[Dict[str, Union[str, List[str]]]] = []
    all_docs: List[Dict[str, Union[str, List[Dict[str, Union[str, List[str]]]]]]] = []
    line_idx: int = 1
    prep_line: str = ''
    err_msg = f'{fname}: The line {line_idx} is wrong!'
    re_for_title = re.compile(r'title=".+">$')
    with codecs.open(fname, mode='r', encoding='utf-8', errors='ignore') as fp:
        cur_line = fp.readline()
        while len(cur_line) > 0:
            prep_line = cur_line.strip()
            if len(prep_line) > 0:
                err_msg = f'{fname}: The line {line_idx} is wrong!'
                if prep_line.startswith('<doc id='):
                    if len(doc_title) > 0:
                        err_msg += ' The previous document is not finalized!'
                        raise IOError(err_msg + f' {prep_line}')
                    search_res = re_for_title.search(prep_line)
                    if search_res is None:
                        err_msg += ' The document title is not found!'
                        raise IOError(err_msg + f' {prep_line}')
                    if search_res.start() < 0:
                        err_msg += ' The document title is not found!'
                        raise IOError(err_msg + f' {prep_line}')
                    if search_res.end() <= search_res.start():
                        err_msg += ' The document title is not found!'
                        raise IOError(err_msg + f' {prep_line}')
                    doc_title = prep_line[(search_res.start() + 7):(search_res.end() - 2)].strip()
                    if len(doc_title) == 0:
                        err_msg += ' The document title is not found!'
                        raise IOError(err_msg + f' {prep_line}')
                    doc_title = ' '.join(doc_title.split())
                elif prep_line.startswith('</doc>'):
                    if len(doc_title) == 0:
                        err_msg += ' The document ends unexpectedly!'
                        raise IOError(err_msg + f' {prep_line}')
                    if prep_line != '</doc>':
                        raise IOError(err_msg + f' {prep_line}')
                    if len(section_paragraphs) > 0:
                        doc_structure.append({
                            'section_title': section_title,
                            'section_body': section_paragraphs
                        })
                        del section_paragraphs
                        section_paragraphs: List[str] = []
                    if len(doc_structure) > 0:
                        all_docs.append(
                            {
                                'title': doc_title,
                                'document': doc_structure
                            }
                        )
                        del doc_structure
                        doc_structure: List[Dict[str, Union[str, List[str]]]] = []
                    doc_title = ''
                    section_title = ''
                else:
                    words = list(filter(lambda x: x.isalnum(), wordpunct_tokenize(prep_line)))
                    is_title = (len(words) <= MAX_WORDS_IN_TITLE)
                    if is_title:
                        if len(section_paragraphs) > 0:
                            doc_structure.append({
                                'section_title': section_title,
                                'section_body': section_paragraphs
                            })
                            del section_paragraphs
                            section_paragraphs: List[str] = []
                            section_title = remove_wiki_links(prep_line)
                        else:
                            if len(section_title) > 0:
                                section_paragraphs.append(section_title)
                            else:
                                section_title = remove_wiki_links(prep_line)
                    else:
                        if len(section_paragraphs) == 0:
                            if len(section_title) == 0:
                                section_title = remove_wiki_links(prep_line)
                            else:
                                section_paragraphs.append(remove_wiki_links(prep_line))
                        else:
                            if len(section_title) == 0:
                                err_msg += ' The section title is not found!'
                                raise IOError(err_msg + f' {prep_line}')
                            section_paragraphs.append(remove_wiki_links(prep_line))
            cur_line = fp.readline()
            line_idx += 1
    if len(doc_title) > 0:
        err_msg += ' The previous document is not finalized!'
        raise IOError(err_msg + f' {prep_line}')
    return all_docs
