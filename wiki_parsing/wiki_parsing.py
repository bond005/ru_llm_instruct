import codecs
import re
from typing import Dict, List, Union


def load_wiki_file(fname: str) -> List[Dict[str, Union[str, List[Dict[str, Union[str, List[str]]]]]]]:
    doc_title: str = ''
    doc_structure: Dict[str, Union[str, List[str]]] = dict()
    all_docs: List[Dict[str, Union[str, List[Dict[str, Union[str, List[str]]]]]]] = dict()
    line_idx: int = 1
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
                        raise IOError(err_msg)
                    search_res = re_for_title.search(prep_line)
                    if search_res is None:
                        err_msg += ' The document title is not found!'
                        raise IOError(err_msg)
                    if search_res.start() < 0:
                        err_msg += ' The document title is not found!'
                        raise IOError(err_msg)
                    if search_res.end() <= search_res.start():
                        err_msg += ' The document title is not found!'
                        raise IOError(err_msg)
                    doc_title = prep_line[(search_res.start() + 7):(search_res.end() - 2)].strip()
                    if len(doc_title) == 0:
                        err_msg += ' The document title is not found!'
                        raise IOError(err_msg)
                    doc_title = ' '.join(doc_title.split())
                elif prep_line.startswith('</doc>'):
                    if len(doc_title) == 0:
                        err_msg += ' The document ends unexpectedly!'
                        raise IOError(err_msg)
                    if prep_line != '</doc>':
                        raise IOError(err_msg)
                    if len(doc_structure) > 0:
                        all_docs.append(
                            {
                                'title': doc_title,
                                'document': doc_structure
                            }
                        )
                        doc_structure.clear()
                    doc_title = ''
                else:
                    pass
            cur_line = fp.readline()
            line_idx += 1
