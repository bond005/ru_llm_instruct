import copy
from typing import List, Optional

import torch
from transformers import GPT2Tokenizer, GenerationConfig, T5ForConditionalGeneration


def fix_recognition_error(texts: List[str], tokenizer: GPT2Tokenizer, config: GenerationConfig,
                          model: T5ForConditionalGeneration, max_time: Optional[float] = None) -> List[str]:
    nonempty_texts = []
    for cur in texts:
        if len(cur.strip()) > 3:
            nonempty_texts.append(cur.strip())
    if len(nonempty_texts) == 0:
        return texts
    x = tokenizer(nonempty_texts, return_tensors='pt', padding=True).to(model.device)
    max_size = int(x.input_ids.shape[1] * 2.0 + 10)
    if max_time is None:
        config_ = config
    else:
        config_ = copy.deepcopy(config)
        config_.max_time = max_time
    out = model.generate(**x, generation_config=config_, max_length=max_size)
    del x, config_
    results_for_nonempty_texts = [
        ' '.join(tokenizer.decode(cur, skip_special_tokens=True).strip().split()) for cur in out
    ]
    del out
    torch.cuda.empty_cache()
    united_results = []
    idx = 0
    for cur in texts:
        if len(cur.strip()) > 3:
            united_results.append(results_for_nonempty_texts[idx])
            idx += 1
        else:
            united_results.append(cur.strip())
    return united_results


def generate_answer(questions: List[str], tokenizer: GPT2Tokenizer, config: GenerationConfig,
                    model: T5ForConditionalGeneration, max_time: Optional[float] = None) -> List[str]:
    nonempty_questions = []
    for cur in questions:
        if len(cur.strip()) > 0:
            nonempty_questions.append(cur)
    if len(nonempty_questions) == 0:
        return ['' for _ in range(len(questions))]
    if max_time is None:
        config_ = config
    else:
        config_ = copy.deepcopy(config)
        config_.max_time = max_time
    x = tokenizer(nonempty_questions, return_tensors='pt', padding=True).to(model.device)
    out = model.generate(**x, generation_config=config_)
    del x, config_
    answers_for_nonempty_texts = [
        tokenizer.decode(cur, skip_special_tokens=True).strip().replace('\r\n', '\n') for cur in out
    ]
    del out
    torch.cuda.empty_cache()
    united_answers = []
    idx = 0
    for cur in questions:
        if len(cur.strip()) > 0:
            united_answers.append(answers_for_nonempty_texts[idx])
            idx += 1
        else:
            united_answers.append('')
    return united_answers
