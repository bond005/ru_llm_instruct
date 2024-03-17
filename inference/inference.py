from transformers import GPT2Tokenizer, GenerationConfig, T5ForConditionalGeneration


def fix_recognition_error(text: str, tokenizer: GPT2Tokenizer, config: GenerationConfig,
                          model: T5ForConditionalGeneration) -> str:
    if len(text) == 0:
        return ''
    x = tokenizer(text, return_tensors='pt', padding=True).to(model.device)
    max_size = int(x.input_ids.shape[1] * 2.0 + 10)
    min_size = 3
    if x.input_ids.shape[1] <= min_size:
        return text
    out = model.generate(**x, generation_config=config, max_length=max_size)
    res = tokenizer.decode(out[0], skip_special_tokens=True).strip()
    return ' '.join(res.split())


def generate_answer(answer: str, tokenizer: GPT2Tokenizer, config: GenerationConfig,
                    model: T5ForConditionalGeneration) -> str:
    if len(answer) == 0:
        return ''
    x = tokenizer(answer, return_tensors='pt', padding=True).to(model.device)
    out = model.generate(**x, generation_config=config)
    res = tokenizer.decode(out[0], skip_special_tokens=True).strip()
    return res.replace('\r\n', '\n')
