import math
from typing import List, Tuple, Optional, Union

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import torch
from transformers import LongformerForMaskedLM, LongformerTokenizerFast


MAX_TEXT_LEN: int = 16_000


def calculate_token_embeddings(texts: List[str], embedder: Tuple[LongformerTokenizerFast, LongformerForMaskedLM],
                               batch_size: Optional[int] = None) -> List[Union[np.ndarray, None]]:
    input_ids = []
    attention_mask = []
    useful_token_indices = []
    if batch_size is None:
        tokenized = embedder[0].batch_encode_plus(texts, max_length=MAX_TEXT_LEN, return_length=True,
                                                  truncation=True, padding=True, return_special_tokens_mask=True,
                                                  return_tensors='pt')
        input_ids.append(tokenized.input_ids)
        attention_mask.append(tokenized.attention_mask)
        for sample_idx in range(tokenized.special_tokens_mask.shape[0]):
            useful_token_indices_of_cur_text = []
            for time_idx in range(tokenized.special_tokens_mask.shape[1]):
                if time_idx >= tokenized.length[sample_idx]:
                    break
                mask_val = int(tokenized.special_tokens_mask[sample_idx, time_idx])
                if mask_val not in {0, 1}:
                    raise RuntimeError(f'The mask value = {mask_val} is wrong!')
                if mask_val == 0:
                    useful_token_indices_of_cur_text.append(time_idx)
            useful_token_indices.append(useful_token_indices_of_cur_text)
            del useful_token_indices_of_cur_text
        del tokenized
    else:
        if batch_size < 1:
            raise ValueError(f'The minibatch size = {batch_size} is wrong!')
        n_batches = math.ceil(len(texts) / batch_size)
        for idx in range(n_batches):
            batch_start = idx * batch_size
            batch_end = min(len(texts), batch_start + batch_size)
            tokenized = embedder[0].batch_encode_plus(texts[batch_start:batch_end], max_length=MAX_TEXT_LEN,
                                                      return_length=True, truncation=True, padding=True,
                                                      return_special_tokens_mask=True, return_tensors='pt')
            input_ids.append(tokenized.input_ids)
            attention_mask.append(tokenized.attention_mask)
            for sample_idx in range(tokenized.special_tokens_mask.shape[0]):
                useful_token_indices_of_cur_text = []
                for time_idx in range(tokenized.special_tokens_mask.shape[1]):
                    if time_idx >= tokenized.length[sample_idx]:
                        break
                    mask_val = int(tokenized.special_tokens_mask[sample_idx, time_idx])
                    if mask_val not in {0, 1}:
                        raise RuntimeError(f'The mask value = {mask_val} is wrong!')
                    if mask_val == 0:
                        useful_token_indices_of_cur_text.append(time_idx)
                useful_token_indices.append(useful_token_indices_of_cur_text)
                del useful_token_indices_of_cur_text
            del tokenized
    text_idx = 0
    embeddings = []
    for batched_input_ids, batched_attention_mask in zip(input_ids, attention_mask):
        global_attention_mask = [
            [1 if token_id == embedder[0].cls_token_id else 0 for token_id in cur_input_ids]
            for cur_input_ids in batched_input_ids
        ]
        with torch.no_grad():
            outputs = embedder[1](input_ids=batched_input_ids, attention_mask=batched_attention_mask,
                                  global_attention_mask=torch.tensor(global_attention_mask),
                                  return_dict=True)
        del global_attention_mask
        last_hidden_state = outputs.last_hidden_state.numpy()
        del outputs
        for idx in range(last_hidden_state.shape[0]):
            if len(useful_token_indices[text_idx]) > 0:
                emb_matrix = last_hidden_state[idx, useful_token_indices[text_idx], :]
                embeddings.append(emb_matrix)
            else:
                embeddings.append(None)
            text_idx += 1
        del last_hidden_state
    del input_ids, attention_mask, useful_token_indices
    return embeddings


def bert_score(references: List[str], predictions: List[str],
               evaluator: Tuple[LongformerTokenizerFast, LongformerForMaskedLM],
               batch_size: Optional[int] = None) -> List[float]:
    if len(references) != len(predictions):
        err_msg = f'The reference texts do not correspond to the predicted texts! ' \
                  f'{len(references)} != {len(predictions)}'
        raise ValueError(err_msg)
    embeddings_of_references = calculate_token_embeddings(references, evaluator, batch_size)
    embeddings_of_predictions = calculate_token_embeddings(predictions, evaluator, batch_size)
    scores = []
    for ref, pred in zip(embeddings_of_references, embeddings_of_predictions):
        if (ref is None) or (pred is None):
            if (ref is None) and (pred is None):
                scores.append(1.0)
            else:
                scores.append(0.0)
        else:
            similarity_matrix = cosine_similarity(ref, pred)
            recall = 0.0
            for ref_idx in range(ref.shape[0]):
                pred_idx = np.argmax(similarity_matrix[ref_idx, :])
                recall += float(similarity_matrix[ref_idx, pred_idx])
            recall /= float(ref.shape[0])
            precision = 0.0
            for pred_idx in range(pred.shape[0]):
                ref_idx = np.argmax(similarity_matrix[:, pred_idx])
                precision += float(similarity_matrix[ref_idx, pred_idx])
            precision /= float(pred.shape[0])
            if (precision > 0.0) and (recall > 0.0):
                f1 = 2 * precision * recall / (precision + recall)
            else:
                f1 = 0.0
            scores.append(f1)
    return scores
