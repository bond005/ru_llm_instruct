from typing import Any

import torch


L2_REGULARIZER_WEIGHT: float = 1e-1


class EBD(torch.nn.Module):
    def __init__(self, envs_num: int, num_classes: int, device: Any, dtype: torch.dtype):
        super(EBD, self).__init__()
        self.envs_num = envs_num
        self.num_classes = num_classes
        self.device = device
        self.dtype = dtype
        self.embeddings = torch.nn.Embedding(self.envs_num, self.num_classes, dtype=self.dtype).to(self.device)
        self.re_init()

    def re_init(self):
        self.embeddings.weight.data.fill_(1.)

    def re_init_with_noise(self, noise_sd: float = 0.1):
        rd = torch.normal(
            torch.Tensor([1.0] * self.envs_num * self.num_classes),
            torch.Tensor([noise_sd] * self.envs_num * self.num_classes)
        ).to(self.dtype)
        self.embeddings.weight.data = rd.view(-1, self.num_classes).to(self.device)

    def forward(self, e: torch.Tensor) -> torch.Tensor:
        return self.embeddings(e.long())[:, None, :]


def calculate_env(x_len: int, y_len: int, mean_text_len: int) -> int:
    if x_len <= mean_text_len:
        if y_len <= mean_text_len:
            env = 0
        else:
            env = 1
    else:
        if y_len <= mean_text_len:
            env = 2
        else:
            env = 3
    return env


def calculate_environments(x_attention_mask: torch.LongTensor, y_attention_mask: torch.LongTensor,
                           mean_text_len: int) -> torch.LongTensor:
    x_len = torch.sum(x_attention_mask, dim=-1, dtype=torch.long)
    y_len = torch.sum(y_attention_mask, dim=-1, dtype=torch.long)
    sizes = x_len.size()
    if sizes != y_len.size():
        err_msg = f'The x_attention_mask does not correspond to the y_attention_mask! {sizes} != {y_len.size()}'
        raise RuntimeError(err_msg)
    if len(sizes) != 1:
        err_msg = f'The x_attention_mask has wrong dimensions! Expected 1, got {len(sizes)}.'
        raise RuntimeError(err_msg)
    batchsize = int(sizes[0])
    environments = torch.tensor(
        data=[calculate_env(x_len[idx], y_len[idx], mean_text_len) for idx in range(batchsize)],
        dtype=torch.long
    )
    return environments.long()
