from __future__ import annotations
from transformers import T5Tokenizer
import torch

class CustomTokenizer(T5Tokenizer):
    _num_pads: int = 40

def __init__(self, **kwargs):
    super().__init__(**kwargs)

def __call__(self, *args, **kwargs):
    return print('something, but')
    # tok = super().__call__(*args, **kwargs)
    # input_ids = tok['input_ids']

    # input_ids = torch.tensor([[super().pad_token_id] * self.num_pads + input_ids[1:]])
    # attention_mask = torch.ones(1, len(input_ids[0]))
    # attention_mask[:, :1 + self.num_pads] = 0

    # return {'input_ids': input_ids, 'attention_mask': attention_mask}



if __name__ == '__main__':
    tokenizer = CustomTokenizer.from_pretrained('t5-small')
    print(tokenizer._num_pads)
    tokenizer.encode_plus('this is a test')
    # tokenizer.num_pads = 23
    # print(tokenizer('this is a test'))
    # print(tokenizer.__getattribute__('num_pads'))