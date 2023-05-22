from torch import nn
__author__ = 'Fabio'

class PaddedTransformer(nn.Module):

    def __init__(self, model, tokenizer, num_pads, device='cpu'):
        super().__init__()
        self.model = model
        self.tokenizer = tokenizer
        self.num_pads = num_pads
        self.device = device

    @property
    def num_pads(self):
        return self._num_pads

    @num_pads.setter
    def num_pads(self, value):
        self._num_pads = value

    def _pad_text(self, sentences):
        sentences = [self.tokenizer.pad_token * self.num_pads + sentence for sentence in sentences]
        inputs = self.tokenizer(sentences, return_tensors='pt', padding=True)
        inputs['attention_mask'][:, 1:self.num_pads + 1] = 0
        return inputs#{key: value.cuda() for key, value in inputs.items()}

    def forward(self, input):
        input = self._pad_text(input)
        logits = self.model(**input).logits
        return logits


if __name__ == '__main__':
    from transformers import AutoTokenizer, BertForSequenceClassification

    name = "philschmid/tiny-bert-sst2-distilled"
    model = BertForSequenceClassification.from_pretrained(name)
    tokenizer = AutoTokenizer.from_pretrained(name)

    transformer = PaddedTransformer(model, tokenizer, num_pads=0)
    s= 'I love you'
    print(transformer([s]))
    transformer.num_pads = 130
    print(transformer([s]))
