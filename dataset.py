from datasets import load_dataset
from torch.utils.data import Dataset
import torch

class ClassificationDataset(Dataset):
    def __init__(self, sentences, labels):
        self.sentences = sentences
        self.labels = torch.tensor(labels)

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, item):
        return self.sentences[item], self.labels[item]

#todo add conversion of labels
class BaseClassificationDataset:
    def __init__(self):
        dataset = load_dataset(self._dataset_name)

        for split, split_name in self._dict_splits.items():
            split_sentences = dataset[split_name][self._dict_fields['text']]
            # split_labels = list(map(lambda x: self._dict_labels[x], self.dataset[split]['label']))
            split_labels = dataset[split_name][self._dict_fields['label']]
            setattr(self, split, ClassificationDataset(split_sentences, split_labels))


class SST2(BaseClassificationDataset):
    _dataset_name: str = "sst2"
    _dict_labels: dict = {0: 'NEG', 1: 'POS'}
    _dict_splits: dict = {'train': 'train', 'val': 'validation'}
    _dict_fields: dict = {'text':'sentence', 'label':'label'}

class IMDB(BaseClassificationDataset):
    _dataset_name: str = "imdb"
    _dict_labels: dict = {0: 'NEG', 1: 'POS'}
    _dict_splits: dict = {'train': 'train', 'val': 'test'}
    _dict_fields: dict = {'text': 'text', 'label': 'label'}





if __name__ == '__main__':
    dataset = SST2()

    import numpy as np

    for split in ['train', 'val']:
        print('=' * 30 + split + '=' * 30)
        l = len(getattr(dataset, split))
        print(f"len = {l}")

        for _ in range(10):
            item = np.random.randint(0, l)
            print(getattr(dataset, split)[item])

