from typing import List

import torch.cuda

from sklearn.metrics import precision_recall_fscore_support
from transformers import AutoModelForSequenceClassification
from model import PaddedTransformer
from datasets import Dataset
import pandas as pd

from torch.utils.data import DataLoader
from torch.nn import CrossEntropyLoss

class Task():
    def __init__(
        self,
        padded_transformer: PaddedTransformer,
        dataset: Dataset,

        task: str,
        metrics: List[str]
    ):
        pass



class SequenceClassification:
    criterion = CrossEntropyLoss()

    def __init__(self, model: PaddedTransformer, dataset: Dataset):
        self.model = model
        self.dataset = dataset

    def inference(self, split='val', batch_size=16, num_max_pad=450):

        dataset = getattr(self.dataset, split)
        dataloader = DataLoader(dataset, batch_size=batch_size)
        dataset_dim = len(dataset)
        df_results = pd.DataFrame(columns = ['num_pads', 'precision', 'recall', 'fscore', 'cross_entropy_loss'])

        for num_pads in range(num_max_pad):
            self.model.num_pads = num_pads
            y_true = []
            y_pred = []
            tot_loss = 0

            for sentences, labels in dataloader:

                y_true += labels

                with torch.no_grad():
                    logits = self.model(sentences).cpu()
                    loss = self.criterion(logits, labels)
                    predicted_class_ids = torch.where(torch.softmax(logits, -1) > 0.5)[1].numpy().tolist()

                    y_pred += predicted_class_ids
                    tot_loss += loss.item()

            scores = precision_recall_fscore_support(y_true, y_pred, average='macro')
            df_results.loc[num_pads] = {'num_pads': num_pads, 'precision': scores[0], 'recall':scores[1], 'fscore':scores[2], 'cross_entropy_loss': tot_loss}

            print(f" [{num_pads}] prec/rec/fscore/supp = {scores} cross_entropy_loss = {tot_loss:.3f}")

        return df_results
