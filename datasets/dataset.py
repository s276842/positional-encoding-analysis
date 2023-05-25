from datasets import load_dataset
from datasets import TextClassification, Features, ClassLabel

class Dataset():

    def __init__(
            self,
            dataset_name: str,
            config_name: str = None,
    ):
        
        """
        dataset_name    :   Name of the dataset from the HuggingFace hub.

        """
        ## TODO

        self.dataset_name = dataset_name
        self.dataset = load_dataset(dataset_name, config_name)
        self.textual_fields = []
        self.label_fields = []

        for k, v in self.dataset.features.items():
            if type(v) == ClassLabel:
                self.label_fields.append(k)
            elif v.dtype == "string":
                self.textual_fields.append(k)

        # Infer data
        if len(self.textual_fields) != 1:
            pass